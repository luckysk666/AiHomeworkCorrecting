"""
阿里云OCR识别模块

功能：
1. 调用阿里云RecognizeEduPaperCut（试卷切题）API — 提取题干区域坐标
2. 使用多模态大模型（Qwen-VL）直接理解图片中的手写题目和答案
3. 支持单张识别 + 批量识别 + JSON结果导出

接口说明（供A、D同学调用）：
    from aliyun_ocr import ocr_recognize
    question_text, answer_text = await ocr_recognize(processed_img)
    # processed_img: A同学 image_process.py 处理后的 numpy.ndarray (OpenCV格式)

依赖：pip install alibabacloud_ocr_api20210707 opencv-python numpy requests
"""

import json
import time
import asyncio
import base64
import argparse
import re
import requests
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sys
import io
import cv2
import numpy as np

# 阿里云OCR SDK
from alibabacloud_ocr_api20210707.client import Client as OCRClient
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models

if sys.platform == 'win32':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ============================================================
# 配置
# ============================================================
ALIYUN_ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
ALIYUN_ACCESS_KEY_SECRET = "YOUR_ACCESS_KEY_SECRET"
ALIYUN_OCR_ENDPOINT = "ocr-api.cn-hangzhou.aliyuncs.com"
ALIYUN_REGION_ID = "cn-hangzhou"

QWEN_API_KEY = "sk-36a2641b1d174f5680dff25f537ec022"
QWEN_MODEL = "qwen-vl-plus"
QWEN_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 3
RETRY_DELAY = 1.0
CONNECT_TIMEOUT_MS = 10000
READ_TIMEOUT_MS = 60000

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# 统一的识别提示词模板
UNIFIED_PROMPT = """你是专业的数学作业批改老师,擅长识别各种数学题目和手写答案。

【识别要求】
请仔细观察图片,完整识别:
1. 题目原文(包括所有数学公式、符号、上下标、根号)
2. 学生的完整手写答案和解题过程(包括所有步骤)

【数学符号识别规范】
- 根号: √(内容) 如 √(16×25/81)
- 分数: (分子)/(分母) 如 (1/2)、(3/4)
- 乘除: × ÷ (不要用*和/)
- 带分数: 整数+(分子)/(分母) 如 2+(1/3)
- 负数: (-1/8) 括号必须保留
- 指数: x^2 x^3 (用^表示上标)
- 不等式: ≥ ≤ ≠
- 根号嵌套: (a+√b) 用括号明确范围

【输出格式】
输出JSON格式(不要markdown包裹):
{
  "question": "完整题目原文,严格按照上述规范",
  "answer": "学生手写答案和所有解题步骤",
  "type": "calculation"
}

【注意事项】
- 必须保留所有括号、运算符号
- 复杂公式要完整,不要省略步骤
- 如果字迹模糊写'(无法识别)',不要猜测
- 多个小题要完整保留题号和内容
- 直接输出JSON,不要解释"""

# 统一的批量输出提示词（用于整体识别多道题）
UNIFIED_BATCH_PROMPT = """你是专业的数学作业批改老师,擅长识别各种数学题目和手写答案。

【识别要求】
请仔细观察整张作业图片,找出所有题目和学生手写答案。

【数学符号识别规范】
- 根号: √(内容) 如 √(x^2+y^2)
- 分数: (分子)/(分母) 
- 乘除: × ÷
- 带分数: 整数+(分子)/(分母)
- 负数: (-1/8) 必须保留括号
- 不等式: ≥ ≤
- 指数: x^2 x^3 (用^表示上标)
- 根号嵌套: 用括号明确范围

【输出格式】
输出JSON数组格式(不要markdown包裹):
[
  {
    "question_id": 1,
    "question": "题目原文,严格遵循上述规范",
    "answer": "学生答案和所有解题步骤",
    "type": "calculation"
  }
]

【重要提示】
- 长题目要完整识别,不要省略
- 多个小题都要保留编号(1)(2)(3)(4)
- 复杂公式必须完整,包括所有括号
- 只输出JSON数组,不要其他文字"""


# ============================================================
# 数据结构
# ============================================================

@dataclass
class QuestionBlock:
    """单道题目的结构化信息"""
    question_id: int
    question_text: str = ""
    answer_text: str = ""
    options: List[str] = field(default_factory=list)
    question_type: str = "unknown"
    confidence: float = 0.0
    bbox: List[int] = field(default_factory=list)


@dataclass
class OCRResult:
    """OCR识别完整结果（供D同学使用）"""
    total_questions: int = 0
    questions: List[QuestionBlock] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class SingleResult:
    """单张图片的识别结果（含元信息）"""
    image_name: str = ""
    total_questions: int = 0
    questions: List[QuestionBlock] = field(default_factory=list)
    processing_time: float = 0.0
    error: str = ""


@dataclass
class BatchResult:
    """批量识别结果"""
    total_images: int = 0
    success_count: int = 0
    fail_count: int = 0
    total_questions: int = 0
    total_time: float = 0.0
    results: List[SingleResult] = field(default_factory=list)


# ============================================================
# 辅助函数：安全解析LLM返回的JSON
# ============================================================

def safe_parse_llm_json(response_text: str) -> Any:
    """
    安全解析大模型返回的JSON，处理常见的格式问题
    包括：多余的反斜杠、markdown包裹、不完整的JSON等
    """
    # 移除markdown代码块标记
    response_text = response_text.strip()
    if "```" in response_text:
        # 提取代码块中的内容
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            response_text = matches[0].strip()
        else:
            # 如果没匹配到，就移除所有```标记
            response_text = re.sub(r"```\w*\n?", "", response_text)

    # 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"  [LLM JSON解析失败] 尝试修复...")

        # 修复常见的JSON问题
        # 1. 修复未转义的反斜杠（数学公式中的 \sqrt, \frac 等）
        # 在JSON字符串中，反斜杠需要转义为 \\
        def fix_unescaped_backslashes(match):
            content = match.group(0)
            # 如果不是已经是转义的反斜杠，则进行转义
            return content.replace('\\', '\\\\')

        # 更精确的方法：只在字符串值内部转义反斜杠
        # 匹配字符串值（简单版，不处理嵌套引号）
        pattern = r'("(?:[^"\\]|\\.)*")'

        def escape_backslashes_in_string(m):
            s = m.group(0)
            # 保留已转义的内容，只转义未转义的反斜杠
            # 将 \ 变成 \\，但避免重复转义
            s = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', s)
            return s

        try:
            # 先处理字符串内的反斜杠
            fixed = re.sub(pattern, escape_backslashes_in_string, response_text)
            # 再尝试解析
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 2. 尝试找到JSON数组或对象
        # 查找第一个 [ 或 { 到最后一个 ] 或 }
        start_idx = -1
        end_idx = -1
        for i, char in enumerate(response_text):
            if char in '[{' and start_idx == -1:
                start_idx = i
            if char in ']}':
                end_idx = i + 1

        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            try:
                extracted = response_text[start_idx:end_idx]
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass

        # 3. 如果还是失败，尝试修复单引号为双引号
        try:
            # 将单引号包裹的字符串转为双引号（但不影响已转义的）
            fixed = re.sub(r"(?<!\\)'", '"', response_text)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 所有方法都失败，重新抛出原始错误
        print(f"  [LLM JSON解析失败] 原始响应: {response_text[:500]}...")
        raise e


# ============================================================
# 核心引擎
# ============================================================

class AliyunOCREngine:
    """阿里云OCR引擎 + 多模态大模型"""

    def __init__(self):
        self._client: Optional[OCRClient] = None
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._init_client()

    def _init_client(self):
        try:
            ocr_config = open_api_models.Config(
                access_key_id=ALIYUN_ACCESS_KEY_ID,
                access_key_secret=ALIYUN_ACCESS_KEY_SECRET,
                endpoint=ALIYUN_OCR_ENDPOINT,
                region_id=ALIYUN_REGION_ID,
                connect_timeout=CONNECT_TIMEOUT_MS,
                read_timeout=READ_TIMEOUT_MS,
            )
            self._client = OCRClient(ocr_config)
            print("[OCR] 阿里云客户端初始化成功")
        except Exception as e:
            print(f"[OCR] 客户端初始化失败: {e}")
            raise

    # ============================================================
    # 【核心接口】供D同学调用 — 接收numpy数组
    # ============================================================

    async def ocr_recognize(self, processed_img: np.ndarray) -> Tuple[str, str]:
        """
        核心函数：对A同学预处理后的图片进行OCR识别

        Args:
            processed_img: A同学 image_process.py 返回的 numpy.ndarray (OpenCV BGR格式)

        Returns:
            (题目文本, 学生答案) 字符串元组
        """
        result = await self.process_image(processed_img)

        if not result.questions:
            print("[OCR] 警告：未识别到任何题目")
            return "", ""

        question_parts = []
        answer_parts = []

        for q in result.questions:
            q_text = f"【第{q.question_id}题】({q.question_type})\n{q.question_text}"
            if q.options:
                q_text += f"\n选项: {' | '.join(q.options)}"
            question_parts.append(q_text)
            answer_parts.append(f"第{q.question_id}题: {q.answer_text}")

        question_text = "\n\n".join(question_parts)
        answer_text = "\n".join(answer_parts)

        print(f"[OCR] 识别完成: {len(result.questions)}道题目, 耗时{result.processing_time:.1f}s")
        return question_text, answer_text

    async def recognize_from_array(self, img: np.ndarray, image_name: str = "unknown") -> SingleResult:
        """
        接收A同学传来的numpy数组进行识别

        Args:
            img: OpenCV格式的numpy数组 (BGR)
            image_name: 图片名称（用于日志）

        Returns:
            SingleResult 包含结构化的题目列表
        """
        start_time = time.time()
        result = SingleResult(image_name=image_name)

        try:
            if img is None or img.size == 0:
                result.error = "传入的图片为空"
                return result

            print(f"\n[OCR] 正在处理: {image_name} ({img.shape[1]}x{img.shape[0]})")

            ocr_result = await self.process_image(img)
            result.questions = ocr_result.questions
            result.total_questions = ocr_result.total_questions

            if result.total_questions == 0:
                result.error = "未识别到任何题目"

        except Exception as e:
            result.error = str(e)
            print(f"[OCR] 处理失败: {e}")

        result.processing_time = round(time.time() - start_time, 2)
        return result

    async def process_image(self, image: np.ndarray) -> OCRResult:
        """处理单张图片的核心流程（内部使用）"""
        start_time = time.time()
        result = OCRResult()

        image_bytes = self._image_to_bytes(image)

        # 步骤1：切题（获取题目区域）
        question_regions = await self._cut_paper(image_bytes)

        if question_regions and len(question_regions) > 1:
            print(f"[OCR] 切题完成: {len(question_regions)}个区域，逐题识别...")
            questions = await self._recognize_batch_with_llm(image, question_regions)
        else:
            print(f"[OCR] 使用大模型整体识别...")
            questions = await self._recognize_whole_with_llm(image)

        result.questions = questions
        result.total_questions = len(questions)
        result.processing_time = round(time.time() - start_time, 2)
        return result

    # ============================================================
    # 批量识别 — 接收numpy数组列表
    # ============================================================

    async def recognize_batch_from_arrays(
        self,
        images: List[Tuple[np.ndarray, str]]
    ) -> BatchResult:
        """
        批量识别多张A同学处理后的图片

        Args:
            images: [(numpy数组, 图片名), ...] 的列表

        Returns:
            BatchResult
        """
        start_time = time.time()
        batch = BatchResult(total_images=len(images))

        print(f"\n{'='*60}")
        print(f"[批量识别] 共 {len(images)} 张图片")
        print(f"{'='*60}")

        tasks = [self.recognize_from_array(img, name) for img, name in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                batch.fail_count += 1
                batch.results.append(SingleResult(
                    image_name=images[i][1],
                    error=str(res),
                ))
            else:
                if res.error:
                    batch.fail_count += 1
                else:
                    batch.success_count += 1
                    batch.total_questions += res.total_questions
                batch.results.append(res)

        batch.total_time = round(time.time() - start_time, 2)

        print(f"\n{'='*60}")
        print(f"[批量识别完成] 成功: {batch.success_count} | 失败: {batch.fail_count}")
        print(f"  总题目数: {batch.total_questions} | 总耗时: {batch.total_time:.1f}s")
        print(f"{'='*60}")

        return batch

    # ============================================================
    # 图片处理
    # ============================================================

    def _image_to_bytes(self, image: np.ndarray) -> bytes:
        h, w = image.shape[:2]
        max_edge = 1024
        if max(h, w) > max_edge:
            scale = max_edge / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def _image_to_base64_url(self, image: np.ndarray) -> str:
        bytes_data = self._image_to_bytes(image)
        b64_str = base64.b64encode(bytes_data).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

    # ============================================================
    # 试卷切题
    # ============================================================

    async def _cut_paper(self, image_bytes: bytes, retry: int = 0) -> List[Dict]:
        try:
            request = ocr_models.RecognizeEduPaperCutRequest(
                body=image_bytes,
                image_type="photo",
                cut_type="question",
            )
            runtime = util_models.RuntimeOptions()
            response = await self._async_call(
                lambda: self._client.recognize_edu_paper_cut_with_options(request, runtime)
            )
            data = json.loads(response.body.data)

            regions = self._parse_regions(data)
            if not regions:
                regions = self._parse_regions_v2(data)
            if not regions:
                regions = self._parse_regions_v3(data)

            return regions

        except Exception as e:
            if retry < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
                return await self._cut_paper(image_bytes, retry + 1)
            print(f"[OCR] 切题失败: {e}")
            return []

    def _parse_regions(self, data: dict) -> List[Dict]:
        regions = []
        page_list = data.get("data", {}).get("page_list", [])
        for page in page_list:
            subject_list = page.get("subject_list", [])
            for idx, subject in enumerate(subject_list):
                ids = subject.get("ids", [])
                question_num = ids[0] if (isinstance(ids, list) and ids) else idx + 1
                text = subject.get("text", "")
                content_list = subject.get("content_list_info", [])
                if content_list:
                    pos_info = content_list[0].get("pos", [])
                    if pos_info:
                        regions.append({
                            "question_num": question_num,
                            "text": text,
                            "left": int(min(p["x"] for p in pos_info)),
                            "top": int(min(p["y"] for p in pos_info)),
                            "width": int(max(p["x"] for p in pos_info) - min(p["x"] for p in pos_info)),
                            "height": int(max(p["y"] for p in pos_info) - min(p["y"] for p in pos_info)),
                        })
            break
        return regions

    def _parse_regions_v2(self, data: dict) -> List[Dict]:
        regions = []
        question_list = data.get("data", {}).get("question_info", [])
        for idx, item in enumerate(question_list):
            prism = item.get("prism_gtInfo", [{}])
            pos = prism[0] if isinstance(prism, list) and prism else prism
            if pos.get("width", 0) > 0:
                regions.append({
                    "question_num": item.get("question_num", idx + 1),
                    "text": "",
                    "left": int(pos.get("x", 0)), "top": int(pos.get("y", 0)),
                    "width": int(pos.get("width", 0)), "height": int(pos.get("height", 0)),
                })
        return regions

    def _parse_regions_v3(self, data: dict) -> List[Dict]:
        regions = []
        figures = data.get("data", {}).get("figure", [])
        for idx, fig in enumerate(figures):
            if fig.get("type") == "subject_pattern":
                regions.append({
                    "question_num": idx + 1,
                    "text": "",
                    "left": int(fig.get("x", 0)), "top": int(fig.get("y", 0)),
                    "width": int(fig.get("w", 0)), "height": int(fig.get("h", 0)),
                })
        return regions

    # ============================================================
    # 大模型识别 - 使用统一的prompt
    # ============================================================

    async def _recognize_batch_with_llm(self, image: np.ndarray, regions: List[Dict]) -> List[QuestionBlock]:
        async def recognize_one(region: Dict) -> Optional[QuestionBlock]:
            sub_img = self._crop_region(image, region)
            base64_url = self._image_to_base64_url(sub_img)

            for retry in range(MAX_RETRIES):
                try:
                    result_json = await self._call_llm(UNIFIED_PROMPT, base64_url)
                    data = safe_parse_llm_json(result_json)

                    # 确保data是字典
                    if isinstance(data, list) and data:
                        data = data[0]

                    return QuestionBlock(
                        question_id=region.get("question_num", 0),
                        question_text=data.get("question", ""),
                        answer_text=data.get("answer", ""),
                        question_type=data.get("type", "calculation"),
                        bbox=[region.get("left", 0), region.get("top", 0),
                              region.get("width", 0), region.get("height", 0)],
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  [识别区域{region.get('question_num')}] JSON解析失败, 重试 {retry+1}/{MAX_RETRIES}")
                    if retry < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)

            return QuestionBlock(
                question_id=region.get("question_num", 0),
                question_text="（识别失败）",
                answer_text="（识别失败）",
                bbox=[region.get("left", 0), region.get("top", 0),
                      region.get("width", 0), region.get("height", 0)],
            )

        tasks = [recognize_one(r) for r in regions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        questions = [r for r in results if isinstance(r, QuestionBlock)]
        questions.sort(key=lambda q: q.question_id)
        return questions

    async def _recognize_whole_with_llm(self, image: np.ndarray) -> List[QuestionBlock]:
        base64_url = self._image_to_base64_url(image)
        h, w = image.shape[:2]

        try:
            result_text = await self._call_llm(UNIFIED_BATCH_PROMPT, base64_url)
            print(f"  [LLM原始响应] {result_text[:200]}...")

            data = safe_parse_llm_json(result_text)

            if not isinstance(data, list):
                data = [data]

            questions = []
            for item in data:
                bbox = item.get("bbox", [0, 0, w, h])
                questions.append(QuestionBlock(
                    question_id=item.get("question_id", len(questions) + 1),
                    question_text=item.get("question", ""),
                    answer_text=item.get("answer", ""),
                    question_type=item.get("type", "calculation"),
                    bbox=bbox if isinstance(bbox, list) else [0, 0, w, h],
                ))

            print(f"  [LLM] 整体识别完成: {len(questions)}道题目")
            return questions

        except Exception as e:
            print(f"[LLM] 整体识别失败: {e}")
            return []

    # ============================================================
    # 大模型调用
    # ============================================================

    async def _call_llm(self, prompt: str, base64_url: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._call_qwen_api(prompt, base64_url))

    def _call_qwen_api(self, prompt: str, base64_url: str) -> str:
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": QWEN_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_url}},
                ],
            }],
            "max_tokens": 4096,  # 增大token限制，避免输出被截断
        }
        resp = requests.post(QWEN_ENDPOINT, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ============================================================
    # 辅助方法
    # ============================================================

    def _crop_region(self, image: np.ndarray, region: Dict, pad: int = 15) -> np.ndarray:
        """裁剪区域，对大区域自动增加padding"""
        h, w = image.shape[:2]
        x = max(0, region.get("left", 0) - pad)
        y = max(0, region.get("top", 0) - pad)

        # 对于大区域增加更多padding，确保长题目完整
        region_size = region.get("width", 0) * region.get("height", 0)
        if region_size > 50000:  # 大题目区域
            pad = pad * 2
            x = max(0, region.get("left", 0) - pad)
            y = max(0, region.get("top", 0) - pad)

        rw = min(region.get("width", w) + 2 * pad, w - x)
        rh = min(region.get("height", h) + 2 * pad, h - y)

        return image[y:y + rh, x:x + rw]

    async def _async_call(self, func):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)


# ============================================================
# 【对外暴露的核心函数】供D同学直接 import 使用
# ============================================================

_engine: Optional[AliyunOCREngine] = None


def get_engine() -> AliyunOCREngine:
    global _engine
    if _engine is None:
        _engine = AliyunOCREngine()
    return _engine


async def ocr_recognize(processed_img: np.ndarray) -> Tuple[str, str]:
    """
    【A、D同学调用的核心函数】

    接收A同学 image_process.py 处理后的图片（numpy数组），
    返回(题目文本, 学生答案)供D同学传给C同学批改。

    Args:
        processed_img: A同学 preprocess_image() 返回的 numpy.ndarray

    Returns:
        (题目文本, 学生答案)
    """
    engine = get_engine()
    return await engine.ocr_recognize(processed_img)


async def ocr_recognize_batch(images: List[Tuple[np.ndarray, str]]) -> BatchResult:
    """
    批量识别接口

    Args:
        images: [(numpy数组, 图片名), ...]

    Returns:
        BatchResult
    """
    engine = get_engine()
    return await engine.recognize_batch_from_arrays(images)


async def ocr_recognize_for_c(processed_img: np.ndarray, subject_type: str = "general") -> List[Dict]:
    """
    【供D同学调用】识别图片，返回完全匹配C同学要求的格式

    Args:
        processed_img: A同学处理后的图片
        subject_type: 学科类型（math/chinese/english/history/physics/chemistry/geography/general）

    Returns:
        [
            {
                "question_id": "Q1",
                "question_text": "题目原文+选项",
                "student_answer": "学生答案",
                "max_score": 10,
                "subject_type": "history",
                "bbox": [x, y, w, h]
            },
            ...
        ]
    """
    engine = get_engine()
    result = await engine.recognize_from_array(processed_img, "homework")

    if result.error or not result.questions:
        return []

    questions_for_c = []
    for q in result.questions:
        full_question = q.question_text
        if q.options:
            full_question += "\n" + "\n".join(q.options)

        questions_for_c.append({
            "question_id": f"Q{q.question_id}",
            "question_text": full_question,
            "student_answer": q.answer_text,
            "max_score": 10,
            "subject_type": subject_type,
            "bbox": q.bbox if q.bbox else [0, 0, 0, 0],
        })

    return questions_for_c


# ============================================================
# 文件工具（独立测试用）
# ============================================================


def collect_images(path: str) -> List[str]:
    """收集图片文件"""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_FORMATS:
            return [str(p)]
        return []
    elif p.is_dir():
        files = []
        for ext in SUPPORTED_FORMATS:
            for f in p.glob(f"*{ext}"):
                if f.is_file():
                    files.append(str(f))
        return sorted(set(files))
    return []


def load_image_as_array(image_path: str) -> Optional[np.ndarray]:
    """从文件路径加载图片为numpy数组（独立测试时使用）"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def export_to_json(batch: BatchResult, output_path: str):
    """导出结果为JSON"""
    output = {
        "summary": {
            "total_images": batch.total_images,
            "success_count": batch.success_count,
            "fail_count": batch.fail_count,
            "total_questions": batch.total_questions,
            "total_time_seconds": batch.total_time,
        },
        "results": []
    }
    for r in batch.results:
        result_item = {
            "image_name": r.image_name,
            "total_questions": r.total_questions,
            "processing_time": r.processing_time,
            "error": r.error,
            "questions": []
        }
        for q in r.questions:
            result_item["questions"].append({
                "question_id": f"Q{q.question_id}",
                "question_text": q.question_text,
                "student_answer": q.answer_text,
                "max_score": 10,
                "subject_type": "general",
                "bbox": q.bbox if q.bbox else [0, 0, 0, 0],
            })
        output["results"].append(result_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[导出] 结果已保存到: {output_path}")


def print_result(result: SingleResult):
    if result.error:
        print(f"\n  ❌ {result.image_name}: {result.error}")
        return

    print(f"\n  ✅ {result.image_name} ({result.total_questions}题, {result.processing_time:.1f}s)")
    for q in result.questions:
        print(f"    第{q.question_id}题 [{q.question_type}] bbox={q.bbox}:")
        print(f"      题目: {q.question_text[:100]}")
        print(f"      答案: {q.answer_text[:100]}")
        print()


# ============================================================
# 主入口（独立测试 / 命令行模式）
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="阿里云OCR作业批改系统 - OCR模块")
    parser.add_argument("--image", "-i", type=str, help="单张图片路径")
    parser.add_argument("--dir", "-d", type=str, help="图片文件夹路径（批量识别）")
    parser.add_argument("--output", "-o", type=str, help="导出JSON结果路径")
    args = parser.parse_args()

    engine = get_engine()

    # ========== 命令行模式 ==========
    if args.image or args.dir:
        if args.image:
            images = collect_images(args.image)
        else:
            images = collect_images(args.dir)

        if not images:
            print("❌ 未找到任何图片文件")
            return

        if len(images) == 1:
            img = load_image_as_array(images[0])
            if img is None:
                print(f"❌ 无法读取图片: {images[0]}")
                return
            result = await engine.recognize_from_array(img, Path(images[0]).name)
            print_result(result)
            if args.output:
                batch = BatchResult(total_images=1, success_count=1 if not result.error else 0,
                                    fail_count=1 if result.error else 0,
                                    total_questions=result.total_questions,
                                    total_time=result.processing_time,
                                    results=[result])
                export_to_json(batch, args.output)
        else:
            img_arrays = []
            for p in images:
                img = load_image_as_array(p)
                if img is not None:
                    img_arrays.append((img, Path(p).name))
            if not img_arrays:
                print("❌ 没有可读取的图片")
                return
            batch = await engine.recognize_batch_from_arrays(img_arrays)
            for r in batch.results:
                print_result(r)
            if args.output:
                export_to_json(batch, args.output)
        return

    # ========== 交互模式 ==========
    print("=" * 60)
    print("  阿里云OCR作业识别系统")
    print("=" * 60)

    while True:
        print("\n请选择操作：")
        print("  1. 识别单张图片（输入路径）")
        print("  2. 批量识别文件夹")
        print("  3. 退出")
        choice = input("请输入选项 (1/2/3): ").strip()

        if choice == "1":
            img_path = input("请输入图片路径: ").strip().strip('"')
            images = collect_images(img_path)
            if not images:
                print(f"❌ 未找到图片: {img_path}")
                continue
            img = load_image_as_array(images[0])
            if img is None:
                print(f"❌ 无法读取: {images[0]}")
                continue
            result = await engine.recognize_from_array(img, Path(images[0]).name)
            print_result(result)

        elif choice == "2":
            dir_path = input("请输入文件夹路径: ").strip().strip('"')
            images = collect_images(dir_path)
            if not images:
                print(f"❌ 文件夹中没有图片: {dir_path}")
                continue
            img_arrays = []
            for p in images:
                img = load_image_as_array(p)
                if img is not None:
                    img_arrays.append((img, Path(p).name))
            batch = await engine.recognize_batch_from_arrays(img_arrays)
            for r in batch.results:
                print_result(r)

            save = input("\n是否导出JSON结果? (y/n): ").strip().lower()
            if save == "y":
                output_path = input("请输入导出路径 (如 results.json): ").strip()
                if not output_path.endswith('.json'):
                    output_path += '.json'
                export_to_json(batch, output_path)

        elif choice == "3":
            print("再见！")
            break
        else:
            print("❌ 无效选项")


if __name__ == "__main__":
    asyncio.run(main())