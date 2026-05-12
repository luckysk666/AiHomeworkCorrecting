"""
阿里云OCR识别模块

功能：
1. 调用阿里云RecognizeEduPaperCut（试卷切题）API — 提取题干区域坐标
2. 使用多模态大模型（Qwen-VL）直接理解图片中的手写题目和答案
3. 支持单张识别 + 批量识别 + JSON结果导出

使用方式：
    - 单张识别: python aliyun_ocr.py --image test.jpg
    - 批量识别: python aliyun_ocr.py --dir test_imgs/
    - 批量导出: python aliyun_ocr.py --dir test_imgs/ --output results.json
    - 交互模式: python aliyun_ocr.py

依赖：pip install alibabacloud_ocr_api20210707 opencv-python numpy requests
"""

import json
import time
import asyncio
import base64
import os
import sys
import argparse
import requests
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

# 阿里云OCR SDK
from alibabacloud_ocr_api20210707.client import Client as OCRClient
from alibabacloud_ocr_api20210707 import models as ocr_models
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models


# ============================================================
# 配置
# ============================================================
ALIYUN_ACCESS_KEY_ID = "LTAI5t9nfB48Qbvi8fmU9jJW"
ALIYUN_ACCESS_KEY_SECRET = "ADRDeu42z2IXlRx8593O1Y0QYmMwhK"
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

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


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


@dataclass
class SingleResult:
    """单张图片的识别结果"""
    image_path: str = ""
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
            print("[OCR] 阿里云OCR客户端初始化成功")
        except Exception as e:
            print(f"[OCR] 客户端初始化失败: {e}")
            raise

    # ============================================================
    # 公共接口
    # ============================================================

    async def recognize_single(self, image_path: str) -> SingleResult:
        """识别单张图片"""
        start_time = time.time()
        result = SingleResult(
            image_path=image_path,
            image_name=Path(image_path).name,
        )

        try:
            # 【关键修复】用 numpy 读取，解决中文路径问题
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                result.error = f"无法读取图片: {image_path}"
                return result

            print(f"\n[OCR] 正在处理: {result.image_name} ({img.shape[1]}x{img.shape[0]})")

            questions = await self._process_image(img)
            result.questions = questions
            result.total_questions = len(questions)

            if result.total_questions == 0:
                result.error = "未识别到任何题目"

        except Exception as e:
            result.error = str(e)
            print(f"[OCR] 处理失败: {e}")

        result.processing_time = round(time.time() - start_time, 2)
        return result

    async def recognize_batch(self, image_paths: List[str]) -> BatchResult:
        """批量识别多张图片"""
        start_time = time.time()
        batch = BatchResult(total_images=len(image_paths))

        print(f"\n{'='*60}")
        print(f"[批量识别] 共 {len(image_paths)} 张图片")
        print(f"{'='*60}")

        tasks = [self.recognize_single(p) for p in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                batch.fail_count += 1
                batch.results.append(SingleResult(
                    image_path=image_paths[i],
                    image_name=Path(image_paths[i]).name,
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
        print(f"[批量识别完成]")
        print(f"  总图片: {batch.total_images}")
        print(f"  成功: {batch.success_count} | 失败: {batch.fail_count}")
        print(f"  总题目数: {batch.total_questions}")
        print(f"  总耗时: {batch.total_time:.1f}s")
        print(f"{'='*60}")

        return batch

    async def _process_image(self, image: np.ndarray) -> List[QuestionBlock]:
        """处理单张图片的核心流程"""
        image_bytes = self._image_to_bytes(image)

        # 步骤1：切题（获取题目区域）
        question_regions = await self._cut_paper(image_bytes)

        if question_regions and len(question_regions) > 1:
            # 多个区域，逐题识别
            print(f"[OCR] 切题完成: {len(question_regions)}个区域，逐题识别...")
            questions = await self._recognize_batch_with_llm(image, question_regions)
        else:
            # 切题失败或只有1个区域，整体识别
            print(f"[OCR] 使用大模型整体识别...")
            questions = await self._recognize_whole_with_llm(image)

        return questions

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
    # 大模型识别
    # ============================================================

    async def _recognize_batch_with_llm(self, image: np.ndarray, regions: List[Dict]) -> List[QuestionBlock]:
        async def recognize_one(region: Dict) -> Optional[QuestionBlock]:
            sub_img = self._crop_region(image, region)
            base64_url = self._image_to_base64_url(sub_img)

            prompt = """你是一位作业批改老师。请仔细观察这张图片，识别其中的题目和学生手写答案。

请严格输出JSON（不要markdown包裹）：
{
  "question": "题目原文",
  "answer": "学生手写答案",
  "type": "calculation"
}
看不清就写"（无法识别）"。"""

            for retry in range(MAX_RETRIES):
                try:
                    result_json = await self._call_llm(prompt, base64_url)
                    # 清理markdown包裹
                    if "```" in result_json:
                        result_json = result_json.split("```")[1]
                        if result_json.startswith("json"):
                            result_json = result_json[4:]
                    data = json.loads(result_json.strip())
                    return QuestionBlock(
                        question_id=region.get("question_num", 0),
                        question_text=data.get("question", ""),
                        answer_text=data.get("answer", ""),
                        question_type=data.get("type", "calculation"),
                    )
                except (json.JSONDecodeError, KeyError):
                    if retry < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)

            return QuestionBlock(
                question_id=region.get("question_num", 0),
                question_text="（识别失败）",
                answer_text="（识别失败）",
            )

        tasks = [recognize_one(r) for r in regions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        questions = [r for r in results if isinstance(r, QuestionBlock)]
        questions.sort(key=lambda q: q.question_id)

        for q in questions:
            print(f"  [LLM] 第{q.question_id}题: {q.question_text[:40]}... → {q.answer_text[:30]}...")

        return questions

    async def _recognize_whole_with_llm(self, image: np.ndarray) -> List[QuestionBlock]:
        base64_url = self._image_to_base64_url(image)

        prompt = """请仔细查看这张作业图片，找出所有题目和学生手写答案。

输出JSON数组（不要markdown包裹）：
[
  {"question_id": 1, "question": "题目内容", "answer": "学生手写答案", "type": "calculation"},
  ...
]
只输出JSON数组。"""

        try:
            result_text = await self._call_llm(prompt, base64_url)
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            result_text = result_text.strip()

            data = json.loads(result_text)
            if not isinstance(data, list):
                data = [data]

            questions = []
            for item in data:
                questions.append(QuestionBlock(
                    question_id=item.get("question_id", len(questions) + 1),
                    question_text=item.get("question", ""),
                    answer_text=item.get("answer", ""),
                    question_type=item.get("type", "calculation"),
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
            "max_tokens": 2000,
        }
        resp = requests.post(QWEN_ENDPOINT, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    # ============================================================
    # 辅助方法
    # ============================================================

    def _crop_region(self, image: np.ndarray, region: Dict, pad: int = 15) -> np.ndarray:
        h, w = image.shape[:2]
        x = max(0, region.get("left", 0) - pad)
        y = max(0, region.get("top", 0) - pad)
        rw = min(region.get("width", w) + 2 * pad, w - x)
        rh = min(region.get("height", h) + 2 * pad, h - y)
        return image[y:y + rh, x:x + rw]

    async def _async_call(self, func):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)


# ============================================================
# 模块级便捷函数
# ============================================================

_engine: Optional[AliyunOCREngine] = None


def get_engine() -> AliyunOCREngine:
    global _engine
    if _engine is None:
        _engine = AliyunOCREngine()
    return _engine


async def ocr_recognize(processed_img: np.ndarray) -> Tuple[str, str]:
    """对外暴露的核心函数 — 供D同学调用"""
    engine = get_engine()
    questions = await engine._process_image(processed_img)
    if not questions:
        return "", ""
    q_parts = [f"【第{q.question_id}题】({q.question_type})\n{q.question_text}" for q in questions]
    a_parts = [f"第{q.question_id}题: {q.answer_text}" for q in questions]
    return "\n\n".join(q_parts), "\n".join(a_parts)


# ============================================================
# 文件工具
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
            # 统一转小写比较，避免重复
            for f in p.glob(f"*"):
                if f.is_file() and f.suffix.lower() == ext:
                    files.append(str(f))
        # 去重并排序
        return sorted(set(files))
    return []


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
        output["results"].append({
            "image_name": r.image_name,
            "image_path": r.image_path,
            "total_questions": r.total_questions,
            "processing_time": r.processing_time,
            "error": r.error,
            "questions": [asdict(q) for q in r.questions],
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[导出] 结果已保存到: {output_path}")


def print_result(result: SingleResult):
    """打印单张识别结果"""
    if result.error:
        print(f"\n  ❌ {result.image_name}: {result.error}")
        return

    print(f"\n  ✅ {result.image_name} ({result.total_questions}题, {result.processing_time:.1f}s)")
    for q in result.questions:
        print(f"    第{q.question_id}题 [{q.question_type}]:")
        print(f"      题目: {q.question_text[:80]}...")
        print(f"      答案: {q.answer_text[:80]}...")


# ============================================================
# 主入口
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
            result = await engine.recognize_single(images[0])
            print_result(result)
            if args.output:
                batch = BatchResult(total_images=1, success_count=1 if not result.error else 0,
                                    fail_count=1 if result.error else 0,
                                    total_questions=result.total_questions,
                                    total_time=result.processing_time,
                                    results=[result])
                export_to_json(batch, args.output)
        else:
            batch = await engine.recognize_batch(images)
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
            result = await engine.recognize_single(images[0])
            print_result(result)

        elif choice == "2":
            dir_path = input("请输入文件夹路径: ").strip().strip('"')
            images = collect_images(dir_path)
            if not images:
                print(f"❌ 文件夹中没有图片: {dir_path}")
                continue
            batch = await engine.recognize_batch(images)
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