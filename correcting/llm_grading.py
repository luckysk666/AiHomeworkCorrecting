# llm_grading.py - 确保文本完整显示
import base64
import json
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GradingResult:
    """批改结果数据类"""
    question_id: str = ""
    score: float = 0.0
    max_score: float = 10.0
    comment: str = ""
    errors: List[Dict] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.suggestions is None:
            self.suggestions = []


class LLMGradingEngine:
    """大模型批改引擎"""

    def __init__(self, api_key: str, api_type: str = "aliyun"):
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = "qwen-max-latest"

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def grade_question(self, question_text: str, student_answer: str,
                             max_score: float = 10.0,
                             subject_type: str = "history") -> GradingResult:
        """AI批改单个题目"""

        prompt = self._build_strict_grading_prompt(question_text, student_answer, max_score, subject_type)

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是严格的历史老师。只返回JSON，不要有任何其他文字。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }

            try:
                async with session.post(f"{self.base_url}/chat/completions",
                                        headers=self.headers,
                                        json=payload,
                                        timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]

                        content = content.strip()

                        # 提取JSON
                        json_match = re.search(r'\{[^{}]*\}', content)
                        if json_match:
                            content = json_match.group()
                        else:
                            brace_count = 0
                            start_idx = -1
                            for i, ch in enumerate(content):
                                if ch == '{':
                                    if start_idx == -1:
                                        start_idx = i
                                    brace_count += 1
                                elif ch == '}':
                                    brace_count -= 1
                                    if brace_count == 0 and start_idx != -1:
                                        content = content[start_idx:i + 1]
                                        break

                        if content.startswith("```json"):
                            content = content[7:]
                        if content.startswith("```"):
                            content = content[3:]
                        if content.endswith("```"):
                            content = content[:-3]

                        grading_data = json.loads(content)
                        score = float(grading_data.get("score", 0))

                        return GradingResult(
                            question_id=grading_data.get("question_id", ""),
                            score=score,
                            max_score=max_score,
                            comment=grading_data.get("comment", ""),
                            errors=grading_data.get("errors", []),
                            suggestions=grading_data.get("suggestions", [])
                        )
                    else:
                        return self._get_default_result(max_score)

            except Exception as e:
                logger.error(f"批改出错: {e}")
                return self._get_default_result(max_score)

    def _build_strict_grading_prompt(self, question_text: str, student_answer: str,
                                     max_score: float, subject_type: str) -> str:
        prompt = f"""批改以下{subject_type}选择题：

题目：{question_text}

学生答案：{student_answer}

满分：{max_score}分

规则：选对给{max_score}分，选错给0分。

只返回JSON，格式：{{"score": 分数, "comment": "评语", "suggestions": []}}"""
        return prompt

    def _get_default_result(self, max_score: float) -> GradingResult:
        return GradingResult(
            score=0,
            max_score=max_score,
            comment="批改失败",
            errors=[],
            suggestions=[]
        )


class ImageAnnotator:
    """图片批注意见添加器"""

    def __init__(self):
        self.font_size = 26
        self.font = self._load_font()

    def _load_font(self):
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, self.font_size)
            except:
                continue
        return ImageFont.load_default()

    def _get_text_lines(self, text, max_chars_per_line=60):
        """将长文本分成多行，每行最多60字符"""
        if len(text) <= max_chars_per_line:
            return [text]
        lines = []
        for i in range(0, len(text), max_chars_per_line):
            lines.append(text[i:i + max_chars_per_line])
        return lines

    async def add_all_annotations(self, image: Image.Image,
                                  grading_results: List[GradingResult],
                                  questions_data: List[Dict]) -> Image.Image:
        """在图片右侧添加批注面板"""

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 计算每道题的行数
        lines_per_question = []
        max_line_chars = 0

        for i, (result, question) in enumerate(zip(grading_results, questions_data)):
            lines = []
            is_correct = result.score == result.max_score and result.max_score > 0
            status = "正确" if is_correct else "错误"
            lines.append(f"第{i + 1}题 ({status}) | 得分: {result.score:.1f}/{result.max_score:.0f}")

            comment_text = f"评语: {result.comment}"
            lines.extend(self._get_text_lines(comment_text, 60))

            student_ans = question.get('student_answer', '')
            lines.extend(self._get_text_lines(f"学生答案: {student_ans}", 60))

            if not is_correct and result.suggestions:
                lines.append(f"💡 提示: {result.suggestions[0]}")
            elif is_correct:
                lines.append("🎉 回答正确！")

            lines_per_question.append(lines)
            for line in lines:
                max_line_chars = max(max_line_chars, len(line))

        # ========== 大幅增加面板宽度 ==========
        # 每字符约25像素，60字符需要1500像素
        char_width = self.font_size * 1.3
        panel_width = int(max_line_chars * char_width) + 120
        panel_width = max(700, min(panel_width, 1000))  # 最小700，最大1000

        line_height = 38
        title_height = 65
        bottom_padding = 60
        total_lines = sum(len(lines) for lines in lines_per_question) + len(grading_results)
        panel_height = title_height + total_lines * line_height + bottom_padding

        # ========== 扩展图片宽度 ==========
        original_width = image.width
        original_height = image.height

        # 批注面板需要的总宽度（包括左边距和右边距）
        required_width = original_width + panel_width + 50

        # 确保图片高度足够
        new_height = max(original_height, panel_height + 150)
        new_width = required_width

        # 创建白色背景的新图片
        new_image = Image.new('RGB', (new_width, new_height), 'white')
        new_image.paste(image, (20, 20))  # 原图放在左上角，留出边距

        image = new_image
        draw = ImageDraw.Draw(image)

        logger.info(f"图片已扩展: {original_width}x{original_height} -> {new_width}x{new_height}")
        logger.info(f"面板宽度: {panel_width}, 最大字符数: {max_line_chars}")

        # 批注面板位置（在原图右侧）
        annotation_x = original_width + 40
        annotation_y = 30

        # 绘制背景框
        draw.rectangle([annotation_x - 10, annotation_y - 10,
                        annotation_x + panel_width, annotation_y + panel_height + 10],
                       fill=(248, 248, 255), outline=(180, 180, 200), width=2)
        draw.rectangle([annotation_x - 5, annotation_y - 5,
                        annotation_x + panel_width - 5, annotation_y + panel_height + 5],
                       fill=(255, 255, 255))

        current_y = annotation_y

        # 总分标题
        total_score = sum(r.score for r in grading_results)
        total_max = sum(r.max_score for r in grading_results)
        percentage = (total_score / total_max * 100) if total_max > 0 else 0

        title = f"📊 批改结果 | 总分: {total_score:.1f}/{total_max:.0f} ({percentage:.0f}%)"
        draw.rectangle([annotation_x - 3, current_y - 3,
                        annotation_x + panel_width - 10, current_y + 50],
                       fill=(240, 248, 255))
        draw.text((annotation_x, current_y + 10), title, fill=(0, 0, 0), font=self.font)
        current_y += 55
        draw.line([annotation_x, current_y, annotation_x + panel_width - 10, current_y],
                  fill=(220, 220, 230), width=1)
        current_y += 15

        # 每道题的批注
        for i, (result, question, lines) in enumerate(zip(grading_results, questions_data, lines_per_question)):
            is_correct = result.score == result.max_score and result.max_score > 0
            emoji = "✅" if is_correct else "❌"
            color = (0, 128, 0) if is_correct else (220, 20, 60)

            for line_idx, line in enumerate(lines):
                if line_idx == 0:
                    draw.text((annotation_x, current_y), f"{emoji} {line}", fill=color, font=self.font)
                else:
                    draw.text((annotation_x + 15, current_y), line, fill=(80, 80, 80), font=self.font)
                current_y += line_height

            current_y += 10
            if i < len(grading_results) - 1:
                draw.line([annotation_x, current_y, annotation_x + panel_width - 10, current_y],
                          fill=(235, 235, 245), width=1)
                current_y += 15

        # 鼓励语
        current_y += 10
        if percentage >= 80:
            encouragement = "🎉 优秀！继续保持！ 🎉"
        elif percentage >= 60:
            encouragement = "👍 良好！继续努力！ 👍"
        else:
            encouragement = "💪 加油！认真复习，下次会更好！ 💪"

        draw.rectangle([annotation_x - 3, current_y - 3,
                        annotation_x + panel_width - 10, current_y + 45],
                       fill=(240, 255, 240))
        draw.text((annotation_x, current_y + 10), encouragement, fill=(0, 128, 0), font=self.font)

        return image


class BatchGradingProcessor:
    """批量批改处理器"""

    def __init__(self, api_key: str, api_type: str = "aliyun", max_concurrent: int = 3):
        self.llm_engine = LLMGradingEngine(api_key, api_type)
        self.annotator = ImageAnnotator()
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_question(self, question_data: Dict, question_index: int) -> GradingResult:
        async with self.semaphore:
            logger.info(f"🤖 AI批改第{question_index + 1}题...")
            result = await self.llm_engine.grade_question(
                question_text=question_data.get("question_text", ""),
                student_answer=question_data.get("student_answer", ""),
                max_score=question_data.get("max_score", 10.0),
                subject_type=question_data.get("subject_type", "history")
            )
            is_correct = result.score == result.max_score
            logger.info(f"  结果: {'✅正确' if is_correct else '❌错误'} | 得分: {result.score}/{result.max_score}")
            return result

    async def process_batch(self, questions: List[Dict]) -> List[GradingResult]:
        tasks = [self.process_single_question(q, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        grading_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                grading_results.append(GradingResult(
                    question_id=f"q_{i + 1}",
                    score=0,
                    max_score=questions[i].get("max_score", 10),
                    comment="批改失败",
                    errors=[],
                    suggestions=[]
                ))
            else:
                grading_results.append(result)
        return grading_results


async def process_homework(image_data: bytes,
                           questions_data: List[Dict],
                           api_key: str,
                           api_type: str = "aliyun") -> Dict:
    """处理作业批改的主函数"""
    try:
        processor = BatchGradingProcessor(api_key, api_type, max_concurrent=3)
        grading_results = await processor.process_batch(questions_data)

        image = Image.open(io.BytesIO(image_data))

        # 如果原图太宽，先缩小
        max_original_width = 1000
        if image.width > max_original_width:
            ratio = max_original_width / image.width
            new_size = (max_original_width, int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"原图已缩小到: {new_size}")

        annotated_image = await processor.annotator.add_all_annotations(
            image, grading_results, questions_data
        )

        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format='PNG')

        total_score = sum(r.score for r in grading_results)
        total_max_score = sum(r.max_score for r in grading_results)

        print("\n" + "=" * 60)
        print("🤖 AI批改结果")
        print("=" * 60)
        for i, r in enumerate(grading_results, 1):
            status = "✅ 正确" if r.score == r.max_score else "❌ 错误"
            print(f"\n第{i}题 {status}")
            print(f"  得分: {r.score}/{r.max_score}")
            print(f"  评语: {r.comment}")
        print(f"\n📊 总分: {total_score}/{total_max_score} ({total_score / total_max_score * 100:.0f}%)")
        print("=" * 60)

        return {
            "success": True,
            "grading_results": [
                {
                    "question_id": r.question_id,
                    "score": r.score,
                    "max_score": r.max_score,
                    "comment": r.comment,
                    "errors": r.errors,
                    "suggestions": r.suggestions
                }
                for r in grading_results
            ],
            "total_score": total_score,
            "total_max_score": total_max_score,
            "percentage": (total_score / total_max_score * 100) if total_max_score > 0 else 0,
            "annotated_image": base64.b64encode(img_byte_arr.getvalue()).decode('utf-8'),
            "message": "AI批改完成"
        }

    except Exception as e:
        logger.error(f"批改失败: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "AI批改失败"
        }