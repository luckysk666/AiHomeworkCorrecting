# llm_grading.py - 只生成HTML，修复换行和空行问题
import base64
import json
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
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
        prompt = self._build_grading_prompt(question_text, student_answer, max_score, subject_type)

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是专业的批改老师。请严格返回JSON格式，不要添加额外文字。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }

            try:
                async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=45)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]
                        return self._parse_grading_response(content, max_score)
                    else:
                        logger.error(f"API错误: {resp.status}")
                        return self._get_default_result(max_score)
            except asyncio.TimeoutError:
                logger.error("API请求超时")
                return self._get_default_result(max_score)
            except Exception as e:
                logger.error(f"批改出错: {e}")
                return self._get_default_result(max_score)

    def _build_grading_prompt(self, question_text: str, student_answer: str,
                              max_score: float, subject_type: str) -> str:
        return f"""你是一位严格的{subject_type}老师，请批改以下选择题。

【题目】
{question_text}

【学生答案】
{student_answer}

【满分】{max_score}分

【批改规则】
1. 选择题：答案完全正确得满分，错误得0分
2. 给出详细的解析，说明正确答案是什么、为什么
3. 如果回答错误，给出学习建议

【返回格式】严格返回JSON：
{{
    "score": 分数(数字),
    "correct_answer": "正确答案选项",
    "comment": "详细评语，包含正确答案和解析",
    "error_analysis": "错误分析（如果正确则写'回答正确'）",
    "knowledge_point": "考察的知识点",
    "suggestions": ["学习建议1", "学习建议2"]
}}"""

    def _parse_grading_response(self, content: str, max_score: float) -> GradingResult:
        content = content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        json_data = self._extract_json(content)

        if json_data:
            score = float(json_data.get("score", 0))
            comment = json_data.get("comment", "")
            suggestions = json_data.get("suggestions", [])
            correct_answer = json_data.get("correct_answer", "")
            error_analysis = json_data.get("error_analysis", "")
            knowledge_point = json_data.get("knowledge_point", "")

            full_comment = comment
            if correct_answer:
                full_comment = f"正确答案：{correct_answer}\n{full_comment}"
            if error_analysis and error_analysis != "回答正确":
                full_comment += f"\n\n错误分析：{error_analysis}"
            if knowledge_point:
                full_comment += f"\n\n知识点：{knowledge_point}"

            return GradingResult(
                score=min(score, max_score),
                max_score=max_score,
                comment=full_comment,
                suggestions=suggestions
            )

        return self._get_default_result(max_score)

    def _extract_json(self, content: str) -> Optional[Dict]:
        json_patterns = [
            r'\{[^{}]*"score"[^{}]*\}',
            r'\{[\s\S]*"score"[\s\S]*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    continue

        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(content[start:end + 1])
        except:
            pass

        return None

    def _get_default_result(self, max_score: float) -> GradingResult:
        return GradingResult(
            score=0,
            max_score=max_score,
            comment="AI批改暂时不可用，请稍后重试",
            suggestions=["建议联系老师手动批改"]
        )


class BatchGradingProcessor:
    """批量批改处理器"""

    def __init__(self, api_key: str, api_type: str = "aliyun", max_concurrent: int = 3):
        self.llm_engine = LLMGradingEngine(api_key, api_type)
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
            is_correct = result.score == result.max_score and result.max_score > 0
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
                    comment=f"批改失败: {str(result)}",
                    suggestions=["请重试或联系老师"]
                ))
            else:
                grading_results.append(result)
        return grading_results


def clean_text(text: str) -> str:
    """清理文本：去掉首尾空白、去掉开头的多余空行、规范化换行"""
    if not text:
        return ""

    # 去掉首尾空白
    text = text.strip()

    # 把连续的多个换行（3个以上）压缩成2个换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 把 \\n 转成真正的换行
    text = text.replace('\\n', '\n')

    # 再次去掉首尾空白
    text = text.strip()

    return text


def safe_html(text: str) -> str:
    """安全转义HTML，并处理换行"""
    text = clean_text(text)
    # 转义HTML特殊字符
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    # 换行转<br>
    text = text.replace('\n', '<br>')
    return text


def generate_interactive_html(image_base64: str,
                              questions_data: List[Dict],
                              grading_results: List[GradingResult],
                              total_score: float,
                              total_max_score: float) -> str:
    """生成交互式HTML"""

    percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0

    # 构建按钮数据 - 预先处理好文本
    buttons_json_list = []
    for i, (question, result) in enumerate(zip(questions_data, grading_results)):
        bbox = question.get("bbox", [50, 100 + i * 250, 700, 200])
        x, y, w, h = bbox
        is_correct = result.score == result.max_score and result.max_score > 0

        # 清理并转义文本
        comment_cleaned = safe_html(result.comment or "暂无评语")
        question_cleaned = safe_html(question.get("question_text", ""))
        student_answer_cleaned = safe_html(question.get("student_answer", ""))
        suggestions_cleaned = [safe_html(s) for s in (result.suggestions or [])]

        buttons_json_list.append({
            "id": f"q_{i + 1}",
            "num": i + 1,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "is_correct": is_correct,
            "score": result.score,
            "max_score": result.max_score,
            "student_answer": student_answer_cleaned,
            "question_text": question_cleaned,
            "comment": comment_cleaned,
            "suggestions": suggestions_cleaned,
        })

    buttons_json = json.dumps(buttons_json_list, ensure_ascii=False)

    score_class = "excellent" if percentage >= 80 else "good" if percentage >= 60 else "needs-work"

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI作业批改 - 点击按钮查看解析</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Microsoft YaHei", "PingFang SC", sans-serif;
            background: #f0f2f5;
            min-height: 100vh;
        }}

        .top-bar {{
            background: white;
            padding: 12px 24px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
            display: flex;
            align-items: center;
            gap: 16px;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .top-bar .title {{ font-size: 16px; font-weight: 600; color: #333; flex: 1; }}

        .total-score {{
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 15px;
        }}
        .total-score.excellent {{ background: #E8F5E9; color: #2E7D32; }}
        .total-score.good {{ background: #FFF3E0; color: #E65100; }}
        .total-score.needs-work {{ background: #FFEBEE; color: #C62828; }}

        .main-area {{
            display: flex;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
            align-items: flex-start;
        }}

        .image-panel {{
            flex: 1;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            padding: 20px;
            position: relative;
        }}

        .img-wrapper {{
            position: relative;
            display: inline-block;
            line-height: 0;
        }}

        .img-wrapper img {{
            max-width: 100%;
            height: auto;
            display: block;
            border-radius: 4px;
        }}

        .btn-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}

        .q-btn {{
            position: absolute;
            cursor: pointer;
            border: none;
            border-radius: 20px;
            font-weight: 700;
            font-size: 13px;
            color: white;
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 10px 18px;
            transition: all 0.2s;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            z-index: 10;
            white-space: nowrap;
        }}

        .q-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        }}

        .q-btn.active {{
            transform: scale(1.06);
            box-shadow: 0 6px 24px rgba(0,0,0,0.35);
            z-index: 20;
        }}

        .q-btn.correct {{ background: linear-gradient(135deg, #43A047, #66BB6A); }}
        .q-btn.wrong {{ background: linear-gradient(135deg, #E53935, #EF5350); }}

        .detail-sidebar {{
            width: 380px;
            flex-shrink: 0;
            position: sticky;
            top: 80px;
        }}

        .detail-card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            overflow: hidden;
            transition: opacity 0.3s;
        }}
        .detail-card.hidden {{ opacity: 0.45; }}

        .card-header {{
            padding: 14px 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid #eee;
        }}
        .card-header.correct {{ background: linear-gradient(135deg, #E8F5E9, #C8E6C9); }}
        .card-header.wrong {{ background: linear-gradient(135deg, #FFEBEE, #FFCDD2); }}

        .card-header .status-emoji {{ font-size: 22px; }}
        .card-header .card-title {{ font-size: 15px; font-weight: 600; }}

        .card-body {{
            padding: 18px;
            max-height: 55vh;
            overflow-y: auto;
        }}

        .answer-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
            padding: 10px 14px;
            background: #f8f8f8;
            border-radius: 8px;
        }}
        .answer-row .label {{ color: #888; font-size: 13px; }}
        .answer-row .value {{ 
            font-size: 20px; 
            font-weight: 700;
            padding: 4px 16px;
            border-radius: 6px;
        }}
        .answer-row .value.correct {{ background: #E8F5E9; color: #2E7D32; }}
        .answer-row .value.wrong {{ background: #FFEBEE; color: #C62828; }}

        .score-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }}
        .score-big {{
            font-size: 34px;
            font-weight: 800;
        }}
        .score-big.high {{ color: #2E7D32; }}
        .score-big.low {{ color: #C62828; }}

        .comment-box {{
            background: #fafafa;
            padding: 14px;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 14px;
            margin-bottom: 10px;
        }}

        .comment-box p {{
            margin: 0 0 8px 0;
        }}

        .comment-box p:last-child {{
            margin-bottom: 0;
        }}

        .suggestion-box {{
            background: #FFF8E1;
            padding: 14px;
            border-radius: 8px;
            border-left: 3px solid #FFC107;
        }}
        .suggestion-box li {{
            margin: 5px 0 5px 16px;
            font-size: 13px;
            color: #555;
        }}

        .empty-state {{
            text-align: center;
            padding: 50px 20px;
            color: #bbb;
        }}
        .empty-state .icon {{ font-size: 50px; display: block; margin-bottom: 12px; }}

        .hint-bar {{
            text-align: center;
            color: #aaa;
            font-size: 12px;
            margin-top: 12px;
            padding: 8px;
            background: #fafafa;
            border-radius: 8px;
        }}
        .hint-bar kbd {{
            background: #e8e8e8;
            padding: 1px 6px;
            border-radius: 3px;
            border: 1px solid #ddd;
            font-family: monospace;
            font-size: 11px;
        }}

        @media (max-width: 900px) {{
            .main-area {{ flex-direction: column; }}
            .detail-sidebar {{ width: 100%; position: relative; top: 0; }}
        }}
    </style>
</head>
<body>
    <div class="top-bar">
        <span class="title">📝 AI作业批改系统</span>
        <span class="total-score {score_class}">总分 {total_score:.0f}/{total_max_score:.0f} ({percentage:.0f}%)</span>
    </div>

    <div class="main-area">
        <div class="image-panel" id="imagePanel">
            <div class="img-wrapper" id="imgWrapper">
                <img src="data:image/png;base64,{image_base64}" alt="作业原图" id="mainImg">
                <div class="btn-overlay" id="btnOverlay"></div>
            </div>
            <div class="hint-bar">
                💡 点击按钮查看解析 · 按 <kbd>1</kbd>-<kbd>{len(questions_data)}</kbd> 切换 · <kbd>Esc</kbd> 关闭 · 再次点击同一按钮隐藏
            </div>
        </div>

        <div class="detail-sidebar" id="detailSidebar">
            <div class="detail-card hidden" id="detailCard">
                <div class="card-header correct" id="cardHeader">
                    <span class="status-emoji" id="statusEmoji">👆</span>
                    <span class="card-title" id="cardTitle">点击按钮查看解析</span>
                </div>
                <div class="card-body" id="cardBody">
                    <div class="empty-state">
                        <span class="icon">📋</span>
                        点击题目旁边的按钮<br>查看详细批改解析
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const questionsData = {buttons_json};
        let activeQuestionId = null;
        let btnElements = [];

        function init() {{
            const img = document.getElementById('mainImg');
            if (img.complete) {{
                buildButtons();
            }} else {{
                img.onload = buildButtons;
            }}
        }}

        function buildButtons() {{
            const img = document.getElementById('mainImg');
            const overlay = document.getElementById('btnOverlay');
            const imgRect = img.getBoundingClientRect();

            const scaleX = imgRect.width / img.naturalWidth;
            const scaleY = imgRect.height / img.naturalHeight;

            overlay.innerHTML = '';
            overlay.style.width = imgRect.width + 'px';
            overlay.style.height = imgRect.height + 'px';
            btnElements = [];

            questionsData.forEach(data => {{
                const left = Math.round(data.x * scaleX) + 8;
                const top = Math.round(data.y * scaleY) + 8;

                const btn = document.createElement('button');
                btn.className = 'q-btn ' + (data.is_correct ? 'correct' : 'wrong');
                btn.style.cssText = `left: ${{left}}px; top: ${{top}}px;`;
                btn.innerHTML = `
                    <span>${{data.is_correct ? '✓' : '✗'}}</span>
                    第${{data.num}}题
                    <span style="font-size:11px;opacity:0.85;">${{data.score.toFixed(0)}}/${{data.max_score.toFixed(0)}}</span>
                `;
                btn.title = '点击查看/隐藏第' + data.num + '题解析';

                btn.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    toggleQuestion(data.id);
                }});

                overlay.appendChild(btn);
                btnElements.push({{ id: data.id, el: btn }});
            }});
        }}

        function toggleQuestion(questionId) {{
            if (activeQuestionId === questionId) {{
                hideDetail();
            }} else {{
                showDetail(questionId);
            }}
        }}

        function showDetail(questionId) {{
            const data = questionsData.find(q => q.id === questionId);
            if (!data) return;

            activeQuestionId = questionId;

            btnElements.forEach(b => {{
                if (b.id === questionId) {{
                    b.el.classList.add('active');
                }} else {{
                    b.el.classList.remove('active');
                }}
            }});

            const card = document.getElementById('detailCard');
            const header = document.getElementById('cardHeader');
            card.classList.remove('hidden');

            const isCorrect = data.is_correct;
            header.className = 'card-header ' + (isCorrect ? 'correct' : 'wrong');
            document.getElementById('statusEmoji').textContent = isCorrect ? '✅' : '❌';
            document.getElementById('cardTitle').textContent = 
                '第' + data.num + '题 - ' + (isCorrect ? '回答正确' : '回答错误');

            // 把<br>分割成段落，避免空行
            const commentText = data.comment || '暂无评语';
            const commentParts = commentText.split('<br>').filter(function(p) {{
                return p.trim() !== '';
            }});
            const commentHtml = commentParts.map(function(p) {{
                return '<p>' + p.trim() + '</p>';
            }}).join('');

            var suggestionsHtml = '';
            if (data.suggestions && data.suggestions.length > 0) {{
                var items = data.suggestions.map(function(s) {{
                    return '<li>' + s + '</li>';
                }}).join('');
                suggestionsHtml = `
                    <div class="suggestion-box">
                        <strong>💡 学习建议：</strong>
                        <ul>${{items}}</ul>
                    </div>
                `;
            }}

            document.getElementById('cardBody').innerHTML = `
                <div class="answer-row">
                    <span class="label">学生答案</span>
                    <span class="value ${{isCorrect ? 'correct' : 'wrong'}}">${{data.student_answer || '?'}}</span>
                </div>
                <div class="score-row">
                    <span style="color:#888;">得分</span>
                    <span class="score-big ${{isCorrect ? 'high' : 'low'}}">${{data.score.toFixed(0)}}</span>
                    <span style="color:#888;">/ ${{data.max_score.toFixed(0)}} 分</span>
                </div>
                <div class="comment-box">
                    <strong>📝 评语：</strong>
                    ${{commentHtml}}
                </div>
                ${{suggestionsHtml}}
            `;
        }}

        function hideDetail() {{
            activeQuestionId = null;

            btnElements.forEach(b => b.el.classList.remove('active'));

            var card = document.getElementById('detailCard');
            card.classList.add('hidden');
            document.getElementById('cardHeader').className = 'card-header correct';
            document.getElementById('statusEmoji').textContent = '👆';
            document.getElementById('cardTitle').textContent = '点击按钮查看解析';
            document.getElementById('cardBody').innerHTML = `
                <div class="empty-state">
                    <span class="icon">📋</span>
                    点击题目旁边的按钮<br>查看详细批改解析
                </div>
            `;
        }}

        document.getElementById('imagePanel').addEventListener('click', function(e) {{
            if (!e.target.closest('.q-btn')) {{
                hideDetail();
            }}
        }});

        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                hideDetail();
                return;
            }}

            var num = parseInt(e.key);
            if (num >= 1 && num <= questionsData.length) {{
                toggleQuestion('q_' + num);
                return;
            }}

            if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
                e.preventDefault();
                if (!activeQuestionId) {{
                    toggleQuestion('q_1');
                    return;
                }}
                var current = parseInt(activeQuestionId.replace('q_', ''));
                var next = e.key === 'ArrowRight' ? current + 1 : current - 1;
                if (next >= 1 && next <= questionsData.length) {{
                    toggleQuestion('q_' + next);
                }}
            }}
        }});

        var resizeTimer;
        window.addEventListener('resize', function() {{
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function() {{
                buildButtons();
                if (activeQuestionId) showDetail(activeQuestionId);
            }}, 300);
        }});

        document.addEventListener('DOMContentLoaded', init);
    </script>
</body>
</html>'''

    return html


def _print_results(grading_results, total_score, total_max_score):
    """打印批改结果"""
    percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0

    print("\n" + "=" * 60)
    print("🤖 AI批改结果")
    print("=" * 60)
    for i, r in enumerate(grading_results, 1):
        status = "✅ 正确" if r.score == r.max_score and r.max_score > 0 else "❌ 错误"
        print(f"\n第{i}题 {status}")
        print(f"  得分: {r.score:.1f}/{r.max_score:.1f}")
        comment_clean = clean_text(r.comment)
        preview = comment_clean[:120] + "..." if len(comment_clean) > 120 else comment_clean
        print(f"  评语: {preview}")
    print(f"\n📊 总分: {total_score:.1f}/{total_max_score:.1f} ({percentage:.0f}%)")
    print(f"\n📁 文件已保存:")
    print(f"   - 交互HTML: graded_result_interactive.html")
    print(f"\n💡 用浏览器打开HTML:")
    print(f"   - 点击按钮查看/隐藏题目解析")
    print(f"   - 按数字键1-{len(grading_results)}快速切换")
    print(f"   - 按ESC关闭解析")
    print("=" * 60)


async def process_homework(image_data: bytes,
                           questions_data: List[Dict],
                           api_key: str,
                           api_type: str = "aliyun") -> Dict:
    """处理作业批改的主函数 - 只生成HTML"""
    try:
        logger.info("🚀 开始AI作业批改...")

        # 1. AI批改
        processor = BatchGradingProcessor(api_key, api_type, max_concurrent=3)
        grading_results = await processor.process_batch(questions_data)

        # 2. 加载图片转base64
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"📷 原图尺寸: {image.size}")

        max_width = 1200
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)
            logger.info(f"   缩放到: {image.size}")

        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        # 3. 生成HTML
        total_score = sum(r.score for r in grading_results)
        total_max_score = sum(r.max_score for r in grading_results)

        html_content = generate_interactive_html(
            img_base64, questions_data, grading_results, total_score, total_max_score
        )

        html_path = "graded_result_interactive.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"🌐 交互HTML已保存: {html_path}")

        # 4. 打印结果
        _print_results(grading_results, total_score, total_max_score)

        return {
            "success": True,
            "grading_results": [
                {
                    "question_id": r.question_id,
                    "score": r.score,
                    "max_score": r.max_score,
                    "comment": clean_text(r.comment),
                    "suggestions": r.suggestions
                }
                for r in grading_results
            ],
            "total_score": total_score,
            "total_max_score": total_max_score,
            "percentage": (total_score / total_max_score * 100) if total_max_score > 0 else 0,
            "annotated_image": img_base64,
            "html_path": html_path,
            "message": "✅ 批改完成！只生成了HTML文件"
        }

    except Exception as e:
        logger.error(f"批改失败: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "AI批改失败"
        }