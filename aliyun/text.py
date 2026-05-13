"""
测试B同学接口 — 模拟A同学传入numpy数组
运行: python test_b_interface.py
"""

import asyncio
import cv2
import numpy as np
from pathlib import Path
from python大作业.aliyun.aliyun_ocr import get_engine

async def test_with_file():
    """测试1：从文件读取图片，模拟A同学传入"""
    print("=" * 60)
    print("测试1：模拟A同学传入文件图片")
    print("=" * 60)

    engine = get_engine()

    # 模拟A同学读图
    test_img = "test_imgs/sample_homework.jpg"
    img = cv2.imdecode(np.fromfile(test_img, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"❌ 找不到 {test_img}")
        return

    print(f"✅ 图片加载成功: {img.shape}")

    # 传入B同学接口
    result = await engine.recognize_from_array(img, Path(test_img).name)

    if result.error:
        print(f"❌ 识别失败: {result.error}")
    else:
        print(f"✅ 识别成功! {result.total_questions}道题, 耗时{result.processing_time:.1f}s")
        for q in result.questions:
            print(f"\n  --- 第{q.question_id}题 [{q.question_type}] ---")
            print(f"  题目: {q.question_text[:100]}")
            print(f"  答案: {q.answer_text[:100]}")


async def test_with_synthetic():
    """测试2：用合成图片测试（不依赖文件）"""
    print("\n" + "=" * 60)
    print("测试2：模拟A同学传入合成图片")
    print("=" * 60)

    engine = get_engine()

    # 创建一张模拟的"预处理后"图片
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    cv2.putText(img, "1. 25 x 4 = ?", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "Ans: 100", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(img, "2. 360 / 90 = ?", (50, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "Ans: 4", (50, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    print(f"✅ 合成图片: {img.shape}")

    # 传入B同学
    result = await engine.recognize_from_array(img, "合成测试.jpg")

    if result.error:
        print(f"⚠️ 合成图识别: {result.error} (正常，合成图可能无手写)")
    else:
        print(f"✅ 识别成功! {result.total_questions}道题")


async def test_d_student_interface():
    """测试3：模拟D同学调用 ocr_recognize()"""
    print("\n" + "=" * 60)
    print("测试3：模拟D同学调用 ocr_recognize()")
    print("=" * 60)

    from python大作业.aliyun.aliyun_ocr import ocr_recognize

    # A同学传来的图片
    test_img = "test_imgs/sample_homework.jpg"
    processed_img = cv2.imdecode(np.fromfile(test_img, dtype=np.uint8), cv2.IMREAD_COLOR)

    if processed_img is None:
        print(f"❌ 找不到 {test_img}")
        return

    # D同学调用
    question_text, answer_text = await ocr_recognize(processed_img)

    print(f"✅ 题目文本长度: {len(question_text)} 字符")
    print(f"✅ 答案文本长度: {len(answer_text)} 字符")
    print(f"\n题目预览:\n{question_text[:300]}")
    print(f"\n答案预览:\n{answer_text[:300]}")


if __name__ == "__main__":
    asyncio.run(test_with_file())
    asyncio.run(test_with_synthetic())
    asyncio.run(test_d_student_interface())