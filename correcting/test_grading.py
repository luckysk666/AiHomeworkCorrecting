# test_grading.py - 不告诉AI答案，让AI自己判断
import asyncio
from llm_grading import process_homework


async def main():
    # 请在这里填写你的API密钥
    API_KEY = "sk-d1180959db3f4c4f8df22442246ca23a"  # 你的API密钥
    API_TYPE = "aliyun"

    # 读取测试图片
    try:
        with open("test_homework.jpg", "rb") as f:
            image_data = f.read()
        print("✅ 已加载图片: test_homework.jpg")
    except FileNotFoundError:
        from PIL import Image
        img = Image.new('RGB', (1200, 900), color='white')
        img.save("test_homework.jpg")
        with open("test_homework.jpg", "rb") as f:
            image_data = f.read()
        print("已创建测试图片")

    # 准备题目数据 - 不告诉AI正确答案，让它自己判断！
    questions_data = [
        {
            "question_id": "Q1",
            "question_text": """某学者认为，鸦片战争只是这场战争（指鸦片战争）的直接原因而非根本原因，由于中西方对国际关系、贸易和司法管辖的观念大相径庭。即使没有鸦片，双方之间的冲突也照样会爆发。在该学者看来：
    A. 鸦片战争与鸦片战争无关。
    B. 鸦片战争促进中国近代化。
    C. 中西观念差异是根本原因。
    D. 鸦片战争爆发具有必然性。""",
            "student_answer": "D",
            "max_score": 10,
            "subject_type": "history",
            "bbox": [80, 200, 700, 280]
        },
        {
            "question_id": "Q2",
            "question_text": """"借来上帝结联盟，竖起军旗反满清。金殿难堪遥圣殿，北京苦闷对南京。"该诗句描写的是（ ）。
    A. 三元里人民抗英斗争。
    B. 太平天国运动。
    C. 义和团反帝运动。
    D. 中华民国成立。""",
            "student_answer": "B",
            "max_score": 10,
            "subject_type": "history",
            "bbox": [80, 550, 700, 250]
        },
        {
            "question_id": "Q3",
            "question_text": """近代上海开埠后不久，原来为广州贸易服务的人，买办，船民、运输工人大量转向上海，利用乡土关系来上海谋求职业的游民不断增加。这反映出，当时的上海（ ）。
    A. 贸易地位快速上升。
    B. 鸦片贸易合法化。
    C. 自然经济彻底破坏。
    D. 复归新运动中心。""",
            "student_answer": "B",
            "max_score": 10,
            "subject_type": "history",
            "bbox": [80, 880, 700, 230]
        }
    ]

    print(f"开始AI批改，共 {len(questions_data)} 道题目...")
    print("🤖 AI正在分析题目和学生答案...\n")

    # 执行AI批改
    result = await process_homework(image_data, questions_data, API_KEY, API_TYPE)

    # 输出结果
    if result["success"]:
        print(f"\n✅ AI批改完成!")
        print(f"📊 总分: {result['total_score']}/{result['total_max_score']} ({result['percentage']:.1f}%)")
        print(f"\n📁 结果文件: {result.get('html_path', 'graded_result_interactive.html')}")
        print(f"\n💡 用浏览器打开 HTML 文件即可查看交互式批改结果")
    else:
        print(f"\n❌ AI批改失败: {result.get('message', '未知错误')}")


if __name__ == "__main__":
    asyncio.run(main())