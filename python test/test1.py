# 简单的Python程序 - 打印问候语

def main():
    # 打印基本问候
    print("Hello, World!")

    # 获取用户输入
    name = input("请输入您的名字: ")

    # 使用格式化输出
    print(f"你好，{name}！欢迎学习Python！")

    # 简单的计算示例
    number1 = 10
    number2 = 5
    result = number1 + number2
    print(f"{number1} + {number2} = {result}")


# 程序入口点
if __name__ == "__main__":
    main()