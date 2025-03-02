def isValidEncoding(s):
    mapping = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    stack = []

    for char in s:
        if char in mapping.keys():
            stack.append(char)
        elif char in mapping.values():
            if not stack or mapping[stack.pop()] != char:
                return False
        else:
            return False

    return not stack

# 示例用法
input_string = input("请输入一个字符串：")
if isValidEncoding(input_string):
    print("输入的字符串是有效的编码。")
else:
    print("输入的字符串不是有效的编码。")
