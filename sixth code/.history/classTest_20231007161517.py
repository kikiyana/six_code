def isValidEncoding(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if stack == [] or mapping[char] != stack.pop():
                return False
        else:
            return False
    return len(stack) == 0

# 测试示例
print(isValidEncoding("()"))  # 输出: True
print(isValidEncoding("()[]{}"))  # 输出: True
print(isValidEncoding("(]"))  # 输出: False
print(isValidEncoding("([)]"))  # 输出: False
