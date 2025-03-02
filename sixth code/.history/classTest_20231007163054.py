def op_priority(p):
    if p in ['(', '（']:
        return 0
    elif p in ['[', '【']:
        return 1
    elif p == '{':
        return 2
    elif p in [')', '）']:
        return 3
    elif p in [']', '】']:
        return 4
    elif p == '}':
        return 5
    else:
        return 6

def matching(a, b):
    if a == 0 and b == 3:
        return True
    elif a == 1 and b == 4:
        return True
    elif a == 2 and b == 5:
        return True
    else:
        return False

def is_valid_encoding(s):
    op = []
    parenthesis = []

    for char in s:
        if op_priority(char) != 6:
            parenthesis.append(op_priority(char))
    
    for i in range(len(parenthesis)):
        if parenthesis[i] == -1:
            continue
        if i < len(parenthesis) - 1:
            if matching(parenthesis[i], parenthesis[i + 1]):
                if op and op[-1] <= parenthesis[i]:
                    return False
                i += 1
                continue
        if not op:
            op.append(parenthesis[i])
        else:
            if parenthesis[i] <= 2:
                if op[-1] <= parenthesis[i]:
                    return False
                else:
                    op.append(parenthesis[i])
            else:
                if not matching(op[-1], parenthesis[i]):
                    return False
                has_match = False
                while not matching(op[0], parenthesis[i]):
                    for j in range(i + 1, len(parenthesis)):
                        if matching(op[0], parenthesis[j]):
                            op.pop(0)
                            parenthesis[j] = -1
                            has_match = True
                            break
                    if not has_match:
                        break
                if matching(op[0], parenthesis[i]):
                    op.pop(0)
                    continue
                else:
                    return False

    return not op

if __name__ == "__main__":
    while True:
        user_input = input("请输入您的算数表达式(输入e或E退出程序): ")
        if user_input.lower() == 'e':
            break

        if is_valid_encoding(user_input):
            print("输入的算式括号匹配")
        else:
            print("输入的算式括号不匹配")
