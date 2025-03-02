def get_index(theta):
    index = 0
    operators = ['+', '-', '*', '/', '(', ')', '#']
    if theta in operators:
        index = operators.index(theta)
    return index

def get_priority(theta1, theta2):
    priority = [['>', '>', '<', '<', '<', '>', '>'],
                ['>', '>', '<', '<', '<', '>', '>'],
                ['>', '>', '>', '>', '<', '>', '>'],
                ['>', '>', '>', '>', '<', '>', '>'],
                ['<', '<', '<', '<', '<', '=', '0'],
                ['>', '>', '>', '>', '0', '>', '>'],
                ['<', '<', '<', '<', '<', '0', '=']]
    
    index1 = get_index(theta1)
    index2 = get_index(theta2)
    return priority[index1][index2]

def calculate(b, theta, a):
    if theta == '+':
        return b + a
    elif theta == '-':
        return b - a
    elif theta == '*':
        return b * a
    elif theta == '/':
        return b / a

def get_answer(expression):
    opter = ['#']
    opval = []
    counter = 0
    i = 0
    while expression[i] != '#' or opter[-1] != '#':
        c = expression[i]
        if c.isdigit():
            if counter == 1:
                t = opval.pop()
                opval.append(t * 10 + int(c))
                counter = 1
            else:
                opval.append(int(c))
                counter += 1
            i += 1
        else:
            counter = 0
            if get_priority(opter[-1], c) == '<':
                opter.append(c)
                i += 1
            elif get_priority(opter[-1], c) == '=':
                opter.pop()
                i += 1
            elif get_priority(opter[-1], c) == '>':
                theta = opter.pop()
                a = opval.pop()
                b = opval.pop()
                opval.append(calculate(b, theta, a))
    return opval[0]

def main():
    t = int(input("Enter the number of expressions: "))
    expressions = []
    
    for i in range(t):
        expression = input(f"Enter expression {i + 1} (e.g., 5+6*3/(3-1)#): ")
        expressions.append(expression)
    
    for i, expression in enumerate(expressions):
        ans = get_answer(expression)
        print(f"Result {i + 1}: {ans}")

if __name__ == "__main__":
    main()
1