import tkinter as tk

def calculate():
    expression = entry.get()
    try:
        result = evaluate_expression(expression)
        result_label.config(text=f"Result: {result}")
    except Exception as e:
        result_label.config(text="Error")

def evaluate_expression(expression):
    operators = {'+': 1, '-': 1, '*': 2, '/': 2}
    stack = []
    output = []
    
    for token in expression:
        if token.isdigit() or token == '.':
            output.append(token)
        elif token in operators:
            while (stack and stack[-1] in operators and
                   operators[token] <= operators[stack[-1]]):
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            if not stack or stack[-1] != '(':
                raise Exception("Mismatched parentheses")
            stack.pop()
    
    while stack:
        if stack[-1] == '(':
            raise Exception("Mismatched parentheses")
        output.append(stack.pop())
    
    result_stack = []
    for token in output:
        if token.isdigit() or ('.' in token):
            result_stack.append(float(token))
        elif token in operators:
            b = result_stack.pop()
            a = result_stack.pop()
            if token == '+':
                result_stack.append(a + b)
            elif token == '-':
                result_stack.append(a - b)
            elif token == '*':
                result_stack.append(a * b)
            elif token == '/':
                result_stack.append(a / b)
    
    if len(result_stack) != 1:
        raise Exception("Invalid expression")
    
    return result_stack[0]

# 创建主窗口
window = tk.Tk()
window.title("Calculator")

# 创建输入框
entry = tk.Entry(window, width=20)
entry.grid(row=0, column=0, columnspan=4)

# 创建按钮
buttons = [
    '7', '8', '9', '/',
    '4', '5', '6', '*',
    '1', '2', '3', '-',
    '0', '+', '=', 'C',
    '(', ')'
]

row, col = 1, 0
for button in buttons:
    if button == '=':
        tk.Button(window, text=button, command=calculate).grid(row=row, column=col)
    elif button == 'C':
        tk.Button(window, text=button, command=lambda b=button: entry.delete(0, 'end')).grid(row=row, column=col)
    else:
        tk.Button(window, text=button, command=lambda b=button: entry.insert('end', b)).grid(row=row, column=col)
    col += 1
    if col > 3:
        col = 0
        row += 1

# 创建结果标签
result_label = tk.Label(window, text="", width=20)
result_label.grid(row=row, column=0, columnspan=4)

# 启动主循环
window.mainloop()
