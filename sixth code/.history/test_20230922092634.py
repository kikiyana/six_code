import tkinter as tk

def calculate():
    expression = entry.get()
    try:
        result = eval(expression)
        result_label.config(text=f"Result: {result}")
    except Exception as e:
        result_label.config(text="Error")

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
    '0', '+', '=', 'C'
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
