#include <iostream>
#include <stack>
#include <string>

using namespace std;

// 函数用于执行操作并将结果推回栈中
void performOperation(stack<double> &numStack, char op) {
    double operand2 = numStack.top();
    numStack.pop();
    double operand1 = numStack.top();
    numStack.pop();

    double result;

    switch (op) {
        case '+':
            result = operand1 + operand2;
            break;
        case '-':
            result = operand1 - operand2;
            break;
        case '*':
            result = operand1 * operand2;
            break;
        case '/':
            result = operand1 / operand2;
            break;
        default:
            cout << "Invalid operator: " << op << endl;
            exit(1);
    }

    numStack.push(result);
}

int main() {
    stack<double> numStack;
    string expression;

    cout << "Enter an expression (e.g., 3 + 4 * 2 - 6 / 3): ";
    getline(cin, expression);

    for (char c : expression) {
        if (isdigit(c)) {
            // 处理数字字符，将其转换为实际数值并入栈
            double num = c - '0';  // 将字符转换为数字
            numStack.push(num);
        } else if (c == ' ') {
            // 忽略空格
            continue;
        } else if (c == '+' || c == '-' || c == '*' || c == '/') {
            // 处理操作符，执行相应的操作
            performOperation(numStack, c);
        } else {
            // 非法字符
            cout << "Invalid character: " << c << endl;
            exit(1);
        }
    }

    // 最终栈中的元素就是计算结果
    if (!numStack.empty()) {
        double result = numStack.top();
        cout << "Result: " << result << endl;
    } else {
        cout << "Invalid expression." << endl;
    }

    return 0;
}