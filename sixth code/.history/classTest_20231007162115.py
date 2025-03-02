class CircularQueue:
    def __init__(self, capacity):
        self.queue = [None] * (capacity + 1)
        self.front = 0
        self.rear = 0
        self.maxSIZE = capacity

    def enqueue(self, item):
        if (self.rear + 1) % (self.maxSIZE + 1) == self.front:
            return False
        self.queue[self.rear] = item
        self.rear = (self.rear + 1) % (self.maxSIZE + 1)
        return True

    def dequeue(self):
        if self.front == self.rear:
            return None
        item = self.queue[self.front]
        self.front = (self.front + 1) % (self.maxSIZE + 1)
        return item

def isValidEncoding(s):
    queue = CircularQueue(len(s) + 1)
    num1 = num2 = num3 = 0
    a = b = d = 0

    if len(s) % 2 != 0:
        return False

    if s[0] in [')', ']', '}']:
        return False
    else:
        if s[0] == '(':
            queue.enqueue(s[0])
            a = 1
            num1 += 1
        elif s[0] == '[':
            queue.enqueue(s[0])
            b = 1
            num2 += 1
        elif s[0] == '{':
            queue.enqueue(s[0])
            d = 1
            num3 += 1

    i = 1
    c = 1
    while s[i] != '\0':
        if s[i] == '(':
            queue.enqueue(s[i])
            queue.dequeue()
            c = 0
            a = 1
            num1 += 1
        elif s[i] == '[':
            queue.enqueue(s[i])
            queue.dequeue()
            c = 0
            b = 1
            num2 += 1
        elif s[i] == '{':
            queue.enqueue(s[i])
            queue.dequeue()
            c = 0
            d = 1
            num3 += 1
        elif s[i] == ')':
            num1 -= 1
            if queue.queue[queue.front] != '(' and a == 1:
                return False
            elif queue.queue[queue.front] == '(':
                a = 0
        elif s[i] == ']':
            num2 -= 1
            if queue.queue[queue.front] != '[' and b == 1:
                return False
            elif queue.queue[queue.front] == '[':
                b = 0
        elif s[i] == '}':
            num3 -= 1
            if queue.queue[queue.front] != '{' and d == 1:
                return False
            elif queue.queue[queue.front] == '{':
                d = 0
        i += 1
        c = 2

    if num1 % 2 != 0 or num2 % 2 != 0 or num3 % 2 != 0:
        return False

    if c != 0:
        return True

    return False

if __name__ == "__main__":
    for _ in range(5):
        print("请输入一个字符串：")
        user_input = input()
        if isValidEncoding(user_input):
            print("正确的输入")
        else:
            print("错误的输入")
