class Queue:
    def __init__(self, size):
        self.array = [''] * (size + 1)  # 多一个空位，用于区分队列是空还是满
        self.size = size
        self.front = self.rear = 0

    def enqueue(self, c):
        if (self.rear + 1) % (self.size + 1) == self.front:
            return False  # 队列已满
        self.array[self.rear] = c
        self.rear = (self.rear + 1) % (self.size + 1)
        return True

    def dequeue(self):
        if self.front == self.rear:
            return ''  # 队列为空
        c = self.array[self.front]
        self.front = (self.front + 1) % (self.size + 1)
        return c

def isValidEncoding(s):
    length = len(s)
    queue = Queue(length)  # 留一个位置空着，用于区分队列是空还是满
    mapping = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '({[':
            if not queue.enqueue(char):
                return False  # 队列已满
        else:
            if queue.front == queue.rear or queue.dequeue() != mapping.get(char):
                return False
    return queue.front == queue.rear  # 队列为空时 front 和 rear 相等

# 测试
print(isValidEncoding("()"))  # 应返回 True
print(isValidEncoding("()[]{}"))  # 应返回 True
print(isValidEncoding("(]"))  # 应返回 False
print(isValidEncoding("([)]"))  # 应返回 False
