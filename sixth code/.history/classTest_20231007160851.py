class Queue:
    def __init__(self, size):
        self.array = [''] * size
        self.size = size
        self.front = self.rear = -1

    def enqueue(self, c):
        if (self.rear + 1) % self.size == self.front:
            return False  # 队列已满
        if self.rear == -1:
            self.front = 0
            self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.size
        self.array[self.rear] = c
        return True

    def dequeue(self):
        if self.front == -1:
            return ''  # 队列为空
        c = self.array[self.front]
        if self.front == self.rear:
            self.front = self.rear = -1  # 队列中只有一个元素
        else:
            self.front = (self.front + 1) % self.size
        return c

def isValidEncoding(s):
    length = len(s)
    queue = Queue(length)
    mapping = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in '({[':
            queue.enqueue(char)
        else:
            if not queue or queue.dequeue() != mapping.get(char):
                return False
    return not queue

# 测试
print(isValidEncoding("()"))  # 应返回 True
print(isValidEncoding("()[]{}"))  # 应返回 True
print(isValidEncoding("(]"))  # 应返回 False
print(isValidEncoding("([)]"))  # 应返回 False
