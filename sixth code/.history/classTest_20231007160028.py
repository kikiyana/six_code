class Queue:
    def __init__(self, size):
        self.array = [''] * size
        self.size = size
        self.front = self.rear = -1

    def enqueue(self, c):
        if (self.rear + 1) % self.size == self.front:
            return  # 队列已满
        if self.rear == -1:
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.size
        self.array[self.rear] = c

    def dequeue(self):
        if self.front == -1:
            return ''  # 队列为空
        c = self.array[self.front]
        if self.front == self.rear:
            self.front = self.rear = -1  # 队列中只有一个元素
        else:
            self.front = (self.front + 1) % self.size
        return c

def is_valid_encoding(s):
    length = len(s)
    queue = Queue(length)
    for i in range(length):
        if s[i] in '({[':
            queue.enqueue(s[i])
        else:
            if queue.front == -1:
                return False
            left = queue.dequeue()
            if (s[i] == ')' and left != '(') or (s[i] == ']' and left != '[') or (s[i] == '}' and left != '{'):
                return False
    return queue.front == -1

# 测试
print(is_valid_encoding("()"))  # 应返回 True
print(is_valid_encoding("()[]{}"))  # 应返回 True
print(is_valid_encoding("(]"))  # 应返回 False
print(is_valid_encoding("([)]"))  # 应返回 False
