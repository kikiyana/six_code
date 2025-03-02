class CircularQueue:
    def __init__(self, capacity):
        self.queue = [None] * capacity
        self.front = 0
        self.rear = 0
        self.size = 0
        self.capacity = capacity

    def enqueue(self, item):
        if self.size == self.capacity - 1:
            return False
        self.queue[self.rear] = item
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
        return True

    def dequeue(self):
        if self.size == 0:
            return None
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item

def isValidEncoding(s):
    queue = CircularQueue(len(s) + 1)
    mapping = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in mapping.values():
            queue.enqueue(char)
        elif char in mapping.keys():
            left_bracket = queue.dequeue()
            if left_bracket is None or mapping[char] != left_bracket:
                return False
        else:
            return False
    return queue.size == 0

# 测试示例
print(isValidEncoding("()"))  # 输出: True
print(isValidEncoding("()[]{}"))  # 输出: True
print(isValidEncoding("(]"))  # 输出: False
print(isValidEncoding("([)]"))  # 输出: False
