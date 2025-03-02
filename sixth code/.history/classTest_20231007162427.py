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

    for char in s:
        if char in "([{":
            queue.enqueue(char)
        elif char in ")]}":
            if queue.front == queue.rear:
                return False
            top = queue.dequeue()
            if (char == ')' and top != '(') or (char == ']' and top != '[') or (char == '}' and top != '{'):
                return False
    return queue.front == queue.rear

if __name__ == "__main__":
    for _ in range(5):
        print("请输入一个字符串：")
        user_input = input()
        if isValidEncoding(user_input):
            print("正确的输入")
        else:
            print("错误的输入")
