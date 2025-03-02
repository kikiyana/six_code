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
    left_queue = CircularQueue(len(s) + 1)

    for char in s:
        if char in "([{":
            left_queue.enqueue(char)
        elif char in ")]}":
            if left_queue.size == 0:
                return False
            top = left_queue.dequeue()
            if (char == ')' and top != '(') or (char == ']' and top != '[') or (char == '}' and top != '{'):
                return False
    
    return left_queue.size == 0

if __name__ == "__main__":
    for _ in range(5):
        print("请输入一个字符串：")
        user_input = input()
        if isValidEncoding(user_input):
            print("正确的输入")
        else:
            print("错误的输入")
