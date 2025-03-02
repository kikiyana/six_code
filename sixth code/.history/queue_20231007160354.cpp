#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *array;
    int size;
    int front;
    int rear;
} Queue;

Queue *initQueue(int size) {
    Queue *queue = (Queue *)malloc(sizeof(Queue));
    queue->array = (char *)malloc(size * sizeof(char));
    queue->size = size;
    queue->front = queue->rear = -1;
    return queue;
}

void enqueue(Queue *queue, char c) {
    if ((queue->rear + 1) % queue->size == queue->front) {
        return;  // 队列已满
    }
    if (queue->rear == -1) {
        queue->front = 0;
        queue->rear = 0;
    } else {
        queue->rear = (queue->rear + 1) % queue->size;
    }
    queue->array[queue->rear] = c;
}

char dequeue(Queue *queue) {
    if (queue->front == -1) {
        return '\0';  // 队列为空
    }
    char c = queue->array[queue->front];
    if (queue->front == queue->rear) {
        queue->front = queue->rear = -1;  // 队列中只有一个元素
    } else {
        queue->front = (queue->front + 1) % queue->size;
    }
    return c;
}

bool isValidEncoding(char *s) {
    int length = strlen(s);
    Queue *queue = initQueue(length);
    for (int i = 0; i < length; ++i) {
        if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
            enqueue(queue, s[i]);
        } else {
            if (queue->front == -1) {
                free(queue->array);
                free(queue);
                return false;
            }
            char left = dequeue(queue);
            if ((s[i] == ')' && left != '(') || (s[i] == ']' && left != '[') || (s[i] == '}' && left != '{')) {
                free(queue->array);
                free(queue);
                return false;
            }
        }
    }
    bool isValid = (queue->front == -1);
    free(queue->array);
    free(queue);
    return isValid;
}
