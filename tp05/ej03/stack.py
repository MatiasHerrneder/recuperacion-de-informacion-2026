from collections import deque
    

class Queue:
    def __init__(self):
        self._queue = deque()

    def enqueue(self, item):
        self._queue.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from an empty queue")
        return self._queue.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from an empty queue")
        return self._queue[0]

    def is_empty(self):
        return len(self._queue) == 0

    def size(self):
        return len(self._queue)