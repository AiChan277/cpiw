import queue
from typing import Optional
from privacycam.capture.frame import Frame

class FrameQueue:
    """A thread-safe queue for frames with drop policies when full."""
    
    def __init__(self, maxsize: int = 2, drop_policy: str = "oldest"):
        self.maxsize = maxsize
        self.drop_policy = drop_policy
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._dropped_count = 0

    def put(self, frame: Frame) -> bool:
        """
        Puts a frame in the queue.
        If the queue is full, drops the oldest frame and puts the new one.
        Returns True if the frame was placed without dropping, False if a frame was dropped.
        """
        dropped = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._dropped_count += 1
                dropped = True
            except queue.Empty:
                pass
        
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            return False

        return not dropped

    def get(self, timeout: float = 1.0) -> Optional[Frame]:
        """Gets a frame from the queue with an optional timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Clears all frames from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def size(self) -> int:
        """Returns the current number of frames in the queue."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Returns True if the queue is full."""
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        """Returns True if the queue is empty."""
        return self._queue.empty()

    @property
    def dropped_count(self) -> int:
        """Returns the number of frames dropped due to queue being full."""
        return self._dropped_count

    def reset_stats(self) -> None:
        """Resets the dropped frame counter."""
        self._dropped_count = 0
