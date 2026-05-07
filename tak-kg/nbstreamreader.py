from threading import Thread
from queue import Queue, Empty   # Python 2 had Queue.Queue; Python 3 uses queue.Queue


class NonBlockingStreamReader:
    def __init__(self, stream, encoding="utf-8", errors="replace"):
        self._s = stream
        self._q = Queue()
        self._encoding = encoding
        self._errors = errors

        def _populateQueue(stream, queue):
            while True:
                line = stream.readline()
                if line:
                    if isinstance(line, (bytes, bytearray)):
                        line = line.decode(self._encoding, errors=self._errors)
                    queue.put(line)
                # stream ended; keep thread alive (original behavior)

        self._t = Thread(target=_populateQueue, args=(self._s, self._q))
        self._t.daemon = True
        self._t.start()

    def readline(self, timeout=None):
        try:
            return self._q.get(block=(timeout is not None), timeout=timeout)
        except Empty:
            return None