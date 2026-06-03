import time


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    h, m = divmod(sec, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Timer:
    def __init__(self):
        self.t = 0
        self.count = 0

        self.start_time = None
        self.total_time = 0
        self.is_running = False

    def __enter__(self):
        """Start the timer when entering a context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer when exiting a context."""
        self.stop()
        return False

    @property
    def elapsed(self):
        """Return the current elapsed time without stopping the timer."""
        if self.is_running:
            return self.total_time + (time.time() - self.start_time)
        return self.total_time

    @property
    def avg(self):
        """Return the average time per count."""
        if self.count == 0:
            return 0
        return self.total_time / self.count

    @property
    def fps(self):
        """Return the average fps."""
        if self.elapsed == 0:
            return 1
        return self.count / self.elapsed

    def start(self):
        """Start the timer."""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True

    def stop(self):
        """Stop the timer and accumulate elapsed time."""
        if self.is_running:
            self.single_time = time.time() - self.start_time
            self.total_time += self.single_time
            self.is_running = False
            self.count += 1
            return self.total_time
        return None

    def reset(self):
        """Reset the timer."""
        self.t = 0
        self.count = 0
        self.total_time = 0
        self.start_time = None
        self.is_running = False
