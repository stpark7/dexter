import os
from typing import Optional

# Global rank variable for tracking the current process rank
rank: Optional[int] = None


def get_rank() -> int:
    """Get the current process rank from environment variables."""
    global rank
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    return rank


def is_main_process() -> bool:
    """Check if this is the main process in distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    rank = int(os.environ.get("RANK", -1))
    return local_rank <= 0 and rank <= 0


def rank_zero_only(func):
    """Decorator to run a function only on rank 0 process."""

    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper
