import logging
from typing import Mapping, Optional

from . import rank_zero_only as rank_utils

# Configure logging to output to console
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])


def rank_prefixed_message(message: str, rank: int) -> str:
    """Prefix a message with the rank of the process.

    :param message: The message to prefix.
    :param rank: The rank of the process.
    :return: The prefixed message.
    """
    return f"[Rank {rank}] {message}"


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes
        with their rank prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank
        of the process it's being logged from. If `'rank'` is provided, then the log will only
        occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = rank_utils.get_rank()
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Log a debug message.

        :param msg: The message to log.
        :param rank: The rank to log at.
        """
        self.log(logging.DEBUG, msg, rank, *args, **kwargs)

    def info(self, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Log an info message.

        :param msg: The message to log.
        :param rank: The rank to log at.
        """
        self.log(logging.INFO, msg, rank, *args, **kwargs)

    def warning(self, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Log a warning message.

        :param msg: The message to log.
        :param rank: The rank to log at.
        """
        self.log(logging.WARNING, msg, rank, *args, **kwargs)

    def error(self, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Log an error message.

        :param msg: The message to log.
        :param rank: The rank to log at.
        """
        self.log(logging.ERROR, msg, rank, *args, **kwargs)

    def critical(self, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Log a critical message.

        :param msg: The message to log.
        :param rank: The rank to log at.
        """
        self.log(logging.CRITICAL, msg, rank, *args, **kwargs)

    def exception(self, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Log an exception message.

        :param msg: The message to log.
        :param rank: The rank to log at.
        """
        kwargs.setdefault("exc_info", True)
        self.log(logging.ERROR, msg, rank, *args, **kwargs)
