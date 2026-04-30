from ultralytics.utils.ops import Profile
import torch
import time

class AdvancedProfile(Profile):
    """
    Advanced Profile class for timing code execution.

    Use as a decorator with @Profile() or as a context manager with 'with Profile():'. Provides accurate timing
    measurements with CUDA synchronization support for GPU operations.

    Stores per call elapsed time in 'dts' and accumulated time in 't'.

    Attributes:
        t (float): Accumulated time in seconds.
        dts (list): List of elapsed times for each call.
        device (torch.device): Device used for model inference.
        cuda (bool): Whether CUDA is being used for timing synchronization.

    Examples:
        Use as a context manager to time code execution
        >>> with Profile(device=device) as dt:
        ...     pass  # slow operation here
        >>> print(dt)  # prints "Elapsed time is 9.5367431640625e-07 s"

        Use as a decorator to time function execution
        >>> @Profile()
        ... def slow_function():
        ...     time.sleep(0.1)
    """

    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial accumulated time in seconds.
            device (torch.device, optional): Device used for model inference to enable CUDA synchronization.
        """
        super().__init__(t=t, device=device)
        self.dts = []
        if t > 0:
            self.dts.append(t)


    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        super().__exit__(type, value, traceback)
        self.dts.append(self.dt)  # store elapsed time

