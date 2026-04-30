from ultralytics.engine.results import Results


class MultitaskResults:
    """A class to hold multitask prediction results."""

    def __init__(self, results_list: list[Results]):
        """
        Initialize MultitaskResults with a dictionary of results.

        Args:
            results_dict (dict[str, Results]): A dictionary where keys are task names and values are the corresponding
                result objects for each task.
        """
        self.results_dict = {f"task_{i}": results for i, results in enumerate(results_list)}
        self.speed = None