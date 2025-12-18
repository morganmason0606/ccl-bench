from typing import Dict


def metric_cal(directory: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate the kernel compute time from the exported sqlite file from nsys.

    Args:
        directory (str): The directory path containing the exported sqlite file from nsys.

    Returns:
        Dict[str, float]:
    """
    allgather = {
        "total_ms": 0.0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "std_ms": 0.0,
    }

    allreduce = {
        "total_ms": 0.0,
        "mean_ms": 0.0,
        "median_ms": 0.0,
        "std_ms": 0.0,
    }

    return {
        "allgather": allgather,
        "allreduce": allreduce
    }