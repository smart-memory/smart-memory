from dateutil import parser
from typing import Any


def parse_time(time_str: str) -> Any:
    """
    Parse an ISO 8601 time string into a datetime object.

    Args:
        time_str (str): The time string to parse.
    Returns:
        Any: The parsed datetime object, or None if input is None.
    """
    if time_str:
        return parser.isoparse(time_str)
    return None
