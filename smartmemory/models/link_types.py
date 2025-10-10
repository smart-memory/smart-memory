from enum import Enum


class LinkType(Enum):
    RELATED = "related"
    CONTRADICTS = "contradicts"
    EXPANDS = "expands"
    REFERENCES = "references"
    MANUAL = "manual"

    # Add more types as needed

    @classmethod
    def list(cls):
        return [lt.value for lt in cls]
