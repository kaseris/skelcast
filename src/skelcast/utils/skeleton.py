from dataclasses import dataclass
from typing import List


@dataclass
class Joint:
    x: float
    y: float
    z: float


@dataclass
class Skeleton:
    joints: List[Joint]
