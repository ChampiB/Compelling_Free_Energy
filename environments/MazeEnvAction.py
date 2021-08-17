from enum import IntEnum


#
# Actions available in the maze environment.
#
class MazeEnvAction(IntEnum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,
    IDLE = 4

