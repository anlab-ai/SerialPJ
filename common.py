from enum import Enum
from sre_constants import SUCCESS

class ErrorCode(Enum):
	SUCCESS = 1
	ERROR = 2
	INVALID_DATA = 3