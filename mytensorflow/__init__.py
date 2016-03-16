# Set some package-wide constants before importing other modules.
DEFAULT_DEVICE = "/gpu:0"
MOVING_AVERAGE_DECAY_FOR_LOSS = 0.999
MOVING_AVERAGE_DECAY_FOR_VARS = 0.999

from .conv import *
from .data import source
import utils
import trainer
