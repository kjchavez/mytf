# Set some package-wide constants before importing other modules.
DEFAULT_DEVICE = "/gpu:0"
MOVING_AVERAGE_DECAY_FOR_LOSS = 0.999
MOVING_AVERAGE_DECAY_FOR_VARS = 0.999

TRAIN_FEED_DICT_FN = "train_feed_dict_fn"
EVAL_FEED_DICT_FN = "eval_feed_dict_fn"

from .conv import *
from .fully_connected import *
from .data import source
import utils
import train
