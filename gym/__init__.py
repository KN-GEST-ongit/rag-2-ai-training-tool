import gymnasium
import sys

sys.modules["gym"] = gymnasium

from gymnasium import spaces
sys.modules["gym.spaces"] = spaces