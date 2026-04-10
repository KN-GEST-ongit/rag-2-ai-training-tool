from gymnasium.wrappers import NormalizeObservation, NormalizeReward
from typing import Dict
import numpy as np


class NormObsWrapper(NormalizeObservation):
    def __init__(self, env, stats: Dict = None, training=True):
        super().__init__(env)
        if stats is not None:
            self.obs_rms.mean = stats.get("obs_mean", 0.0)
            self.obs_rms.var = stats.get("obs_var", 1.0)
        self.training = training

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self.training:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormRewWrapper(NormalizeReward):
    def __init__(self, env, stats: Dict = None, training=True):
        super().__init__(env)
        if stats is not None:
            self.return_rms.mean = stats.get("rew_mean", 0.0)
            self.return_rms.var = stats.get("rew_var", 1.0)
        self.training = training

    def reward(self, rew: float) -> float:
        return rew / np.sqrt(self.return_rms.var + self.epsilon)
