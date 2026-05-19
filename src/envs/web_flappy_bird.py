import numpy as np

from src.envs.web_env import WebsocketEnv
from gymnasium.spaces import Box, Discrete


class WebsocketFlappyBird(WebsocketEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(2)
        self.observation_space = Box(
            low=np.array([0, -20, 0.5, 5, -50, 100], dtype=np.float32),
            high=np.array([600, 90, 1, 15, 1900, 500], dtype=np.float32),
            dtype=np.float32
        )
        self.prevScore = 0
        self.prevFailCounter = 0
        self.should_start = True
        self.action = {'jump': 0}

    def update_observation(self, data: dict) -> None:
        self.state = data['state']
        nearest_obstacle = min(
            (o for o in self.state['obstacles'] if o["distanceX"] > 0),
            key=lambda o: o["distanceX"]
        )
        self.curr_observation = np.array([
            self.state['birdY'],
            self.state['birdSpeedY'],
            self.state['gravity'],
            self.state['jumpPowerY'],
            nearest_obstacle['distanceX'],
            nearest_obstacle['centerGapY']
        ])

        self.new_obs_event.set()
        if not self.connection_event.is_set():
            self.connection_event.set()

    def get_done(self) -> bool:
        return self.state['failCounter'] > self.prevFailCounter

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.first_step:
            self.action['jump'] = 1
            self.new_action_event.set()

        self.prevScore = 0
        self.prevFailCounter = self.state['failCounter'] if self.state else 0
        observation = self.get_observation()
        self.log_repeated_observation(observation, "reset")

        info = {}
        return observation, info

    def step(self, action: int):
        if self.first_step:
            self.first_step = False
            self.action['jump'] = 1
        else:
            self.action['jump'] = int(action)

        self.new_action_event.set()
        observation = self.get_observation()
        self.log_repeated_observation(observation, "step")
        terminated = self.get_done()
        truncated = False

        if self.state['score'] > self.prevScore:
            self.prevScore = self.state['score']
            reward = 1
        elif terminated:
            reward = -1
        else:
            reward = 0

        info = {}
        return observation, reward, terminated, truncated, info

    def return_prediction(self) -> dict:
        if self.new_action_event.wait(timeout=self.timeout):
            self.new_action_event.clear()
            return self.action
        else:
            self.new_action_event.clear()
            return {'jump': 0}

    def render(self):
        pass

    def close(self):
        pass
