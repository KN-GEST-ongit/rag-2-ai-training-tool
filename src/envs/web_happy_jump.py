import numpy as np

from src.envs.web_env import WebsocketEnv
from gymnasium.spaces import Box, Discrete


class WebsocketHappyJump(WebsocketEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(3)
        self.observation_space = Box(
            low=np.array([0, 0, -5, -20, -1, 0, 0, -1, 0, 0], dtype=np.float32),
            high=np.array([400, 600, 5, 50, 1, 400, 600, 1, 400, 600], dtype=np.float32),
            dtype=np.float32
        )
        self.prevScore = 0
        self.prevFailCounter = 0
        self.should_start = True
        self.action = {'jump': 1, 'move': 0}
        self.action_map = {0: -1, 1: 0, 2: 1}

    def update_observation(self, data: dict) -> None:
        self.state = data['state']
        upper_platform = max(
            (p for p in self.state['platforms'] if p['y'] < self.state['playerY']),
            key=lambda p: p['y'],
            default={'directionX': 0, 'x': 200, 'y': 0}
        )
        lower_platform = min(
            (p for p in self.state['platforms'] if p['y'] > self.state['playerY']),
            key=lambda p: p['y'],
            default={'directionX': 0, 'x': 200, 'y': 600}
        )
        self.curr_observation = np.array([
            self.state['playerX'],
            self.state['playerY'],
            self.state['playerSpeedX'],
            self.state['playerSpeedY'],
            upper_platform['directionX'],
            upper_platform['x'],
            upper_platform['y'],
            lower_platform['directionX'],
            lower_platform['x'],
            lower_platform['y']
        ])

        self.new_obs_event.set()
        if not self.connection_event.is_set():
            self.connection_event.set()

    def get_done(self) -> bool:
        return self.state['failCounter'] > self.prevFailCounter

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.first_step:
            self.action['move'] = 0
            self.new_action_event.set()
        self.prevScore = 0
        self.prevFailCounter = self.state['failCounter'] if self.state else 0
        observation = self.get_observation()
        self.log_repeated_observation(observation, "reset")

        return observation, {}

    def step(self, action: int):
        if self.first_step:
            self.first_step = False

        self.action['move'] = self.action_map[int(action)]
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
            return {'jump': 1, 'move': 0}

    def render(self):
        pass

    def close(self):
        pass
