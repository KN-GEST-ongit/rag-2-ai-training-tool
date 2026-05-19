import numpy as np

from src.envs.web_env import WebsocketEnv
from gymnasium.spaces import Box, Discrete


class WebsocketPong(WebsocketEnv):
    def __init__(self):
        super().__init__()
        self.action_space = Discrete(3)
        self.observation_space = Box(
            low=np.array([0, 0, 0, -100, -100], dtype=np.float32),
            high=np.array([600, 1000, 600, 100, 100], dtype=np.float32),
            dtype=np.float32
        )
        self.playerId = None
        self.prevScoreLeft = self.internalLeftScore = 0
        self.prevScoreRight = self.internalRightScore = 0
        self.action_map = {0: -1, 1: 0, 2: 1}
        self.action = {'move': 0, 'start': 1}

    def update_observation(self, data: dict) -> None:
        self.state = data['state']
        self.playerId = data['playerId']
        if data['playerId'] == 0:
            self.curr_observation = np.array([
                self.state['leftPaddleY'],
                self.state['ballX'],
                self.state['ballY'],
                self.state['ballSpeedX'],
                self.state['ballSpeedY'],
            ], dtype=np.float32)
        else:
            self.curr_observation = np.array([
                self.state['rightPaddleY'],
                1000-self.state['ballX'],
                self.state['ballY'],
                -self.state['ballSpeedX'],
                self.state['ballSpeedY'],
            ], dtype=np.float32)

        self.new_obs_event.set()
        if not self.connection_event.is_set():
            self.connection_event.set()

    def get_done(self) -> bool:
        return self.internalLeftScore == 10 or self.internalRightScore == 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.first_step:
            self.action['move'] = 0
            self.new_action_event.set()
        self.internalLeftScore = self.internalRightScore = 0
        observation = self.get_observation()
        self.log_repeated_observation(observation, "reset")

        if self.state:
            self.prevScoreLeft = self.state['scoreLeft']
            self.prevScoreRight = self.state['scoreRight']
        else:
            self.prevScoreLeft = 0
            self.prevScoreRight = 0

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

        reward = 0
        if self.playerId == 0:
            if self.state['scoreLeft'] > self.prevScoreLeft:
                self.internalLeftScore += 1
                self.prevScoreLeft = self.state['scoreLeft']
                reward = 1
            elif self.state['scoreRight'] > self.prevScoreRight:
                self.internalRightScore += 1
                self.prevScoreRight = self.state['scoreRight']
                reward = -10
        else:
            if self.state['scoreRight'] > self.prevScoreRight:
                self.internalRightScore += 1
                self.prevScoreRight = self.state['scoreRight']
                reward = 1
            elif self.state['scoreLeft'] > self.prevScoreLeft:
                self.internalLeftScore += 1
                self.prevScoreLeft = self.state['scoreLeft']
                reward = -10

        info = {}
        return observation, reward, terminated, truncated, info

    def return_prediction(self) -> dict:
        if self.new_action_event.wait(timeout=self.timeout):
            self.new_action_event.clear()
            return self.action
        else:
            self.new_action_event.clear()
            return {'move': 0, 'start': 1}

    def render(self):
        pass

    def close(self):
        pass
