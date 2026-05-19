import numpy as np

from src.envs.web_env import WebsocketEnv
from gym.spaces import Box, Discrete
import logging

class WebsocketCrossyRoad(WebsocketEnv):
    def __init__(self):
        super().__init__()

        self.action_space = Discrete(5)

        low_bounds = [-8.0] + [-1.0] * 20
        high_bounds = [12.0] + [1.0] * 20

        self.observation_space = Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            dtype=np.float32
        )

        self.max_z_reached = 0
        self.prev_z = 0
        self.action = {'move': 0, 'action': 0}

    def update_observation(self, data: dict) -> None:
        self.state = data['state']

        px = self.state.get('playerX', 0)
        pz = self.state.get('playerZ', 0)
        lanes = self.state.get('lanes', [])

        vision_grid = []
        for dz in [2, 1, 0, -1]:
            lane = next((l for l in lanes if l['z'] == pz + dz), None)
            for dx in [-2, -1, 0, 1, 2]:
                if not lane:
                    vision_grid.append(-1.0)
                else:
                    is_safe_tile = self.is_safe(px + dx, lane)
                    vision_grid.append(1.0 if is_safe_tile else -1.0)

        obs_list = [px] + vision_grid
        self.curr_observation = np.array(obs_list, dtype=np.float32)

        self.new_obs_event.set()
        if not self.connection_event.is_set():
            self.connection_event.set()

    def get_done(self) -> bool:
        return self.state.get('isGameOver', False)

    def reset(self):
        if not self.first_step:
            self.action = {'move': 0, 'action': 1}
            self.new_action_event.set()

        self.max_z_reached = 0
        self.prev_z = 0

        observation = self.get_observation()
        self.log_repeated_observation(observation, "reset")
        return observation

    def step(self, action: int):
        if self.first_step:
            self.first_step = False

        self.action = {'move': int(action), 'action': 0}
        self.new_action_event.set()

        observation = self.get_observation()
        self.log_repeated_observation(observation, "step")

        done = self.get_done()

        current_z = self.state.get('playerZ', 0)

        if done:
            reward = -10.0
        else:
            if current_z > self.max_z_reached:
                reward = 1.0
                self.max_z_reached = current_z
            elif current_z < self.prev_z:
                reward = -0.5
            else:
                reward = -0.05

        self.prev_z = current_z

        info = {}
        return observation, reward, done, info

    def return_prediction(self) -> dict:
        if self.new_action_event.wait(timeout=self.timeout):
            self.new_action_event.clear()
            return self.action
        else:
            self.new_action_event.clear()
            return {'move': 0, 'action': 1}

    def is_safe(self, target_px, lane):
        if target_px < -8 or target_px > 12:
            return False
        lane_type = lane.get('type')
        obstacles = lane.get('obstacles', [])

        if lane_type == 'grass':
            for obs in obstacles:
                if obs.get('type') == 'tree':
                    if abs(obs.get('x', 0) - target_px) < 0.6:
                        return False
            return True

        if lane_type == 'road':
            lookahead_frames = 20
            for obs in obstacles:
                width = obs.get('width', 1.5)
                speed = obs.get('speed', 0)
                direction = obs.get('direction', 1)
                collision_threshold = (width / 2) + 0.2

                for frame in range(lookahead_frames):
                    future_x = obs.get('x', 0) + (speed * direction * frame)
                    if future_x > 20:
                        future_x -= 40
                    elif future_x < -20:
                        future_x += 40
                    if abs(future_x - target_px) < collision_threshold:
                        return False
            return True

        if lane_type == 'water':
            on_log = False
            for obs in obstacles:
                if obs.get('type') == 'log':
                    width = obs.get('width', 3.0)
                    safe_threshold = (width / 2) - 0.3
                    if abs(obs.get('x', 0) - target_px) < safe_threshold:
                        on_log = True
                        break
            return on_log
        return True

    def log_repeated_observation(self, observation: np.array, method: str) -> None:
        if not hasattr(self, 'cr_prev_observation'):
            self.cr_prev_observation = None
            self.cr_repetition_count = 0

        if self.cr_prev_observation is not None and np.array_equal(observation, self.cr_prev_observation):
            self.cr_repetition_count += 1
            if self.cr_repetition_count % 300 == 0:
                logging.warning("Repeated observation #%d in method %s", self.cr_repetition_count, method)
        else:
            self.cr_repetition_count = 0

        self.cr_prev_observation = observation.copy()

    def render(self):
        pass

    def close(self):
        pass