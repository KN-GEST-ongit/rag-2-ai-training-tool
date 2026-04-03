import numpy as np
import logging
from gym import Env
import threading

from threading import Event
from abc import ABC, abstractmethod
from typing import final, Union

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class InterruptableEvent(threading.Event):
    def wait(self, timeout=None):
        wait = super().wait  # get once, use often
        if timeout is None:
            while not wait(0.01):
                pass
        else:
            wait(timeout)


class WebsocketEnv(Env, ABC):
    def __init__(self):
        super().__init__()
        self._state = None
        self._connection_event = InterruptableEvent()
        self._new_obs_event = Event()
        self._new_action_event = Event()
        self._first_step = True
        self._curr_observation = np.array([])
        self._timeout = None
        self.__prev_observation = None
        self.__repetition_count = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value: dict):
        self._state = value

    @property
    def connection_event(self) -> Event:
        return self._connection_event

    @property
    def new_obs_event(self) -> Event:
        return self._new_obs_event

    @property
    def new_action_event(self) -> Event:
        return self._new_action_event

    @property
    def first_step(self):
        return self._first_step

    @first_step.setter
    def first_step(self, value: bool):
        self._first_step = value

    @property
    def curr_observation(self):
        return self._curr_observation

    @curr_observation.setter
    def curr_observation(self, value: np.array):
        self._curr_observation = value

    @property
    def timeout(self) -> Union[int, None]:
        if not self._first_step:
            self._timeout = 5
        return self._timeout

    @final
    def get_observation(self) -> np.array:
        self._new_obs_event.wait()
        self._new_obs_event.clear()
        return self.curr_observation

    @abstractmethod
    def update_observation(self, data: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def return_prediction(self) -> dict:
        raise NotImplementedError

    @final
    def log_repeated_observation(self, observation: np.array, method: str) -> None:
        if self.__prev_observation is not None and np.array_equal(observation, self.__prev_observation):
            self.__repetition_count += 1
            logging.warning("Repeated observation #%d in method %s", self.__repetition_count, method)
        self.__prev_observation = observation.copy()
