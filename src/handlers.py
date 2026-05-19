import json

from tornado.websocket import WebSocketHandler
from gymnasium import Wrapper
from src.envs.web_env import WebsocketEnv
from abc import abstractmethod
from typing import final
from inspect import iscoroutinefunction


class BaseHandler(WebSocketHandler):

    def check_origin(self, origin: str) -> bool:
        return True

    def open(self):
        pass

    def on_close(self):
        pass

    @abstractmethod
    def send_prediction(self, data: dict) -> None:
        raise NotImplementedError

    @final
    async def on_message(self, message: str | bytes):
        data = json.loads(message)
        if iscoroutinefunction(self.send_prediction):
            await self.send_prediction(data)
        else:
            self.send_prediction(data)


class AiHandler(BaseHandler):

    def initialize(self, env: WebsocketEnv | Wrapper):
        self.env = env

    async def send_prediction(self, data: dict) -> None:
        self.env.unwrapped.update_observation(data)
        action = self.env.unwrapped.return_prediction()
        await self.write_message(json.dumps(action))
