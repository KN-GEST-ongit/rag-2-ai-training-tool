import gym

from typing import List, Tuple, Type
from src.handlers import AiHandler
from src.bots import PongBot


EXTRA_ROUTES = {
    "WebsocketPong-v0": (r"/ws/bot/", PongBot)
}


def get_routes(env: gym.Env) -> List[Tuple[str, Type, dict]]:
    routes = [(r"/ws/agent/", AiHandler, dict(env=env))]
    env_name = env.spec.id
    if env_name in EXTRA_ROUTES:
        extra_route = EXTRA_ROUTES[env_name]
        routes.append(extra_route)
    return routes
