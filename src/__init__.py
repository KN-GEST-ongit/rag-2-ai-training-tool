from gym.envs.registration import register

register(
    id='WebsocketPong-v0',
    entry_point='src.envs.web_pong:WebsocketPong',
)

register(
    id='WebsocketFlappyBird-v0',
    entry_point='src.envs.web_flappy_bird:WebsocketFlappyBird',
)

register(
    id='WebsocketHappyJump-v0',
    entry_point='src.envs.web_happy_jump:WebsocketHappyJump',
)

register(
    id='WebsocketCrossyRoad-v0',
    entry_point='src.envs.web_crossy_road:WebsocketCrossyRoad',
)