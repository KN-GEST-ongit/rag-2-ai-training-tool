import threading
import socket

from threading import Thread
from tornado.web import Application
from typing import List, Tuple, Type
from tornado.ioloop import IOLoop


def find_free_port(start_port: int = 8001, max_attempts: int = 10) -> int:
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', start_port)) != 0:
                return start_port
        start_port += 1
    raise RuntimeError("No free port found within the allowed range.")


def make_app(routes: List[Tuple[str, Type, dict]]) -> Application:
    return Application(routes)


def run_socket(port: int, routes: List[Tuple[str, Type, dict]]) -> Tuple[IOLoop, Thread]:
    print(f'Waiting for data exchange: http://localhost:{port}{routes[0][0]}')
    if len(routes) > 1:
        print("Helpers:")
        for route in routes[1:]:
            print(f'  http://localhost:{port}{route[0]}')
    app = make_app(routes)
    app.listen(port)
    ioloop = IOLoop.current()
    server_thread = threading.Thread(target=ioloop.start)
    server_thread.start()
    return ioloop, server_thread


def stop_socket(ioloop, server_thread) -> None:
    ioloop.add_callback(ioloop.stop)
    server_thread.join()
