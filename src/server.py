import asyncio
from dataclasses import dataclass
import logging
from typing import Optional
from pathlib import Path
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit, join_room, leave_room
from flask_openapi3 import OpenAPI, Info, Tag



import os

from enum import Enum
from src.api_types import ErrorResponse, ReplayResponse
from src.environment.action import Action, get_direction_from_str

from src.live_game import LiveGame, LivePlayer

class UserState(Enum):
    IN_LOBBY = 1
    IN_GAME = 2
    IN_QUEUE = 3

__dirname__ = os.path.dirname(__file__)
ROOT_DIR = Path(__dirname__).parent

print('name is', __name__)
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
info = Info(title="Chat API", version="1.0.0")
api = OpenAPI(__name__, info=info)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)  # or logging.CRITICAL to suppress more logs


@dataclass
class PlayerStatus:
    status: UserState
    socket_id: str
    # If the user is actively in a game, the player is here.
    live_game_player: Optional[LivePlayer]

# high level, the server mantains two pices of state:

# Mapping from socket id to the player
connected_users: dict[str, PlayerStatus] = {}

# Mapping from game id to the LiveGame class
active_games: dict[str, LiveGame] = {}

# TODO: add users for reals
USERS = [
    {
        "username": "shaya",
        "password": "password"
    },
    {
        "username": "dhruv",
        "password": "test_password"
    }
]


@app.get("/openapi")
def get_openapi_spec():
    return app.api_spec.to_dict()


@api.get("/replay/<path:replay_path>", responses={200: ReplayResponse, 404: ErrorResponse})
def serve_replay(replay_path):
    print("request for replay path:", replay_path)

    # Construct the full path
    replay_full_path = ROOT_DIR / "resources/replays" / f"{replay_path}.json"

    # Check if the file exists and is within the allowed directory
    if not replay_full_path.is_file() or not replay_full_path.is_relative_to(
        ROOT_DIR / "resources/replays"
    ):
        return ErrorResponse(detail="Replay not found"), 404

    with open(replay_full_path, "r") as file:
        r = make_response(file.read())
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r

@api.get("/join-game/<game_id>")
def join_game(game_id):
    return "joined game"

@socketio.on('connect')
def handle_connect(auth):
    print("CONNECTED USERS ARE", connected_users.keys())
    connected_users[request.sid] = PlayerStatus(status=UserState.IN_LOBBY, socket_id=request.sid, live_game_player=None)

def consider_starting_game():
    all_users_in_lobby = [user.socket_id for user in connected_users.values() if user.status == UserState.IN_QUEUE]
    if len(all_users_in_lobby) >= 2:
        print("starting game because there are enough users")
        # start a game
        new_game = LiveGame(all_users_in_lobby)
        # keep a direct mapping from socket id to live player
        for player in new_game.players:
            connected_users[player.player_id].live_game_player = player
            connected_users[player.player_id].status = UserState.IN_GAME
        asyncio.run(new_game.start_game())
        active_games[new_game.game_id] = new_game
    else:
        print("not enough users to start game")

@socketio.on('join-game')
def handle_join_game():
    print("joining game")
    connected_users[request.sid].status = UserState.IN_QUEUE
    consider_starting_game()


@socketio.on('disconnect')
def handle_disconnect(auth):
    if request.sid in connected_users:
        user = connected_users[request.sid]
        if user.status == UserState.IN_GAME and user.live_game_player:
            # end the game
            user.live_game_player.handle_disconnect()
        del connected_users[request.sid]

# move has a game_id, sid of player, and move
@socketio.on('set_move_queue')
def handle_move(data):
    if request.sid not in connected_users:
        return
    user = connected_users[request.sid]
    if user.status != UserState.IN_GAME or not user.live_game_player:
        return
    # deserialize back into python format
    actions = [Action(do_nothing=False, startx=action_dict['columnIndex'], starty=action_dict['rowIndex'], direction=get_direction_from_str(action_dict['direction'])) for action_dict in data]
    user.live_game_player.set_move_queue(actions)



if __name__ == "__main__":
    socketio.run(app, debug=True, port=8000)

# TASKS
# implement socket server to visualize games in real time
# style visualization page
