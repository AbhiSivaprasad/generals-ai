from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from flask import Flask, jsonify, request, make_response
from flask_socketio import SocketIO, send, emit, join_room, leave_room


import os

from enum import Enum

from src.live_game import LiveGame, LivePlayer

class UserState(Enum):
    IN_LOBBY = 1
    IN_GAME = 2
    IN_QUEUE = 3

__dirname__ = os.path.dirname(__file__)
ROOT_DIR = Path(__dirname__).parent

app = Flask(__name__)
socketio = SocketIO(app)

@dataclass
class PlayerStatus:
    status: UserState
    socket_id: str
    # If the user is actively in a game, the player is here.
    live_game_player: Optional[LivePlayer]

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


@app.route("/replay/<path:replay_path>")
def serve_replay(replay_path):
    print("request for replay path:", replay_path)

    # Construct the full path
    replay_full_path = ROOT_DIR / "resources/replays" / f"{replay_path}.json"

    # Check if the file exists and is within the allowed directory
    if not replay_full_path.is_file() or not replay_full_path.is_relative_to(
        ROOT_DIR / "resources/replays"
    ):
        return "Replay not found", 404

    with open(replay_full_path, "r") as file:
        r = make_response(file.read())
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r

@socketio.on('connect')
def handle_connect():
    connected_users[request.sid] = PlayerStatus(status=UserState.IN_LOBBY, socket_id=request.sid)
    # todo: allow for queueing of games
    all_users_in_lobby = [user for user in connected_users.values() if user.status == UserState.IN_LOBBY]
    if len(all_users_in_lobby) >= 2:
        # start a game
        new_game = LiveGame(all_users_in_lobby)


@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in connected_users:
        user = connected_users[request.sid]
        if user.status == UserState.IN_GAME and user.live_game_player:
            # end the game
            user.live_game_player.handle_disconnect()
        del connected_users[request.sid]

# move has a game_id, sid of player, and move
@socketio.on('set-move-queue')
def handle_move(data):
    game_id, player_sid, move = data
    game = active_games[game_id]
    game.players[player_sid].set_move_queue(move)



if __name__ == "__main__":
    app.run(debug=True, port=8000)

# TASKS
# implement socket server to visualize games in real time
# style visualization page
