from argparse import Action
import asyncio
from dataclasses import dataclass
from socket import SocketIO
import socketio
import uuid
from flask_socketio import emit


from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster


class LiveGame(GameMaster):
    def __init__(self, player_sids: str):
        self.board = generate_board_state(15, 15, mountain_probability=0.2, city_probability=0.03)
        self.players = [LivePlayer(player_id) for player_id in player_sids]
        self.game_id = uuid.uuid4()
        self.is_playing = True
        self.turn = 0

    async def disseminate_board_state(self):
        # TODO: make this simultaneous
        for player in self.players:
            player.disseminate_board_state(self.board)

    async def handle_disconnect_player(self, player_id):
        self.is_playing = False

    async def play(self):
        """
        conduct game between players on given board
        :return: index of winning player or -1 if max turns reached
        """
        while self.board.terminal_status() == -1 and self.turn < self.max_turns and self.is_playing:
            # each player outputs a move given their view
            for moving_player_index, player in list(enumerate(self.players)):
                action = player.get_top_move()
                if action is None:
                    continue

                # check for validity of action
                if not self.board.is_action_valid(action, moving_player_index):
                    continue

                # update game board with player's action
                self.update_game_state(action)

            # game logic to add troops to generals, cities, and land on specific ticks
            self.add_troops_to_board()

            await asyncio.sleep(0.5)
            self.turn += 1

        return self.board.terminal_status()


class LivePlayer():
    # todo: separate player id from socket id?
    def __init__(self, player_id, username, live_game):
        self.player_id = player_id
        self.username = username
        self.game = live_game
        self.game_state = {}
    def set_move_queue(self, move_queue: list[Action]):
        self.move_queue = move_queue

    def handle_disconnect(self):
        self.game.handle_disconnect_player(self.player_id)

    def disseminate_game_start(self, board_state):
        emit('board_start', {'board_state': board_state}, to=self.socket_id)

    def disseminate_board_update(self, board_state):
        emit('board_update', {'board_state': board_state}, to=self.socket_id)
    
    def disseminate_game_over(self, winner_player_id):
        emit('game_over', {'winning_player': winner_player_id})

    def get_top_move(self) -> Action:
        return self.move_queue.pop(0)