import asyncio
from dataclasses import dataclass
from enum import Enum
from socket import SocketIO
from typing import Optional
from networkx import ExceededMaxIterations
import socketio
import uuid
from flask_socketio import emit
from src.agents import agent
from src.agents.agent import Agent
from src.environment import action, board
from src.environment.board import Board


from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster
from src.environment.tile import Tile


class LiveGame(GameMaster):
    def __init__(self, player_inits: list["ConnectedUser" | "BotSpec"]):
        self.board = generate_board_state(15, 15, mountain_probability=0.2, city_probability=0.03)
        self.players = [HumanPlayer(player_init.socket_id, player_init.username, self, player_index) if isinstance(player_init, ConnectedUser) else BotPlayer(model_spec=player_init) for player_index, player_init in enumerate(player_inits)]
        self.game_id = uuid.uuid4()
        self.turn = 0
        super().__init__(self.board, self.players)
        

    def disseminate_board_state(self):
        # TODO: make this simultaneous
        for player in self.players:
            player.disseminate_board_state(self.board)

    def process_board_update(self, previous_board_serialized):
        diff = self.get_board_diff(previous_board_serialized), self.board
        for player in self.players:
            player.on_board_update(self.board, diff)

    def handle_disconnect_player(self, player_id):
        for player in self.players:
            if player.player_id != player_id:
                emit('game_over', {"reason": "player_disconnected", "player_id": player_id}, to=player.player_id)
        self.is_playing = False

    async def start_game(self):
        self.is_playing = True
        for player in self.players:
            player.disseminate_game_start(self.board)
        winning_player = await self.play()
        emit('game_over', {"reason": "player_won", "player_id": self.players[winning_player].username})

    # We don't want to send the entire board state over the network every time,
    # so we compute the diff between the previous board state and the current board state
    # and send the diff over the network.
    def get_board_diff(self, previous_board_serialized: dict):
        tiles_changed = []
        for y in range(self.board.num_rows):
            for x in range(self.board.num_cols):
                if self.board.grid[y][x].serialize() != previous_board_serialized[y][x]:
                    tiles_changed.append(self.board.grid[y][x])
        return tiles_changed

    def update_player_stats(self):
        total_armies = [0 for _ in range(len(self.players))]
        total_land = [0 for _ in range(len(self.players))]
        for y in range(self.board.num_rows):
            for x in range(self.board.num_cols):
                tile = self.board.grid[y][x]
                if tile.player_index is not None:
                    total_land[tile.player_index] += 1
                    total_armies[tile.player_index] += tile.army
        for player in self.players:
            player.land_count = total_land[player.player_index]
            player.army_count = total_armies[player.player_index]


    async def play(self):
        """
        conduct game between players on given board
        players can be humans or bots, they both ascribe to the same interface
        :return: index of winning player or -1 if max turns reached
        """
        await asyncio.sleep(0.5)
        while self.board.terminal_status() == -1 and (self.max_turns is None or self.turn < self.max_turns) and self.is_playing:
            # serialize the board so that we can send the diff
            previous_board_serialized = self.board.serialize()
            # each player outputs a move given their view
            for moving_player_index, player in list(enumerate(self.players)):
                # get first valid action from player
                while True:
                    action = player.pop_top_move()
                    if action is None:
                        break

                    # check for validity of action
                    if self.board.is_action_valid(action, moving_player_index):
                        break

                if action is None:
                    continue

                # update game board with player's action
                self.update_game_state(action)

            # game logic to add troops to generals, cities, and land on specific ticks
            self.add_troops_to_board()
            self.update_player_stats()

            self.process_board_update(previous_board_serialized)
                

            await asyncio.sleep(0.5)
            self.turn += 1

        return self.board.terminal_status()


class LivePlayer():
    def __init__(self):
        self.move_queue = []
    def disseminate_game_start(self, board: Brd):
        pas
    def disseminate_game_voer(self, winner_player_id):
        pass
    def pop_top_move(self):
        return self.move_queue.pop(0) if self.move_queue else None
    def on_board_update(self, board: list[list[Tile]], board_diff: list[Tile]):
        # Subclasses must implement this
        raise NotImplementedError
class BotPlayer(LivePlayer):
    def __init__(self, agent: Agent):
        self.agent = agent
        super()

    def on_board_update(self, board: list[list[Tile]], board_diff: list[Tile]):
        self.move_queue = [self.agent.move(board)[0]]
class HumanPlayer(LivePlayer):
    # todo: separate player id from socket id?
    def __init__(self, player_id: str, username: str, live_game: LiveGame, player_index: int):
        self.player_id = player_id
        self.username = username
        self.game = live_game
        self.player_index = player_index
        self.land_count = 0
        self.army_count = 0
        self.game_state = {}
        super()
    def set_move_queue(self, move_queue: list[action.Action]):
        print('setting player move queue', move_queue)
        self.move_queue = move_queue

    def handle_disconnect(self):
        self.game.handle_disconnect_player(self.player_id)

    def disseminate_game_start(self, board: Board):
        emit('game_start', {'board_state': board.serialize(), 'player_index': self.player_index}, to=self.player_id)

    # Only send to the player the diff
    def on_board_update(self, board: list[list[Tile]], board_diff: list[Tile]):
        # Todo: don't do this for each player.
        emit('board_update', {'board_diff': [tile.serialize() for tile in board_diff], 'move_queue': [action.serialize() for action in self.move_queue]}, to=self.player_id)
        
    
    def disseminate_game_over(self, winner_player_id):
        emit('game_over', {'winning_player': winner_player_id})



class UserState(Enum):
    IN_LOBBY = 1
    IN_GAME = 2
    IN_QUEUE = 3



@dataclass
class BotSpec:
    model: str

@dataclass
class ConnectedUser:
    status: UserState
    socket_id: str
    username: str
    # If the user is actively in a game, the player is here.
    live_game_player: Optional[LivePlayer]