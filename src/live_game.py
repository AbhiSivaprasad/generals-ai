import asyncio
from dataclasses import dataclass
from enum import Enum
from socket import SocketIO
from typing import Optional, Union
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
from src.environment.gamestate import GameState
from src.environment.tile import Tile

TICK_TIME = 0.5



# This class represents a live game that takes place on the server.
#
# Key differences from GameMaster:
# 1. Asynchronous play function with "TICK_TIME" seconds between turns.
# 2. Players update their move queue instead of direct agent queries.
#
# Move queue updates:
# - Live players: via websocket queries to "set_move_queue" endpoint.
# - Bots: via on_board_update function, triggering the agent's move() function.
class LiveGame(GameMaster):
    def __init__(self, player_inits: list[Union["ConnectedUser", "BotSpec"]], board_init: Optional[Board] = None):
        if board_init is None:
            board_init = generate_board_state(15, 15, mountain_probability=0.2, city_probability=0.03)
        self.live_players = []
        self.live_players = [self.create_live_player(init, idx, self) for idx, init in enumerate(player_inits)]
        self.game_id = uuid.uuid4()
        super().__init__(board_init, self.live_players)
    
    def create_live_player(self, player_init, player_index, game):
        if isinstance(player_init, ConnectedUser):
            player_init.live_game_player = HumanPlayer(player_init.socket_id, player_init.username, game, player_index)
            player_init.status = UserState.IN_GAME
            return player_init.live_game_player
        else:
            return BotPlayer(player_index, player_init.agent)

        

    def disseminate_board_state(self):
        # TODO: make this simultaneous
        for player in self.live_players:
            player.disseminate_board_state(self.state.board)

    def process_board_update(self, previous_board_serialized):
        diff = self.get_board_diff(previous_board_serialized)
        # We kick off the update tasks here so that we don't block the game thread.
        for player in self.live_players:
            asyncio.create_task(player.on_board_update(self.state, diff))

    def handle_disconnect_player(self, player_id):
        for player in self.liveplayers:
            if player.player_id != player_id:
                emit('game_over', {"reason": "player_disconnected", "player_id": player_id}, to=player.player_id)
        self.is_playing = False

    async def start_game(self):
        winning_player = await self.play()
        emit('game_over', {"reason": "player_won", "player_id": self.live_players[winning_player].username})

    # We don't want to send the entire board state over the network every time,
    # so we compute the diff between the previous board state and the current board state
    # and send the diff over the network.
    # TODO: Coalesce this logic with the more efficient logic used to log update
    # tiles in GameMaster.
    def get_board_diff(self, previous_board_serialized: dict):
        tiles_changed = []
        for y in range(self.state.board.num_rows):
            for x in range(self.state.board.num_cols):
                if self.state.board.grid[y][x].serialize() != previous_board_serialized[y][x]:
                    tiles_changed.append(self.state.board.grid[y][x])
        return tiles_changed


    async def play(self):
        """
        conduct game between players on given board
        players can be humans or bots, they both ascribe to the same interface
        :return: index of winning player or -1 if max turns reached
        """
        self.is_playing = True
        player_names = [player.name() for player in self.live_players]
        for player in self.live_players:
            player.disseminate_game_start(self.state, player_names)
        await asyncio.sleep(TICK_TIME)
        while self.state.board.terminal_status() == -1 and (self.max_turns is None or self.state.turn < self.max_turns) and self.is_playing:
            print('self turn is', self.state.turn)
            # serialize the board so that we can send the diff
            previous_board_serialized = self.state.board.serialize()
            # each player outputs a move given their view
            for moving_player_index, player in list(enumerate(self.live_players)):
                # get first valid action from player
                while True:
                    action = player.pop_top_move()
                    if action is None:
                        break

                    # check for validity of action
                    if self.state.board.is_action_valid(action, moving_player_index):
                        break

                if action is None:
                    continue

                # update game board with player's action
                self.update_game_state(action)

            # game logic to add troops to generals, cities, and land on specific ticks
            self.add_troops_to_board()
            self.recalculate_player_scores()

            self.process_board_update(previous_board_serialized)
                

            await asyncio.sleep(TICK_TIME)
            self.state.turn += 1

        if self.max_turns is not None and self.state.turn >= self.max_turns:
            return -1
        return self.state.board.terminal_status()


class LivePlayer():
    def __init__(self, player_index: int):
        self.move_queue = []
        self.player_index = player_index

    def name(self):
        raise NotImplementedError
    def disseminate_game_start(self, state: GameState, player_names: list[str]):
        pass
    def disseminate_game_over(self, winner_player_id):
        pass
    def pop_top_move(self):
        return self.move_queue.pop(0) if self.move_queue else None
    async def on_board_update(self, game_state: GameState, board_diff: list[Tile]):
        # Subclasses must implement this
        raise NotImplementedError

# A human player is pretty simple, we just send  game updates directly to the client.
class HumanPlayer(LivePlayer):
    # todo: separate player id from socket id?
    def __init__(self, player_id: str, username: str, live_game: LiveGame, player_index: int):
        self.player_id = player_id
        self.username = username
        self.game = live_game
        self.land_count = 0
        # Moves consumer by the server.
        self.server_consumed_moves = 0
        self.army_count = 0
        self.game_state = {}
        super().__init__(player_index)
    def set_move_queue(self, move_queue: list[action.Action]):
        print('SETTING MOVE QUEUE')
        self.move_queue = move_queue

    def name(self):
        return self.username

    def handle_disconnect(self):
        self.game.handle_disconnect_player(self.player_id)

    def disseminate_game_start(self, state: GameState, player_names: list[str]):
        emit('game_start', {'board_state': state.board.serialize(), 'player_names': player_names, 'player_index': self.player_index}, to=self.player_id)

    def pop_top_move(self):
        self.server_consumed_moves += 1
        return super().pop_top_move()

    # Only send to the player the diff
    async def on_board_update(self, game_state: GameState, board_diff: list[Tile]):
        # Todo: don't do this for each player.
        emit('board_update', {'board_diff': [tile.serialize() for tile in board_diff], 'army_counts': game_state.army_counts, 'land_counts': game_state.land_counts, 'server_consumed_moves': self.server_consumed_moves, 'move_queue': [action.serialize() for action in self.move_queue]}, to=self.player_id)
        self.server_consumed_moves = 0
        
    
    def disseminate_game_over(self, winner_player_id):
        emit('game_over', {'winning_player': winner_player_id})

# Small wrapper around a bot to give it the same interface as a human.
# Instead of sending board updates, we use them to trigger an update to the
# bot's move queue.
class BotPlayer(LivePlayer):
    def __init__(self, player_index: int, agent: Agent):
        self.agent = agent
        super().__init__(player_index)
    
    def name(self):
        return "Some Bot"

    async def on_board_update(self, game_state: GameState, board_diff: list[Tile]):
        next_move = self.agent.move(game_state)
        if next_move is not None:
            self.move_queue = [next_move]

class UserState(Enum):
    IN_LOBBY = 1
    IN_GAME = 2
    IN_QUEUE = 3



# Added a layer of indirection here in case there are additional settings for a
# bot
# beyond which agent it uses to select its next move.
# TODO: Maybe remove this?
@dataclass
class BotSpec:
    agent: Agent

@dataclass
class ConnectedUser:
    status: UserState
    socket_id: str
    username: str
    # If the user is actively in a game, the player is here.
    live_game_player: Optional[LivePlayer]