from typing import List
from src.agents.agent import Agent
from src.agents.random_agent import RandomAgent
from src.environment.action import Action
from src.environment.board_generator import generate_board_state
from src.environment.game_master import GameMaster


class GeneralsEnvironment:
    game_master: GameMaster

    def __init__(self, players: List[Agent], max_turns=1000) -> None:
        self.players = players
        self.max_turns = max_turns

    def step(self, actions: List[Action]):
        # execute one tick of the game
        self.game_master.step(actions)

        # return the new state, reward, terminal status, and info dict
        observation = self.game_master.board
        rewards = [self.game_master.board.get_player_score(i) for i in range(2)]
        done = self.board.terminal_status() != -1
        info = {}
        return (observation, rewards, done, info)

    def reset(self, seed=None):
        self.board = self.game_master.reset(seed)

        # generate new board
        board = generate_board_state(3, 3, mountain_probability=0, city_probability=0)
        self.game_master = GameMaster(
            board,
            players=self.players,
            logger=None,
            max_turns=self.max_turns,
        )
