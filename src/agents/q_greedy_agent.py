from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.agents.agent import Agent
from src.agents.utils.observation_receiver import ObservationReceivingInterface
from src.environment.action import Action
from src.environment import ActType, ObsType
from src.environment.gamestate import GameState
from src.utils.replay_buffer import Experience, ReplayBuffer

class QGreedyAgent(Agent):
    q_function: Callable[[ObsType], List[Tuple[ActType, int]]] = None
    
    def __init__(self, player_index: int, q_function: Callable[[ObsType], List[Tuple[ActType, int]]], *args, **kwargs):
        super().__init__(player_index, *args, **kwargs)
        self.q_function = q_function

    def move(self, state: GameState) -> Optional[Action]:
        observation = state.to_observation(self.player_index)
        q_values = self.q_function(observation)
        best_action, q_value = max(q_values, key=lambda x: x[1])
        r, c = state.board.num_rows, state.board.num_cols
        best_action = Action.from_space_sample(best_action, r, c)
        # print("[INFO] QGreedyAgent.move() BEST_ACTION: ", best_action)
        return best_action
    
    def reset(self, *args, **kwargs) -> None:
        pass
    
class QGreedyLearningAgent(QGreedyAgent, ObservationReceivingInterface):
    def __init__(
        self, 
        player_index: int, 
        q_function: Callable[[ObsType], List[Tuple[ActType, int]]], 
        obs_handler: Callable[[Tuple[ObsType, float, bool, bool, Dict]], None],
        *args, 
        **kwargs
    ):
        super().__init__(player_index, q_function, *args, **kwargs)
        assert obs_handler is not None, "An ObservationReceivingInterface object needs an observation handler!"
        self.obs_handler = obs_handler
    
    def observe(self, experience: Experience) -> None:
        self.obs_handler(experience)

class DQNAgent(QGreedyAgent):
    model: Optional[torch.nn.Module]
    device: Optional[torch.device]
    
    def __init__(
        self, 
        player_index: int, 
        model: torch.nn.Module,
        device: torch.device,
        *args, 
        **kwargs
    ):
        self.model = model
        self.device = device
        super().__init__(player_index, self.get_q_function(), *args, **kwargs)
    
    def get_q_function(self):
        device = self.device if self.device is not None else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = self.model
        if model is None:
            return None
        def qf(observation: ObsType) -> List[Tuple[ActType, int]]:
            turn, obs = observation
            rows, cols = obs.shape[0], obs.shape[1]
            obs = torch.tensor(obs, dtype=torch.float32).to(device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = model(obs).detach().cpu().numpy().flatten()
            return [(i, q_values[i]) for i in range(len(q_values))]
        return qf
    
    def update_model(self, model: torch.nn.Module, device: torch.device = None) -> None:
        self.model = model
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        self.q_function = self.get_q_function()
        
    

class DQNLearningAgent(DQNAgent, ObservationReceivingInterface):
    replay_buffer: ReplayBuffer
    
    def __init__(
        self, 
        player_index: int, 
        model: torch.nn.Module,
        replay_buffer: ReplayBuffer,
        device: torch.device,
        *args, 
        **kwargs
    ):
        super().__init__(player_index=player_index, model=model, device=device, *args, **kwargs)
        self.replay_buffer = replay_buffer
    
    def observe(self, experience: Tuple[ObsType, float, bool, bool, Dict]) -> None:
        self.replay_buffer.add([experience])