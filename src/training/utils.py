from typing import Dict
import torch


def convert_agent_dict_to_tensor(
    agent_dict: Dict, device: torch.device, dtype=torch.float32
):
    for agent_name in agent_dict.keys():
        agent_dict[agent_name] = torch.tensor(
            agent_dict[agent_name], dtype=dtype, device=device
        )
