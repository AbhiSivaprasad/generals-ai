from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
 


def optimize_step(target_net: nn.Module, 
                  policy_net: nn.Module, 
                  optimizer: Optimizer, 
                  experiences: List[Tuple], 
                  gamma: float
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    experience_batch = tuple([list(t) for t in zip(*experiences)])
    s_t, a_t, r_t_1, s_t_1, d_t_1 = experience_batch
            
    # decouple (turn, grid) ObsType
    s_t = np.array([np.array(grid) for (_, grid) in s_t])
    s_t = torch.tensor(s_t, dtype=torch.float32).cuda().requires_grad_(True)
    
    batch_size = s_t.shape[0]
    
    s_t_1 = np.array([np.array(grid) for (_, grid) in s_t_1])
    assert s_t_1.shape[0] == batch_size
    with torch.inference_mode():
        s_t_1 = torch.tensor(s_t_1, dtype=torch.float32).cuda()
        # double DQN
        max_action = torch.argmax(policy_net(s_t_1), dim=-1)
        target_next_q_value: torch.Tensor = target_net(s_t_1)[range(batch_size), max_action]
    
    a_t = np.array([int(a) for a in a_t])
    assert len(a_t) == batch_size
    
    r_t_1 = torch.tensor(r_t_1, dtype=torch.float32).cuda().requires_grad_(False)
    assert r_t_1.shape[0] == batch_size
    
    d_t_1 = torch.tensor(d_t_1, dtype=torch.float32).cuda().requires_grad_(False)
    assert d_t_1.shape[0] == batch_size
    
    target_next_q_value = target_next_q_value.cuda().requires_grad_(False)
    
    prediction_outputs = policy_net(s_t)
    predicted_q_vals: torch.Tensor = prediction_outputs[range(batch_size), a_t]
    
    target_q_vals = r_t_1 + gamma * target_next_q_value * (1 - d_t_1)
    tde = target_q_vals - predicted_q_vals
    loss = tde.pow(2).mean()
    # loss = F.huber_loss(predicted_q_vals, target_q_vals)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    ret = {
        "predicted_q_vals": predicted_q_vals,
        "s_t": s_t,
        "s_t_1": s_t_1,
        "a_t": a_t,
        "r_t_1": r_t_1,
        "d_t_1": d_t_1,
        "target_next_q_value": target_next_q_value,
        "target_q_vals": target_q_vals
    }
    
    return loss, ret