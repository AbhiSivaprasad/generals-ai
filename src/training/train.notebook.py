# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
num_episodes = 50000
log_interval = 200
checkpoint_interval = 200
global_step = 0
for i_episode in range(num_episodes):
    logger = Logger()
    # Initialize the environment and get its state
    state, info = env.reset(logger=logger)
    convert_agent_dict_to_tensor(state, device=device)
    num_agents = len(env.unwrapped.agents)
    for t in count():
        actions_with_info = {
            agent_index: agent.move(state[agent_index], env)
            for agent_index, agent in enumerate(env.unwrapped.agents)
        }
        actions = {
            agent_index: action
            for agent_index, (action, _) in actions_with_info.items()
        }
        observation, rewards, terminated, truncated, info = env.step(actions)
        convert_agent_dict_to_tensor(rewards, device=device)
        convert_agent_dict_to_tensor(actions, dtype=torch.long, device=device)
        truncated = list(truncated.values())[0]
        terminated = list(terminated.values())[0]
        done = terminated or truncated

        # next state is none if the game is terminated
        if terminated:
            next_state = {agent_name: None for agent_name in state.keys()}
        else:
            convert_agent_dict_to_tensor(observation, device=device)
            next_state = observation

        # Store the transitions in memory
        for agent_name in state.keys():
            memory.push(
                state[agent_name],
                actions[agent_name],
                next_state[agent_name],
                rewards[agent_name],
            )

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        metrics = optimize_model()
        if metrics is not None:
            minibatch_loss, minibatch_reward_magnitude, minibatch_qvalue_magnitude = (
                metrics
            )
            wandb.log(
                {
                    "loss": minibatch_loss,
                    "reward_magnitude": minibatch_reward_magnitude,
                    "qvalue_magnitude": minibatch_qvalue_magnitude,
                },
                step=global_step,
            )

        # get legal move rate of agent 0
        action, action_info = actions_with_info[0]
        is_action_legal = env.unwrapped.game_master.board.is_action_valid(
            Action.from_index(
                action_info["best_action"], n_columns=env.unwrapped.board_x_size
            ),
            player_index=0,
        )

        # log other metrics
        wandb.log(
            {
                "legal_move": int(is_action_legal),
                "epsilon": env.unwrapped.agents[0].epsilon,
            },
            step=global_step,
        )

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        global_step += 1

        if done:
            wandb.log({"duration": t, "episode": i_episode}, step=global_step)
            if i_episode % checkpoint_interval == 0:
                policy_net.save_checkpoint(CHECKPOINT_DIR, i_episode)
            if i_episode % log_interval == 0:
                logger.write(LOG_DIR / f"{i_episode}.json")
            break
