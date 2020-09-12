import numpy as np
from mlagents_envs.environment import UnityEnvironment

def step_env(env: UnityEnvironment, actions: np.array):
    agents_transitions = {}
    for brain in env.behavior_specs:
        actions = np.resize(actions,
                            (len(env.get_steps(brain)[0]), len(env.behavior_specs[brain].discrete_action_branches)))
        env.set_actions(brain, actions)
        env.step()
        decision_steps, terminal_steps = env.get_steps(brain)

        for agent_id_decisions in decision_steps:
            agents_transitions[agent_id_decisions] = [decision_steps[agent_id_decisions].obs,
                                                      decision_steps[agent_id_decisions].reward, False]

        for agent_id_terminated in terminal_steps:
            agents_transitions[agent_id_terminated] = [terminal_steps[agent_id_terminated].obs,
                                                       terminal_steps[agent_id_terminated].reward, True]

    return agents_transitions