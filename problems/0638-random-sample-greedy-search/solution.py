import numpy as np

def random_sample_greedy_search(q_table: np.ndarray, states: list, sample_size: int, epsilon: float, seed: int) -> dict:

    rng = np.random.RandomState(seed)

    num_actions = q_table.shape[1]
    k = min(sample_size, num_actions)

    actions = []
    q_values = []
    matches = 0

    for state in states:

        # True greedy action (highest Q-value across all actions)
        true_greedy = np.argmax(q_table[state])

        # Decide exploration or exploitation
        if rng.random() < epsilon:

            # Exploration: select any random action
            action = rng.randint(0, num_actions)

        else:

            # Exploitation: randomly sample k actions
            sampled = rng.choice(num_actions, size=k, replace=False)

            # Sort indices to break ties by smallest index
            sampled = np.sort(sampled)

            # Get Q-values of sampled actions
            sampled_q = q_table[state][sampled]

            # Select action with maximum Q-value
            best_index = np.argmax(sampled_q)

            action = sampled[best_index]

        # Store selected action and its Q-value
        actions.append(int(action))
        q_values.append(round(float(q_table[state][action]), 4))

        # Check whether selected action matches true greedy action
        if action == true_greedy:
            matches += 1

    # Calculate greedy match rate
    match_rate = matches / len(states) if len(states) > 0 else 0.0

    return {
        'actions': actions,
        'q_values': q_values,
        'greedy_match_rate': round(match_rate, 4)
    }