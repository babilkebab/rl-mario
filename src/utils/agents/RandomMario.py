import numpy as np

class RandomMario:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.curr_step = 0

    def act(self):
        """
        Executes a random action.
        """
        # EXPLORE
        action_idx = np.random.randint(self.action_dim)

        # increment step
        self.curr_step += 1
        return action_idx