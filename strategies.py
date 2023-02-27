import numpy as np


# Run Right at 'em Strategy
def direct_approach(self_position: np.array,
                    target_position: np.array):
    # Basic behavior: run towards target
    to_target = target_position - self_position

    # Return Move Direction
    return to_target / np.linalg.norm(to_target)


# Run Directly Away Strategy
def direct_flee(self_position: np.array,
                chaser_position: np.array):
    # Basic behavior: run away from chaser
    to_chaser = chaser_position - self_position

    # Return Move Direction
    return -to_chaser / np.linalg.norm(to_chaser)
