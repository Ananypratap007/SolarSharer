
import numpy as np


ACTION_MAP = {
    0: np.array([1, 0, 0, 0], dtype=np.float32),  # Buy from grid
    1: np.array([0, 1, 0, 0], dtype=np.float32),  # Sell to grid
    2: np.array([0, 0, 1, 0], dtype=np.float32),  # Buy from peers
    3: np.array([0, 0, 0, 1], dtype=np.float32),  # Sell to peers
}
