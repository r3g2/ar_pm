import numpy as np


class Observation:
    def __init__(self, pos, value):
        self.pos = pos
        self.value = value
    
    def get_1dpos(self,grid_shape):
        return np.ravel_multi_index(np.array([[self.pos[0]],[self.pos[1]]]),grid_shape)

class StateGrid:
    def __init__(self,grid_matrix):
        self.grid = grid_matrix
        self.h = np.argmax(self.grid)

    @classmethod
    def initialize_random_grid(cls,size):
        index_hidden = np.unravel_index(np.argmax(np.random.rand(*size)),size)
        grid = np.zeros(size)
        grid[index_hidden] = 1
        return cls(grid)