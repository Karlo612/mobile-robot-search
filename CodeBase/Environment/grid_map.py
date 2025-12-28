# grid_map.py
import numpy as np

class GridMap:
    """
    Stores free/obstacles as 0/1 in GRID coordinates
    grid[row][column]  → grid[gy][gx]
    """

    def __init__(self, grid_array, resolution):

        self.height, self.width = grid_array.shape
        self.resolution = resolution
        self.grid = grid_array

        self.inflated_grid = [
            [False for _ in range(self.width)]
            for _ in range(self.height)
        ]

    def set_cell(self, gx, gy, value):
        self.grid[gy][gx] = value

    def mark_inflated(self, gx, gy):
        self.inflated_grid[gy][gx] = True

    def get_cell(self, gx, gy):
        return self.grid[gy][gx]

    def is_inside(self, gx, gy):
        return 0 <= gx < self.width and 0 <= gy < self.height
    
    def is_free(self,gx, gy):
        return True if self.grid[gy][gx]==0 else False

    def is_obstacle(self,gx, gy):
        return True if self.grid[gy][gx]==1 else False
    
    def is_inflated(self, gx, gy):
        return self.inflated_grid[gy][gx]
    
    
    