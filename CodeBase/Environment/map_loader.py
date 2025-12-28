# map_loader.py

import pandas as pd
import numpy as np
from .obstacle import Obstacle

class MapLoader:
    """
    Loads a map from an XLSX file and converts it into:
      - a 2D grid array (0 = free, 1 = obstacle)
      - a list of obstacle coordinates in GRID space
    """
    def load(self, map_path):

        #Read Excel using pandas
        df = pd.read_excel(map_path, header=None)
        data = df.to_numpy()

        #Flip vertically this is because the excle file the top-left is 0,0 
        # while in the numpy array the bottom left is 0,0
        grid_array = data[::-1]

        #generate obstacles list from the grid array
        obstacles = []
        height, width = grid_array.shape
        for gy in range(height):
            for gx in range(width):
                if grid_array[gy][gx] == 1:
                    obstacles.append(Obstacle(gx, gy))

        # return grid_array and obstacle list
        return grid_array, obstacles