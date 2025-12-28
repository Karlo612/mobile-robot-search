from .map_loader import MapLoader
from .grid_map import GridMap

class MobileRobot:

    def __init__(self, radius, start_grid, goal_grid):
        self.radius = radius
        self.sx,self.sy= start_grid
        self.gx,self.gy= goal_grid
        self.x = self.y = None
        self.grid_map=None
        self.world_map=None
        
    def attach_maps(self, grid_map, world_map):
        self.grid_map=grid_map
        self.world_map=world_map
        wx, wy = world_map.grid_to_world(self.sx, self.sy)
        self.x = wx
        self.y = wy
