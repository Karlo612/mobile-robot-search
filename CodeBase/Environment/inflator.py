# inflator.py
import math
import numpy
from .obstacle import Obstacle

class ObstacleInflator:
    """
    in the grid map where cells within robot radius of an obstacle makes it blocked
    """
    def __init__(self, robot_radius):

        self.grid_map = None
        self.world_map = None
        self.robot_radius = robot_radius
        self.BLOCKED = 1
        self.obstacles = None

    def distance(self,obs_x,obs_y,wx,wy):
        return math.sqrt((obs_x-wx)**2+(obs_y-wy)**2)

    def inflate(self, grid_map, world_map, obstacles):

        self.grid_map = grid_map
        self.world_map = world_map
        self.obstacles = obstacles

        for gy in range(grid_map.height):
            for gx in range(grid_map.width):

                if self.grid_map.is_obstacle(gx, gy):
                    continue

                # center of this cell
                wx, wy = self.world_map.grid_to_world(gx, gy)

                for obstacle in self.obstacles:
                    obs_wx, obs_wy = self.world_map.grid_to_world(obstacle.gx, obstacle.gy)
                    # distance between cell center and obstacle center
                    ds = self.distance(obs_wx, obs_wy, wx, wy)
                    if ds < self.robot_radius:
                        self.grid_map.mark_inflated(gx, gy)
                        break

        #count = 0
        #for y in range(grid_map.height):
            #for x in range(grid_map.width):
                #if grid_map.is_inflated(x, y):
                    #count += 1

        #print("TOTAL INFLATED CELLS:", count)                    
                       