# base_planner.py

class Planner:
    """
    Abstract base class for all planners.
    """
    def __init__(self, grid_map, motion_model="8n",visualizer=None):
        self.grid_map = grid_map
        self.visualizer = visualizer
        self.motion_model=motion_model

    def plan(self, start, goal):
        raise NotImplementedError("plan() must be implemented by subclasses")