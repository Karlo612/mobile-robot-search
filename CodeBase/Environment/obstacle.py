# obstacle.py

class Obstacle:
    """
    Represents a single obstacle in GRID and WORLD coordinates.
    """

    def __init__(self, gx, gy, wx=None, wy=None):

        self.gx = gx
        self.gy = gy
        self.wx = wx
        self.wy = wy

    def __repr__(self):
        return f"Obstacle(gx={self.gx}, gy={self.gy})"