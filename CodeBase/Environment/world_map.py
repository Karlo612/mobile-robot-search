class WorldMap:
    """
    WorldMap handles the conversion from grid (gx,gy) to world cordinate (wx,wy):
    """

    def __init__(self, origin, resolution):
        self.origin_x = origin[0]
        self.origin_y = origin[1]
        self.resolution = resolution

    def grid_to_world(self, gx, gy):
        """
        Example:
            gx = 4, gy = 3
            resolution = 1.0
            origin = (0, 0)
            world position = (4.5, 3.5)
        """
        wx = self.origin_x + gx * self.resolution + self.resolution / 2
        wy = self.origin_y + gy * self.resolution + self.resolution / 2
        return wx, wy

 