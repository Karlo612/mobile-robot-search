# visualizer.py

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np

class Visualizer:

    def __init__(self, grid_map, cell_size=1.0):
        self.grid_map = grid_map
        self.cell_size = cell_size

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect("equal")

    def setup(self):
        h = self.grid_map.height
        w = self.grid_map.width
        s = self.cell_size

        # world coordinate boundaries
        self.ax.set_xlim(0, w * s)
        self.ax.set_ylim(0, h * s)

        # ticks every cell
        self.ax.set_xticks(np.arange(0, w * s + s, s))
        self.ax.set_yticks(np.arange(0, h * s + s, s))

        self.ax.grid(True)
        self.draw_grid()

         # Add legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='gray', markersize=10, label='Obstacle'),
            Line2D([0], [0], marker='s', color='red', markersize=10, label='Inflated'),
            Line2D([0], [0], marker='s', color='cyan', markersize=10, label='Free'),
            Line2D([0], [0], marker='s', color='yellow', markersize=10, label='Explored'),
            Line2D([0], [0], marker='s', color='blue', markersize=10, label='Frontier'),
            Line2D([0], [0], marker='s', color='black', markersize=10, label='Blocked'),
            Line2D([0], [0], color='m', lw=2, label='Partial Path'),
            Line2D([0], [0], color='k', lw=2, label='Final Path'),
        ]
        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1.03, 1),
            fontsize=8
        )

    def _to_center(self, gx, gy):
        s = self.cell_size
        return gx * s + s/2, gy * s + s/2

    def draw_grid(self):
        """Draw obstacles and free space squares."""
        for gy in range(self.grid_map.height):
            for gx in range(self.grid_map.width):
                if self.grid_map.is_obstacle(gx, gy) and not self.grid_map.is_inflated(gx, gy):
                    color = "gray"
                else:
                    color = "cyan"  

                self._draw_cell(gx, gy, color)


    def draw_start_goal(self, start, goal):
        sx, sy = start
        gx, gy = goal

        cx, cy = self._to_center(sx, sy)
        self.ax.scatter(cx, cy, s=300, c="red", marker="s")

        cx, cy = self._to_center(gx, gy)
        self.ax.scatter(cx, cy, s=300, c="green", marker="s")

    def draw_explored(self, gx, gy):
        cx, cy = self._to_center(gx, gy)
        self.ax.scatter(cx, cy, s=120, c="yellow", marker="s")

    def draw_frontier(self, gx, gy):
        cx, cy = self._to_center(gx, gy)
        self.ax.scatter(cx, cy, s=120, c="blue", marker="s")

    def draw_blocked(self, gx, gy):
        cx, cy = self._to_center(gx, gy)
        self.ax.scatter(cx, cy, s=120, c="black", marker="s")

    def draw_inflated(self, gx, gy):
        cx, cy = self._to_center(gx, gy)
        self.ax.scatter(cx, cy, s=130, c="red", marker="s")

    def draw_partial_path(self, path):
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            self.draw_path_segment(x1, y1, x2, y2)
     
    def draw_path_segment(self, x1, y1, x2, y2):
        x1, y1 = self._to_center(x1, y1)
        x2, y2 = self._to_center(x2, y2)
        self.ax.plot([x1, x2], [y1, y2], "-m", linewidth=3)

    def draw_final_path(self, path):
        for i in range(len(path) - 1):
            x1, y1 = self._to_center(*path[i])
            x2, y2 = self._to_center(*path[i+1])
            self.ax.plot([x1, x2], [y1, y2], "-k", linewidth=3)

    def _draw_cell(self, gx, gy, color):
        s = self.cell_size
        """Draw a grid square (background)."""
        self.ax.add_patch(Rectangle(
            (gx*s, gy*s),   # bottom-left
            s,        # width
            s,        # height
            facecolor=color,
            edgecolor="none",
            alpha=0.9
        ))

    def update(self, pause=1):
        plt.pause(pause)

    def show(self):
        plt.show()