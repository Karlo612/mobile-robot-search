import numpy as np
import random

from CodeBase.Environment.grid_map import GridMap
from CodeBase.Environment.world_map import WorldMap
from CodeBase.Environment.mobile_robot import MobileRobot
from CodeBase.Environment.inflator import ObstacleInflator
from CodeBase.Environment.obstacle import Obstacle


# -----------------------------------------------------------
#  Create random grid WITH BOUNDARY WALLS
# -----------------------------------------------------------
def generate_random_grid(size, fill_ratio=0.15):
    """
    Create a size×size grid where boundaries are blocked.
    1 = obstacle, 0 = free.
    """
    grid = np.random.choice([0, 1], size=(size, size),
                            p=[1 - fill_ratio, fill_ratio]).astype(int)

    # --- Force boundary to be obstacles ---
    grid[0, :] = 1                # top row
    grid[-1, :] = 1               # bottom row
    grid[:, 0] = 1                # left column
    grid[:, -1] = 1               # right column

    return grid


# -----------------------------------------------------------
#  Convert obstacles in grid to Obstacle objects
# -----------------------------------------------------------
def grid_to_obstacle_list(grid_array):
    obstacles = []
    h, w = grid_array.shape
    for y in range(h):
        for x in range(w):
            if grid_array[y, x] == 1:
                obstacles.append(Obstacle(x, y))
    return obstacles


# -----------------------------------------------------------
#  Pick start & goal from NON-obstacle & NON-inflated cells
# -----------------------------------------------------------
def pick_random_start_goal(grid_map):
    free_cells = []

    h, w = grid_map.height, grid_map.width

    for y in range(h):
        for x in range(w):
            if (not grid_map.is_obstacle(x, y) and
                not grid_map.is_inflated(x, y)):
                free_cells.append((x, y))

    if len(free_cells) < 2:
        raise RuntimeError("No valid free cells to choose start/goal.")

    start = random.choice(free_cells)
    goal = random.choice(free_cells)

    while goal == start:
        goal = random.choice(free_cells)

    return start, goal


# -----------------------------------------------------------
#  MAIN FUNCTION
# -----------------------------------------------------------
def generate_grid(size, robot_radius=1.0, fill_ratio=0.15):
    """
    Full automated benchmark grid generator:
        1. Create random grid with boundary walls.
        2. Create obstacle objects.
        3. Inflate based on robot radius.
        4. Pick valid start/goal.

    Returns:
        grid_array, start, goal
    """
    max_attempts = 50
    
    for attempt in range(max_attempts):
        try:
            # Step 1 — generate random grid
            grid_array = generate_random_grid(size, fill_ratio)

            # Step 2 — convert 1→Obstacle objects
            obstacles = grid_to_obstacle_list(grid_array)

            # Step 3 — Create GridMap + WorldMap
            grid_map = GridMap(grid_array, resolution=1.0)
            world_map = WorldMap(origin=(0, 0), resolution=1.0)

            # Step 4 — create robot with dummy start/goal
            robot = MobileRobot(radius=robot_radius, start_grid=(0, 0), goal_grid=(0, 0))
            robot.attach_maps(grid_map, world_map)

            # Step 5 — inflate obstacles using REAL inflator logic
            inflator = ObstacleInflator(robot_radius)
            inflator.inflate(grid_map, world_map, obstacles)

            # Step 6 — Select start and goal
            start, goal = pick_random_start_goal(grid_map)

            return grid_map, start, goal, obstacles
            
        except RuntimeError:
            # Regenerate grid if start/goal selection fails
            if attempt < max_attempts - 1:
                continue
            else:
                # All attempts failed
                raise RuntimeError(
                    f"Failed to generate valid grid after {max_attempts} attempts. "
                    f"Grid size: {size}x{size}, robot_radius: {robot_radius}. "
                    f"Try reducing fill_ratio or robot_radius, or increasing grid size."
                )