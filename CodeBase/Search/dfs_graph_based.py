from .planner import Planner
from collections import deque

class DFSPlanner_graphbased(Planner):


    def __init__(self, grid_map, motion_model="8n", visualizer=None):
        super().__init__(grid_map, motion_model, visualizer)
        self.expanded_count = 0
        self.expansion_map = {}

    def get_neighbors(self, gx, gy):

        if self.motion_model == "4n":
            moves = [(1,0), (-1,0), (0,1), (0,-1)]
        else:
            moves = [
                (1,0), (-1,0), (0,1), (0,-1),
                (1,1), (1,-1), (-1,1), (-1,-1)
            ]

        for dx, dy in moves:
            nx, ny = gx + dx, gy + dy

            if not self.grid_map.is_inside(nx, ny):
                continue

            if self.grid_map.is_obstacle(nx, ny):
                continue

            if self.grid_map.is_inflated(nx, ny):
                if self.visualizer:
                    self.visualizer.draw_inflated(nx, ny)
                continue

            yield (nx, ny)


    def plan(self, start, goal):

        vis = self.visualizer
        if vis:
            vis.draw_start_goal(start, goal)
            vis.update()


        stack = [(start, None)]  
        parent = {}          
        closed = set()

        while stack:

            v, p = stack.pop()

            # store parent mapping
            if p is not None:
                parent[v] = p

            #generation of expansion count for report only
            self.expanded_count += 1
            #generation of heatmap counter  for report only
            cx, cy = v
            self.expansion_map[(cx, cy)] = self.expansion_map.get((cx, cy), 0) + 1

            if vis:
                vx, vy = v
                vis.draw_explored(vx, vy)

                partial = self.build_partial_path(parent, start, v)
                for i in range(len(partial)-1):
                    x1, y1 = partial[i]
                    x2, y2 = partial[i+1]
                    vis.draw_path_segment(x1, y1, x2, y2)
                vis.update()


            if v == goal:
                return self.reconstruct_path(parent, start, goal)

            closed.add(v)

            cx, cy = v


            for child in self.get_neighbors(cx, cy):


                if child not in closed and child not in [s for s, _ in stack]:


                    stack.append((child, v))

                    if vis:
                        vis.draw_frontier(child[0], child[1])


        return None


    def reconstruct_path(self, parent, start, goal):
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def build_partial_path(self, parent, start, current):
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)
            if current == start:
                break
        path.reverse()
        return path