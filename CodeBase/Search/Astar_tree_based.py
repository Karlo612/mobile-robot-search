from .planner import Planner
import heapq
import math
from collections import defaultdict

class AStarPlanner_treebased(Planner):

    def __init__(self, grid_map, motion_model="8n", visualizer=None):
        super().__init__(grid_map, motion_model, visualizer)
        self.res = grid_map.resolution
        self.counter = 0   # unique visit-id counter

    # ----------------------------------------
    # Heuristic
    # ----------------------------------------
    def heuristic(self, a, b):
        x1, y1 = a
        x2, y2 = b
        dx, dy = abs(x1 - x2), abs(y1 - y2)

        if self.motion_model == "4n":
            return (dx + dy) * self.res
        else:
            return (max(dx, dy) + (math.sqrt(2)-1) * min(dx, dy)) * self.res

    # ----------------------------------------
    # Neighbors
    # ----------------------------------------
    def get_neighbors(self, gx, gy):
        if self.motion_model == "4n":
            moves = [(1,0),(-1,0),(0,1),(0,-1)]
        else:
            moves = [
                (1,0),(-1,0),(0,1),(0,-1),
                (1,1),(1,-1),(-1,1),(-1,-1)
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

    # ----------------------------------------
    # Cost model
    # ----------------------------------------
    def cost(self, x1, y1, x2, y2):
        if x1 != x2 and y1 != y2:
            return math.sqrt(2) * self.res
        return 1 * self.res

    # ----------------------------------------
    # MAIN TREE-BASED A*
    # ----------------------------------------
    def plan(self, start, goal):

        OPEN = []  # heap of (f, g, visit_id, (x,y))

        # 3D parent[gx][visit_id] = (parent_state, parent_vid)
        parent = defaultdict(dict)
        g_cost = defaultdict(dict)

        # push root
        h0 = self.heuristic(start, goal)
        g_cost[start][0] = 0.0
        heapq.heappush(OPEN, (h0, 0.0, 0, start))
        self.counter = 1  # next visit ID

        vis = self.visualizer
        if vis:
            vis.draw_start_goal(start, goal)
            vis.update()

        while OPEN:

            f, g, vid, current = heapq.heappop(OPEN)
            cx, cy = current

            # Debug print (only in visualization mode)
            if vis:
                print(
                    f"EXPAND {current}: g={g:.3f}, "
                    f"h={self.heuristic(current, goal):.3f}, f={f:.3f}"
                )

            # Goal reached → reconstruct path using correct vid chain
            if current == goal:
                return self.reconstruct_path_3d(parent, start, goal, vid)

            # Visualization
            if vis:
                vis.draw_explored(cx, cy)
                partial = self.build_partial_path_3d(parent, start, current, vid)
                for i in range(len(partial)-1):
                    x1, y1 = partial[i]
                    x2, y2 = partial[i+1]
                    vis.draw_path_segment(x1, y1, x2, y2)
                vis.update()

            # TREE SEARCH (no CLOSED): expand ALL children
            for child in self.get_neighbors(cx, cy):

                nx, ny = child
                ng = g + self.cost(cx, cy, nx, ny)
                nf = ng + self.heuristic(child, goal)

                child_vid = self.counter
                self.counter += 1

                # Store explicit parent link: (parent_state, parent_vid)
                parent[child][child_vid] = (current, vid)
                g_cost[child][child_vid] = ng

                heapq.heappush(OPEN, (nf, ng, child_vid, child))

                if vis:
                    vis.draw_frontier(nx, ny)

        return None

    # ----------------------------------------
    # SAFE 3D reconstruction
    # ----------------------------------------
    def reconstruct_path_3d(self, parent, start, goal, vid):
        """
        Reconstruct full final path from goal back to start.
        Works with (state, vid) pointers.
        """
        path = []
        cur_state = goal
        cur_vid = vid

        visited = set()

        while True:
            path.append(cur_state)

            if cur_state == start:
                break

            # detect loops
            if (cur_state, cur_vid) in visited:
                print("[ERROR] Loop detected in 3D final path!")
                break

            visited.add((cur_state, cur_vid))

            # validate parent entry
            if cur_state not in parent or cur_vid not in parent[cur_state]:
                print("[ERROR] Missing parent link in 3D final path. Returning partial path.")
                break

            # follow parent
            prev_state, prev_vid = parent[cur_state][cur_vid]

            cur_state = prev_state
            cur_vid = prev_vid

        return list(reversed(path))
    # ----------------------------------------
    # Build partial path for visualization
    # ----------------------------------------
    def build_partial_path_3d(self, parent, start, state, vid):
        """
        Returns partial path for visualization, safe version.
        Output must be list of (x,y) states.
        """
        path = []
        cur_state = state
        cur_vid = vid
        
        visited = set()

        while True:
            path.append(cur_state)

            if cur_state == start:
                break

            # detect infinite loops
            if (cur_state, cur_vid) in visited:
                print("[WARN] Loop in partial 3D path — breaking early.")
                break

            visited.add((cur_state, cur_vid))

            if cur_state not in parent or cur_vid not in parent[cur_state]:
                print("[WARN] Missing parent in partial 3D path — breaking early.")
                break

            prev_state, prev_vid = parent[cur_state][cur_vid]

            cur_state = prev_state
            cur_vid = prev_vid

        return list(reversed(path))