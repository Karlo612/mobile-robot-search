from .planner import Planner
from heapdict import heapdict
import math

class Node:
    def __init__(self, state, g=0.0, f=0.0, parent=None):
        self.state = state      # (x, y)
        self.g = g              # path cost
        self.f = f              # evaluation function g+h
        self.parent = parent    # pointer to parent Node

    def __repr__(self):
        return f"Node(state={self.state}, g={self.g}, f={self.f})"

class AStarPlanner_graphbased(Planner):

    def __init__(self, grid_map, motion_model="8n", visualizer=None):
        super().__init__(grid_map, motion_model, visualizer)
        self.res = grid_map.resolution

    def heuristic(self, a, b):
        x1, y1 = a
        x2, y2 = b
        dx, dy = abs(x1-x2), abs(y1-y2)

        if self.motion_model == "4n":
            #return Manhattan distance
            return (dx + dy) * self.res
        else:  
            # "8n" - returns Euclidean distance
            #return (math.sqrt(2)*min(dx,dy) + abs(dx-dy)) * self.res
             # "8n" - returns octile distance
            return (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))*self.res

    def get_neighbors(self, gx, gy):
        #generated neigbour coordinates based on the type of mothion '4n' or '8n'  

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

    def cost(self, x1, y1, x2, y2):
        # calculated cost from moving from current cell to the child cell
        if x1 != x2 and y1 != y2:
            return math.sqrt(2) * self.res
        return 1 * self.res

    def plan(self, start, goal):
        # this funtion runs the A-star algorithm

        # openset priority queue 
        OPEN = heapdict()
        # CLOSED set
        CLOSED = set()
        # used to stores Node object and track parent to construct path 
        NODES = {} 

        # Create start node
        start_node = Node(start, g=0.0, f=self.heuristic(start, goal))
        OPEN[start] = start_node.f        
        NODES[start] = start_node

        vis = self.visualizer
        if vis:
            vis.draw_start_goal(start, goal)
            vis.update()

        while OPEN:

            current_state, _ = OPEN.popitem()
            current = NODES[current_state]
            # Print f-cost for debug (only in visualization mode)
            if vis:
                print(
                    f"EXPAND {current_state}: "
                    f"g={current.g:.3f}, "
                    f"h={self.heuristic(current_state, goal):.3f}, "
                    f"f={current.f:.3f}"
                )
            if current_state == goal:
                return self.reconstruct_path(current)

            CLOSED.add(current_state)

            cx, cy = current_state

            if vis:
                vis.draw_explored(cx, cy)

                # draw partial path from start → current
                partial = self.build_partial_path(current)
                for i in range(len(partial) - 1):
                    x1, y1 = partial[i]
                    x2, y2 = partial[i+1]
                    vis.draw_path_segment(x1, y1, x2, y2)

                vis.update()

            for child in self.get_neighbors(cx, cy):

                v = child 
                
                if v in CLOSED:
                    continue

                new_g = current.g + self.cost(cx, cy, v[0], v[1])
                new_f = new_g + self.heuristic(v, goal)
            
                if v not in OPEN:
                    # child not in openset means new node

                    new_node = Node(v, g=new_g, f=new_f, parent=current)
                    NODES[v] = new_node
                    OPEN[v] = new_f

                    if vis:
                        vis.draw_frontier(v[0], v[1])

                else:
                    # otherwise an improved path found , so update the 
                    old_f = OPEN[v]
                    if new_f < old_f:
                        
                        #update the properting of v in the node list
                        node = NODES[v]
                        node.f = new_f
                        node.g = new_g
                        node.parent = current

                        # update the f vaule of the v - node in the priotiy queue with the new f vaule
                        OPEN[v] = new_f  

                        if vis:
                            vis.draw_frontier(child[0], child[1])
        return None

    def reconstruct_path(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))

    def build_partial_path(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))