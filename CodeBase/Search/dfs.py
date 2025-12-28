# Wrapper for DFS imports - provides compatibility with navigation_system imports
from .dfs_graph_based import DFSPlanner_graphbased
from .dfs_tree_based import DFSPlanner_treebased

__all__ = ['DFSPlanner_graphbased', 'DFSPlanner_treebased']
