"""
A* Pathfinding Module for UrbanAirspaceSim
Handles individual agent pathfinding with dynamic obstacle avoidance
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Import from grid system
from grid_system import Position, Agent, AgentType, GridSystem

@dataclass
class PathNode:
    """Node for A* pathfinding with proper ordering"""
    position: Position
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    f_cost: float = field(init=False)  # Total cost
    parent: Optional['PathNode'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        # Tie-breaking: prefer higher g_cost (closer to goal)
        return self.g_cost > other.g_cost

@dataclass
class PathfindingContext:
    """Context for pathfinding operations"""
    agent_id: int
    start: Tuple[int, int]  # (x, y)
    goal: Tuple[int, int]   # (x, y)
    start_time: int = 0
    max_time: int = 100
    
    # Dynamic obstacles to avoid
    emergency_paths: Dict[int, List[Position]] = field(default_factory=dict)
    higher_priority_paths: Dict[int, List[Position]] = field(default_factory=dict)
    
    # Cost parameters
    move_cost: float = 1.0
    wait_cost: float = 1.0
    time_penalty: float = 0.01  # Small penalty for longer paths

class AStarPathfinder:
    """
    A* pathfinding implementation for UrbanAirspaceSim
    Supports dynamic obstacle avoidance and priority-based planning
    """
    
    def __init__(self, grid_system: GridSystem):
        """
        Initialize pathfinder with grid system
        
        Args:
            grid_system: The grid system instance
        """
        self.grid = grid_system
        self.debug_mode = False
        
        # Cache for repeated calculations
        self._heuristic_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two 2D positions"""
        cache_key = (pos1, pos2)
        if cache_key not in self._heuristic_cache:
            self._heuristic_cache[cache_key] = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        return self._heuristic_cache[cache_key]
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two 2D positions"""
        cache_key = (pos1, pos2)
        if cache_key not in self._heuristic_cache:
            self._heuristic_cache[cache_key] = math.sqrt(
                (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2
            )
        return self._heuristic_cache[cache_key]
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Heuristic function for A* (admissible)
        Using Manhattan distance as we have 4-connectivity movement
        """
        return self.manhattan_distance(pos, goal)
    
    def is_position_blocked(self, pos: Position, context: PathfindingContext) -> bool:
        """
        Check if a position is blocked by obstacles or other agents
        
        Args:
            pos: Position to check
            context: Pathfinding context with obstacle information
        """
        # Check grid boundaries and static obstacles
        if not self.grid.is_cell_free(pos, ignore_agent_id=context.agent_id):
            return True
        
        # Check emergency agent paths (always have highest priority)
        for emergency_agent_id, emergency_path in context.emergency_paths.items():
            if emergency_agent_id != context.agent_id:
                for emergency_pos in emergency_path:
                    if (emergency_pos.x == pos.x and 
                        emergency_pos.y == pos.y and 
                        emergency_pos.t == pos.t):
                        return True
        
        # Check higher priority agent paths (auction winners)
        for priority_agent_id, priority_path in context.higher_priority_paths.items():
            if priority_agent_id != context.agent_id:
                for priority_pos in priority_path:
                    if (priority_pos.x == pos.x and 
                        priority_pos.y == pos.y and 
                        priority_pos.t == pos.t):
                        return True
        
        return False
    
    def get_valid_neighbors(self, pos: Position, context: PathfindingContext) -> List[Position]:
        """
        Get valid neighboring positions considering obstacles
        
        Args:
            pos: Current position
            context: Pathfinding context
            
        Returns:
            List of valid neighboring positions
        """
        neighbors = []
        
        # Get all possible neighbors from grid system
        potential_neighbors = self.grid.get_neighbors(pos)
        
        for neighbor in potential_neighbors:
            # Skip if position is blocked
            if self.is_position_blocked(neighbor, context):
                continue
            
            # Skip if exceeds maximum time
            if neighbor.t >= context.max_time:
                continue
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def reconstruct_path(self, goal_node: PathNode) -> List[Position]:
        """
        Reconstruct path from goal node to start node
        
        Args:
            goal_node: The goal node with parent chain
            
        Returns:
            List of positions from start to goal
        """
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path
    
    def find_path(self, context: PathfindingContext) -> Optional[List[Position]]:
        """
        Find path using A* algorithm
        
        Args:
            context: Pathfinding context with all necessary information
            
        Returns:
            List of positions representing the path, or None if no path found
        """
        # Initialize start and goal positions
        start_pos = Position(context.start[0], context.start[1], context.start_time)
        goal_spatial = context.goal
        
        # Check if start position is valid
        if self.is_position_blocked(start_pos, context):
            if self.debug_mode:
                print(f"Start position {start_pos} is blocked")
            return None
        
        # A* data structures
        open_set = []  # Priority queue
        closed_set: Set[Position] = set()
        
        # Node tracking
        g_scores: Dict[Position, float] = defaultdict(lambda: float('inf'))
        nodes: Dict[Position, PathNode] = {}
        
        # Initialize start node
        start_h = self.heuristic((start_pos.x, start_pos.y), goal_spatial)
        start_node = PathNode(start_pos, 0.0, start_h)
        
        g_scores[start_pos] = 0.0
        nodes[start_pos] = start_node
        heapq.heappush(open_set, start_node)
        
        nodes_expanded = 0
        max_nodes = self.grid.width * self.grid.height * context.max_time
        
        while open_set and nodes_expanded < max_nodes:
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            # Skip if already processed
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            nodes_expanded += 1
            
            # Check if we reached the goal (spatial position)
            if (current_pos.x == goal_spatial[0] and 
                current_pos.y == goal_spatial[1]):
                if self.debug_mode:
                    print(f"Path found after expanding {nodes_expanded} nodes")
                return self.reconstruct_path(current_node)
            
            # Explore neighbors
            neighbors = self.get_valid_neighbors(current_pos, context)
            
            for neighbor_pos in neighbors:
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate movement cost
                if (neighbor_pos.x == current_pos.x and 
                    neighbor_pos.y == current_pos.y):
                    # Waiting in place
                    move_cost = context.wait_cost
                else:
                    # Moving to adjacent cell
                    move_cost = context.move_cost
                
                # Add time penalty for longer paths
                time_penalty = neighbor_pos.t * context.time_penalty
                
                tentative_g = current_node.g_cost + move_cost + time_penalty
                
                # Skip if we found a worse path
                if tentative_g >= g_scores[neighbor_pos]:
                    continue
                
                # Calculate heuristic
                h_cost = self.heuristic((neighbor_pos.x, neighbor_pos.y), goal_spatial)
                
                # Create or update neighbor node
                neighbor_node = PathNode(neighbor_pos, tentative_g, h_cost, current_node)
                
                g_scores[neighbor_pos] = tentative_g
                nodes[neighbor_pos] = neighbor_node
                heapq.heappush(open_set, neighbor_node)
        
        if self.debug_mode:
            print(f"No path found after expanding {nodes_expanded} nodes")
        return None
    
    def find_path_for_agent(self, agent: Agent, 
                           emergency_paths: Dict[int, List[Position]] = None,
                           higher_priority_paths: Dict[int, List[Position]] = None,
                           start_time: int = 0,
                           max_time: int = None) -> Optional[List[Position]]:
        """
        Convenience method to find path for an agent
        
        Args:
            agent: Agent to find path for
            emergency_paths: Emergency agent paths to avoid
            higher_priority_paths: Higher priority agent paths to avoid (for auctions)
            start_time: Starting time for pathfinding
            max_time: Maximum time limit
            
        Returns:
            Path as list of positions, or None if no path found
        """
        if emergency_paths is None:
            emergency_paths = {}
        if higher_priority_paths is None:
            higher_priority_paths = {}
        if max_time is None:
            max_time = self.grid.max_time
        
        context = PathfindingContext(
            agent_id=agent.id,
            start=agent.start,
            goal=agent.goal,
            start_time=start_time,
            max_time=max_time,
            emergency_paths=emergency_paths,
            higher_priority_paths=higher_priority_paths
        )
        
        return self.find_path(context)
    
    def find_paths_for_multiple_agents(self, 
                                     agents: List[Agent],
                                     emergency_paths: Dict[int, List[Position]] = None,
                                     agent_priorities: List[int] = None,
                                     start_time: int = 0,
                                     max_time: int = None) -> Dict[int, Optional[List[Position]]]:
        """
        Find paths for multiple agents with priority ordering
        Higher priority agents' paths become obstacles for lower priority agents
        
        Args:
            agents: List of agents to find paths for
            emergency_paths: Emergency agent paths to avoid
            agent_priorities: List of agent IDs in priority order (highest first)
            start_time: Starting time for pathfinding
            max_time: Maximum time limit
            
        Returns:
            Dictionary mapping agent_id to path (or None if no path found)
        """
        if emergency_paths is None:
            emergency_paths = {}
        if max_time is None:
            max_time = self.grid.max_time
        
        # Use provided priorities or default to agent order
        if agent_priorities is None:
            agent_priorities = [agent.id for agent in agents]
        
        results: Dict[int, Optional[List[Position]]] = {}
        assigned_paths: Dict[int, List[Position]] = {}
        
        # Create agent lookup
        agent_dict = {agent.id: agent for agent in agents}
        
        # Process agents in priority order
        for agent_id in agent_priorities:
            if agent_id not in agent_dict:
                continue
                
            agent = agent_dict[agent_id]
            
            # Find path considering emergency paths and higher priority assigned paths
            context = PathfindingContext(
                agent_id=agent.id,
                start=agent.start,
                goal=agent.goal,
                start_time=start_time,
                max_time=max_time,
                emergency_paths=emergency_paths,
                higher_priority_paths=assigned_paths.copy()
            )
            
            path = self.find_path(context)
            results[agent_id] = path
            
            # If path found, add to assigned paths for subsequent agents
            if path is not None:
                assigned_paths[agent_id] = path
        
        return results
    
    def validate_path(self, path: List[Position], agent_id: int,
                     emergency_paths: Dict[int, List[Position]] = None,
                     other_paths: Dict[int, List[Position]] = None) -> bool:
        """
        Validate that a path is conflict-free
        
        Args:
            path: Path to validate
            agent_id: ID of agent owning the path
            emergency_paths: Emergency agent paths to check against
            other_paths: Other agent paths to check against
            
        Returns:
            True if path is valid (no conflicts)
        """
        if not path:
            return False
        
        # Check path continuity
        if not self.grid.validate_path_continuity(path):
            return False
        
        if emergency_paths is None:
            emergency_paths = {}
        if other_paths is None:
            other_paths = {}
        
        # Check each position in the path
        for pos in path:
            # Check grid validity
            if not self.grid.is_cell_free(pos, ignore_agent_id=agent_id):
                return False
            
            # Check emergency path conflicts
            for emergency_agent_id, emergency_path in emergency_paths.items():
                if emergency_agent_id != agent_id:
                    for emergency_pos in emergency_path:
                        if (emergency_pos.x == pos.x and 
                            emergency_pos.y == pos.y and 
                            emergency_pos.t == pos.t):
                            return False
            
            # Check other path conflicts
            for other_agent_id, other_path in other_paths.items():
                if other_agent_id != agent_id:
                    for other_pos in other_path:
                        if (other_pos.x == pos.x and 
                            other_pos.y == pos.y and 
                            other_pos.t == pos.t):
                            return False
        
        return True
    
    def get_path_cost(self, path: List[Position]) -> float:
        """
        Calculate total cost of a path
        
        Args:
            path: Path to calculate cost for
            
        Returns:
            Total path cost
        """
        if not path:
            return float('inf')
        
        total_cost = 0.0
        
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            # Check if this is a wait action
            if (curr_pos.x == prev_pos.x and curr_pos.y == prev_pos.y):
                total_cost += 1.0  # Wait cost
            else:
                total_cost += 1.0  # Move cost
        
        return total_cost
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug output"""
        self.debug_mode = enabled
    
    def clear_cache(self):
        """Clear internal caches"""
        self._heuristic_cache.clear()
    
    def get_statistics(self) -> Dict:
        """Get pathfinding statistics"""
        return {
            'heuristic_cache_size': len(self._heuristic_cache),
            'grid_dimensions': (self.grid.width, self.grid.height, self.grid.max_time)
        }

# Utility functions for integration with other modules

def paths_to_tuples(paths: Dict[int, List[Position]]) -> Dict[int, List[Tuple[int, int, int]]]:
    """Convert Position objects to tuple format for CBS module"""
    return {
        agent_id: [pos.to_tuple() for pos in path] 
        for agent_id, path in paths.items() 
        if path is not None
    }

def tuples_to_paths(tuple_paths: Dict[int, List[Tuple[int, int, int]]]) -> Dict[int, List[Position]]:
    """Convert tuple format back to Position objects"""
    return {
        agent_id: [Position(x, y, t) for x, y, t in path]
        for agent_id, path in tuple_paths.items()
    }

def extract_emergency_paths_from_grid(grid: GridSystem) -> Dict[int, List[Position]]:
    """Extract emergency agent paths from grid system"""
    emergency_paths = {}
    
    for agent_id in grid.emergency_agents:
        if agent_id in grid.agent_paths:
            emergency_paths[agent_id] = grid.agent_paths[agent_id].copy()
    
    return emergency_paths

def create_pathfinding_context_from_agent(agent: Agent, 
                                        grid: GridSystem,
                                        start_time: int = 0,
                                        higher_priority_paths: Dict[int, List[Position]] = None) -> PathfindingContext:
    """
    Create pathfinding context from agent and grid system
    Automatically extracts emergency paths from grid
    """
    emergency_paths = extract_emergency_paths_from_grid(grid)
    
    if higher_priority_paths is None:
        higher_priority_paths = {}
    
    return PathfindingContext(
        agent_id=agent.id,
        start=agent.start,
        goal=agent.goal,
        start_time=start_time,
        max_time=grid.max_time,
        emergency_paths=emergency_paths,
        higher_priority_paths=higher_priority_paths
    )