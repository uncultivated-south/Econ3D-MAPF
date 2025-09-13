"""
A* Pathfinding Module for UrbanAirspaceSim
Handles individual agent pathfinding with dynamic obstacle avoidance and constraint support
REFACTORED VERSION - Addresses identified bugs and integrates with improved grid system
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import math
from enum import Enum
import time
import threading

# Import from refactored grid system
from grid_system import Position, Agent, AgentType, GridSystem

class ConstraintType(Enum):
    """Types of constraints for pathfinding"""
    VERTEX = "vertex"  # Cannot occupy specific position at specific time
    EDGE = "edge"      # Cannot traverse specific edge at specific time
    TEMPORAL = "temporal"  # Cannot be at position during time range

@dataclass
class Constraint:
    """Represents a pathfinding constraint"""
    constraint_type: ConstraintType
    position: Position
    agent_id: Optional[int] = None  # Which agent this constraint applies to
    
    # For edge constraints
    prev_position: Optional[Position] = None
    
    # For temporal constraints
    time_range: Optional[Tuple[int, int]] = None  # (start_time, end_time)
    
    def __hash__(self):
        return hash((self.constraint_type, self.position, self.agent_id, 
                    self.prev_position, self.time_range))
    
    def applies_to_transition(self, from_pos: Position, to_pos: Position) -> bool:
        """Check if this constraint applies to a specific transition"""
        if self.constraint_type == ConstraintType.VERTEX:
            return to_pos == self.position
        
        elif self.constraint_type == ConstraintType.EDGE:
            return (self.prev_position == from_pos and self.position == to_pos)
        
        elif self.constraint_type == ConstraintType.TEMPORAL:
            if self.time_range and to_pos.spatial_position() == self.position.spatial_position():
                return self.time_range[0] <= to_pos.t <= self.time_range[1]
        
        return False

@dataclass
class PathNode:
    """Node for A* pathfinding with proper ordering and constraint tracking"""
    position: Position
    g_cost: float  # Cost from start
    h_cost: float  # Heuristic cost to goal
    f_cost: float = field(init=False)  # Total cost
    parent: Optional['PathNode'] = None
    
    # Additional information for debugging and analysis
    violated_constraints: List[Constraint] = field(default_factory=list)
    movement_cost: float = 1.0  # Cost of the move that led to this node
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        if abs(self.f_cost - other.f_cost) < 1e-9:  # Handle floating point precision
            # Tie-breaking: prefer higher g_cost (closer to goal)
            if abs(self.g_cost - other.g_cost) < 1e-9:
                # Secondary tie-breaking: prefer fewer constraint violations
                return len(self.violated_constraints) < len(other.violated_constraints)
            return self.g_cost > other.g_cost
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return (isinstance(other, PathNode) and 
                self.position == other.position and
                abs(self.f_cost - other.f_cost) < 1e-9)

class HeuristicType(Enum):
    """Types of heuristic functions"""
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    DIAGONAL = "diagonal"
    CUSTOM = "custom"

@dataclass
class PathfindingConfig:
    """Configuration for pathfinding behavior"""
    # Heuristic settings
    heuristic_type: HeuristicType = HeuristicType.MANHATTAN
    heuristic_weight: float = 1.0  # Weight for heuristic (>1 = weighted A*)
    
    # Cost settings
    move_cost: float = 1.0
    wait_cost: float = 1.0
    diagonal_cost: float = math.sqrt(2)
    time_penalty: float = 0.01  # Small penalty for longer paths
    
    # Search limits
    max_nodes_expanded: int = 10000
    max_time_seconds: float = 30.0
    
    # Constraint handling
    soft_constraints: bool = False  # Allow violating constraints with penalty
    constraint_violation_penalty: float = 100.0
    
    # Performance settings
    cache_heuristics: bool = True
    use_jump_point_search: bool = False  # Future optimization
    
    # Debug settings
    track_search_stats: bool = False

@dataclass
class PathfindingResult:
    """Result of pathfinding operation"""
    success: bool
    path: List[Position]
    cost: float
    nodes_expanded: int
    computation_time: float
    
    # Additional information
    constraints_violated: List[Constraint] = field(default_factory=list)
    failure_reason: str = ""
    search_stats: Dict = field(default_factory=dict)

class AStarPathfinder:
    """
    Enhanced A* pathfinding implementation with constraint support and improved integration
    """
    
    def __init__(self, grid_system: GridSystem, config: PathfindingConfig = None):
        """
        Initialize pathfinder with grid system and configuration
        
        Args:
            grid_system: The grid system instance
            config: Pathfinding configuration (uses defaults if None)
        """
        self.grid = grid_system
        self.config = config or PathfindingConfig()
        
        # Separate caches for different heuristic types to prevent collision
        self._heuristic_caches: Dict[HeuristicType, Dict[Tuple, float]] = {
            htype: {} for htype in HeuristicType
        }
        
        # Constraint management
        self.active_constraints: Dict[int, List[Constraint]] = defaultdict(list)  # agent_id -> constraints
        self.global_constraints: List[Constraint] = []  # Apply to all agents
        
        # Thread safety for constraint management
        self._constraint_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_nodes_expanded': 0,
            'total_computation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Custom heuristic function (for CUSTOM heuristic type)
        self.custom_heuristic: Optional[Callable[[Position, Tuple[int, int]], float]] = None
    
    def set_custom_heuristic(self, heuristic_func: Callable[[Position, Tuple[int, int]], float]):
        """Set custom heuristic function"""
        self.custom_heuristic = heuristic_func
    
    def add_constraint(self, constraint: Constraint, agent_id: Optional[int] = None):
        """
        Add a constraint for specific agent or globally
        
        Args:
            constraint: Constraint to add
            agent_id: Agent to apply constraint to (None for global)
        """
        with self._constraint_lock:
            if agent_id is None:
                self.global_constraints.append(constraint)
            else:
                self.active_constraints[agent_id].append(constraint)
    
    def remove_constraints_for_agent(self, agent_id: int):
        """Remove all constraints for a specific agent"""
        with self._constraint_lock:
            if agent_id in self.active_constraints:
                del self.active_constraints[agent_id]
    
    def clear_all_constraints(self):
        """Clear all constraints"""
        with self._constraint_lock:
            self.active_constraints.clear()
            self.global_constraints.clear()
    
    def _get_heuristic_cache(self, heuristic_type: HeuristicType) -> Dict[Tuple, float]:
        """Get the appropriate heuristic cache"""
        return self._heuristic_caches[heuristic_type]
    
    def _calculate_heuristic(self, pos: Position, goal: Tuple[int, int], 
                            heuristic_type: HeuristicType = None) -> float:
        """
        Calculate heuristic with caching and multiple heuristic types
        
        Args:
            pos: Current position
            goal: Goal position (x, y)
            heuristic_type: Type of heuristic to use (uses config default if None)
            
        Returns:
            Heuristic cost estimate
        """
        if heuristic_type is None:
            heuristic_type = self.config.heuristic_type
        
        spatial_pos = (pos.x, pos.y)
        
        # Check cache if enabled
        if self.config.cache_heuristics:
            cache = self._get_heuristic_cache(heuristic_type)
            cache_key = (spatial_pos, goal)
            
            if cache_key in cache:
                self.stats['cache_hits'] += 1
                return cache[cache_key] * self.config.heuristic_weight
            else:
                self.stats['cache_misses'] += 1
        
        # Calculate heuristic
        if heuristic_type == HeuristicType.MANHATTAN:
            h_cost = abs(spatial_pos[0] - goal[0]) + abs(spatial_pos[1] - goal[1])
        
        elif heuristic_type == HeuristicType.EUCLIDEAN:
            h_cost = math.sqrt((spatial_pos[0] - goal[0])**2 + (spatial_pos[1] - goal[1])**2)
        
        elif heuristic_type == HeuristicType.DIAGONAL:
            dx = abs(spatial_pos[0] - goal[0])
            dy = abs(spatial_pos[1] - goal[1])
            h_cost = max(dx, dy) + (self.config.diagonal_cost - 1) * min(dx, dy)
        
        elif heuristic_type == HeuristicType.CUSTOM:
            if self.custom_heuristic is None:
                raise ValueError("Custom heuristic function not set")
            h_cost = self.custom_heuristic(pos, goal)
        
        else:
            h_cost = 0.0
        
        # Cache the result
        if self.config.cache_heuristics:
            cache[cache_key] = h_cost
        
        return h_cost * self.config.heuristic_weight
    
    def _get_constraints_for_agent(self, agent_id: int) -> List[Constraint]:
        """Get all constraints applying to a specific agent"""
        with self._constraint_lock:
            constraints = self.global_constraints.copy()
            constraints.extend(self.active_constraints.get(agent_id, []))
            return constraints
    
    def _is_transition_constrained(self, from_pos: Position, to_pos: Position, 
                                  agent_id: int) -> Tuple[bool, List[Constraint]]:
        """
        Check if a transition violates any constraints
        
        Args:
            from_pos: Starting position
            to_pos: Destination position
            agent_id: Agent making the transition
            
        Returns:
            Tuple of (is_constrained, violated_constraints)
        """
        constraints = self._get_constraints_for_agent(agent_id)
        violated = []
        
        for constraint in constraints:
            # Skip constraints that don't apply to this agent
            if constraint.agent_id is not None and constraint.agent_id != agent_id:
                continue
            
            if constraint.applies_to_transition(from_pos, to_pos):
                violated.append(constraint)
        
        return len(violated) > 0, violated
    
    def _calculate_movement_cost(self, from_pos: Position, to_pos: Position) -> float:
        """Calculate cost of movement between two positions"""
        if from_pos.spatial_position() == to_pos.spatial_position():
            # Waiting in place
            return self.config.wait_cost
        
        # Calculate spatial distance
        dx = abs(to_pos.x - from_pos.x)
        dy = abs(to_pos.y - from_pos.y)
        
        if dx + dy == 1:
            # Adjacent move
            return self.config.move_cost
        elif dx == 1 and dy == 1:
            # Diagonal move (if supported)
            return self.config.diagonal_cost
        else:
            # Invalid move
            return float('inf')
    
    def _get_valid_successors(self, current_node: PathNode, goal: Tuple[int, int], 
                             agent_id: int) -> List[PathNode]:
        """
        Get valid successor nodes considering constraints and occupancy
        
        Args:
            current_node: Current node in search
            goal: Goal position
            agent_id: Agent performing search
            
        Returns:
            List of valid successor nodes
        """
        successors = []
        neighbors = self.grid.get_neighbors(current_node.position, include_wait=True)
        
        for neighbor_pos in neighbors:
            # Check if position is free in grid (considering priorities)
            if not self.grid.is_cell_free(neighbor_pos, ignore_agent_id=agent_id, 
                                        check_priority=True):
                continue
            
            # Calculate movement cost
            movement_cost = self._calculate_movement_cost(current_node.position, neighbor_pos)
            if movement_cost == float('inf'):
                continue  # Invalid movement
            
            # Check constraints
            is_constrained, violated_constraints = self._is_transition_constrained(
                current_node.position, neighbor_pos, agent_id
            )
            
            # Handle constraint violations
            if is_constrained and not self.config.soft_constraints:
                continue  # Hard constraints - skip this successor
            
            # Calculate costs
            g_cost = current_node.g_cost + movement_cost
            
            # Add time penalty for longer paths
            g_cost += neighbor_pos.t * self.config.time_penalty
            
            # Add constraint violation penalty if using soft constraints
            if is_constrained and self.config.soft_constraints:
                g_cost += len(violated_constraints) * self.config.constraint_violation_penalty
            
            # Calculate heuristic
            h_cost = self._calculate_heuristic(neighbor_pos, goal)
            
            # Create successor node
            successor = PathNode(
                position=neighbor_pos,
                g_cost=g_cost,
                h_cost=h_cost,
                parent=current_node,
                violated_constraints=violated_constraints,
                movement_cost=movement_cost
            )
            
            successors.append(successor)
        
        return successors
    
    def _reconstruct_path(self, goal_node: PathNode) -> List[Position]:
        """Reconstruct path from goal node"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        path.reverse()
        return path
    
    def find_path(self, agent: Agent, start_time: int = 0, 
                 max_time: Optional[int] = None) -> PathfindingResult:
        """
        Find path for an agent using A* with constraint support
        
        Args:
            agent: Agent to find path for
            start_time: Starting time for pathfinding
            max_time: Maximum time limit (uses grid max if None)
            
        Returns:
            PathfindingResult with path and metadata
        """
        start_search_time = time.time()
        self.stats['total_searches'] += 1
        
        if max_time is None:
            max_time = self.grid.max_time
        
        # Initialize search
        start_pos = Position(agent.start[0], agent.start[1], start_time)
        goal_spatial = agent.goal
        
        # Validate start position
        if not self.grid.is_cell_free(start_pos, ignore_agent_id=agent.id, check_priority=True):
            return PathfindingResult(
                success=False,
                path=[],
                cost=float('inf'),
                nodes_expanded=0,
                computation_time=time.time() - start_search_time,
                failure_reason="Start position is blocked"
            )
        
        # A* data structures
        open_set = []
        closed_set: Set[Position] = set()
        g_scores: Dict[Position, float] = defaultdict(lambda: float('inf'))
        
        # Initialize start node
        start_h = self._calculate_heuristic(start_pos, goal_spatial)
        start_node = PathNode(start_pos, 0.0, start_h)
        
        g_scores[start_pos] = 0.0
        heapq.heappush(open_set, start_node)
        
        nodes_expanded = 0
        best_node = start_node
        best_distance_to_goal = start_h
        
        # Search loop
        while open_set and nodes_expanded < self.config.max_nodes_expanded:
            # Check time limit
            if time.time() - start_search_time > self.config.max_time_seconds:
                break
            
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            # Skip if already processed
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            nodes_expanded += 1
            
            # Track best node found (closest to goal)
            distance_to_goal = self._calculate_heuristic(current_pos, goal_spatial, 
                                                       HeuristicType.MANHATTAN) / self.config.heuristic_weight
            if distance_to_goal < best_distance_to_goal:
                best_node = current_node
                best_distance_to_goal = distance_to_goal
            
            # Check if we reached the goal
            if (current_pos.x == goal_spatial[0] and 
                current_pos.y == goal_spatial[1]):
                
                path = self._reconstruct_path(current_node)
                computation_time = time.time() - start_search_time
                
                self.stats['successful_searches'] += 1
                self.stats['total_nodes_expanded'] += nodes_expanded
                self.stats['total_computation_time'] += computation_time
                
                return PathfindingResult(
                    success=True,
                    path=path,
                    cost=current_node.g_cost,
                    nodes_expanded=nodes_expanded,
                    computation_time=computation_time,
                    search_stats={
                        'final_f_cost': current_node.f_cost,
                        'path_length': len(path),
                        'constraints_violated': sum(len(node.violated_constraints) 
                                                  for node in self._get_path_nodes(current_node))
                    } if self.config.track_search_stats else {}
                )
            
            # Expand successors
            successors = self._get_valid_successors(current_node, goal_spatial, agent.id)
            
            for successor in successors:
                successor_pos = successor.position
                
                # Skip if time exceeds limit
                if successor_pos.t >= max_time:
                    continue
                
                # Skip if already closed
                if successor_pos in closed_set:
                    continue
                
                # Skip if we found a better path to this position
                if successor.g_cost >= g_scores[successor_pos]:
                    continue
                
                g_scores[successor_pos] = successor.g_cost
                heapq.heappush(open_set, successor)
        
        # Search failed
        computation_time = time.time() - start_search_time
        self.stats['total_nodes_expanded'] += nodes_expanded
        self.stats['total_computation_time'] += computation_time
        
        # Return partial path to best node found
        failure_reason = "No path found"
        if nodes_expanded >= self.config.max_nodes_expanded:
            failure_reason = "Node expansion limit reached"
        elif time.time() - start_search_time > self.config.max_time_seconds:
            failure_reason = "Time limit exceeded"
        
        return PathfindingResult(
            success=False,
            path=self._reconstruct_path(best_node) if best_node != start_node else [],
            cost=best_node.g_cost,
            nodes_expanded=nodes_expanded,
            computation_time=computation_time,
            failure_reason=failure_reason,
            search_stats={
                'best_distance_to_goal': best_distance_to_goal,
                'open_set_size': len(open_set)
            } if self.config.track_search_stats else {}
        )
    
    def _get_path_nodes(self, goal_node: PathNode) -> List[PathNode]:
        """Get all nodes in path from goal to start"""
        nodes = []
        current = goal_node
        while current is not None:
            nodes.append(current)
            current = current.parent
        return list(reversed(nodes))
    
    def find_paths_for_agents(self, agents: List[Agent], 
                             priority_order: Optional[List[int]] = None,
                             start_time: int = 0) -> Dict[int, PathfindingResult]:
        """
        Find paths for multiple agents with priority-based planning
        
        Args:
            agents: List of agents to find paths for
            priority_order: Order to plan agents (highest priority first)
            start_time: Starting time for all agents
            
        Returns:
            Dictionary mapping agent_id to PathfindingResult
        """
        results: Dict[int, PathfindingResult] = {}
        
        # Sort agents by priority if no explicit order given
        if priority_order is None:
            sorted_agents = sorted(agents, key=lambda a: a.priority, reverse=True)
        else:
            agent_dict = {a.id: a for a in agents}
            sorted_agents = [agent_dict[aid] for aid in priority_order if aid in agent_dict]
        
        # Plan paths in priority order
        for agent in sorted_agents:
            result = self.find_path(agent, start_time)
            results[agent.id] = result
            
            # If successful, temporarily reserve the path for subsequent planning
            if result.success:
                # Convert path to constraints for lower-priority agents
                path_constraints = self._path_to_constraints(result.path, agent.id)
                
                # Add constraints for remaining agents
                for remaining_agent in sorted_agents:
                    if remaining_agent.priority < agent.priority:
                        for constraint in path_constraints:
                            self.add_constraint(constraint, remaining_agent.id)
        
        # Clean up temporary constraints
        for agent in sorted_agents:
            self.remove_constraints_for_agent(agent.id)
        
        return results
    
    def _path_to_constraints(self, path: List[Position], agent_id: int) -> List[Constraint]:
        """Convert a path to vertex constraints for other agents"""
        constraints = []
        
        for pos in path:
            constraint = Constraint(
                constraint_type=ConstraintType.VERTEX,
                position=pos,
                agent_id=None  # Apply to all other agents
            )
            constraints.append(constraint)
        
        return constraints
    
    def validate_path(self, path: List[Position], agent_id: int) -> Tuple[bool, List[str]]:
        """
        Validate a path for correctness and constraint compliance
        
        Args:
            path: Path to validate
            agent_id: Agent owning the path
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if not path:
            return False, ["Empty path"]
        
        issues = []
        
        # Check path continuity
        if not self.grid.validate_path_continuity(path):
            issues.append("Path is not continuous")
        
        # Check for constraint violations
        constraints = self._get_constraints_for_agent(agent_id)
        
        for i in range(1, len(path)):
            from_pos = path[i-1]
            to_pos = path[i]
            
            is_constrained, violated = self._is_transition_constrained(from_pos, to_pos, agent_id)
            if is_constrained:
                issues.append(f"Constraint violation at {to_pos}: {len(violated)} constraints")
        
        # Check grid occupancy
        conflicts = self.grid.validate_path_conflicts(path, agent_id)
        if conflicts:
            issues.append(f"Grid conflicts at {len(conflicts)} positions")
        
        return len(issues) == 0, issues
    
    def get_statistics(self) -> Dict:
        """Get pathfinding statistics"""
        total_time = max(self.stats['total_computation_time'], 1e-9)  # Avoid division by zero
        
        return {
            'total_searches': self.stats['total_searches'],
            'successful_searches': self.stats['successful_searches'],
            'success_rate': self.stats['successful_searches'] / max(self.stats['total_searches'], 1),
            'total_nodes_expanded': self.stats['total_nodes_expanded'],
            'average_nodes_per_search': self.stats['total_nodes_expanded'] / max(self.stats['total_searches'], 1),
            'total_computation_time': self.stats['total_computation_time'],
            'average_time_per_search': self.stats['total_computation_time'] / max(self.stats['total_searches'], 1),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'searches_per_second': self.stats['total_searches'] / total_time
        }
    
    def clear_caches(self):
        """Clear all internal caches"""
        for cache in self._heuristic_caches.values():
            cache.clear()
    
    def reset_statistics(self):
        """Reset pathfinding statistics"""
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'total_nodes_expanded': 0,
            'total_computation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

# Utility functions for integration

def create_pathfinder(grid_system: GridSystem, config: PathfindingConfig = None) -> AStarPathfinder:
    """Create pathfinder with default or custom configuration"""
    return AStarPathfinder(grid_system, config)

def create_conservative_config() -> PathfindingConfig:
    """Create configuration for conservative (thorough) pathfinding"""
    return PathfindingConfig(
        heuristic_weight=1.0,  # Admissible A*
        max_nodes_expanded=50000,
        max_time_seconds=60.0,
        soft_constraints=False,
        track_search_stats=True
    )

def create_fast_config() -> PathfindingConfig:
    """Create configuration for fast (less thorough) pathfinding"""
    return PathfindingConfig(
        heuristic_weight=1.2,  # Weighted A* for speed
        max_nodes_expanded=5000,
        max_time_seconds=5.0,
        soft_constraints=True,
        track_search_stats=False
    )

def constraints_from_paths(paths: Dict[int, List[Position]], 
                          exclude_agent: Optional[int] = None) -> List[Constraint]:
    """
    Convert existing agent paths to constraints for pathfinding
    
    Args:
        paths: Dictionary of agent_id -> path
        exclude_agent: Agent to exclude from constraint generation
        
    Returns:
        List of vertex constraints
    """
    constraints = []
    
    for agent_id, path in paths.items():
        if exclude_agent is not None and agent_id == exclude_agent:
            continue
        
        for pos in path:
            constraint = Constraint(
                constraint_type=ConstraintType.VERTEX,
                position=pos
            )
            constraints.append(constraint)
    
    return constraints

def extract_emergency_constraints(grid_system: GridSystem) -> List[Constraint]:
    """Extract constraints from emergency agent paths in grid system"""
    emergency_paths = grid_system.get_emergency_paths()
    return constraints_from_paths(emergency_paths)

def convert_result_to_grid_format(result: PathfindingResult) -> Optional[List[Position]]:
    """Convert pathfinding result to format expected by grid system"""
    return result.path if result.success else None

def batch_pathfind_with_constraints(pathfinder: AStarPathfinder, 
                                   agents: List[Agent],
                                   existing_paths: Dict[int, List[Position]]) -> Dict[int, PathfindingResult]:
    """
    Perform batch pathfinding with existing paths as constraints
    
    Args:
        pathfinder: Pathfinder instance
        agents: Agents to find paths for
        existing_paths: Paths to treat as obstacles
        
    Returns:
        Dictionary of results
    """
    # Clear existing constraints
    pathfinder.clear_all_constraints()
    
    # Add existing paths as constraints
    constraints = constraints_from_paths(existing_paths)
    for constraint in constraints:
        pathfinder.add_constraint(constraint)
    
    # Find paths for agents
    results = pathfinder.find_paths_for_agents(agents)
    
    # Clean up constraints
    pathfinder.clear_all_constraints()
    
    return results