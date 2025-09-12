"""
Enhanced A* Pathfinding Module with Emergency Agent Priority
Integrates with EnhancedMultilayerGridSystem to handle emergency agent routing
and dynamic obstacle avoidance for regular agents.
"""

import heapq
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import time

# Import the enhanced multilayer grid system components
from enhanced_multilayer_grid_system import (
    EnhancedMultilayerGridSystem, LayerType, AgentState, AgentType, ProcessingPhase
)

class PathfindingResult(Enum):
    """Enumeration for pathfinding operation results"""
    SUCCESS = "success"
    NO_PATH = "no_path"
    START_BLOCKED = "start_blocked"
    GOAL_BLOCKED = "goal_blocked"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    EMERGENCY_CONFLICT = "emergency_conflict"

@dataclass
class PathNode:
    """Node representation for A* pathfinding algorithm"""
    position: Tuple[int, int]
    time_step: int
    g_cost: float = 0.0  # Cost from start
    h_cost: float = 0.0  # Heuristic cost to goal
    f_cost: float = field(init=False)  # Total cost
    parent: Optional['PathNode'] = None
    emergency_agent_conflict: Optional[int] = None  # ID of conflicting emergency agent
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        return self.h_cost < other.h_cost
    
    def __eq__(self, other):
        return (self.position == other.position and 
                self.time_step == other.time_step)
    
    def __hash__(self):
        return hash((self.position, self.time_step))

@dataclass 
class PathfindingRequest:
    """Request structure for pathfinding operations"""
    agent_id: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    start_time: int
    layer: LayerType
    agent_type: AgentType
    max_time_steps: int = 100
    allow_waiting: bool = True
    priority_weight: float = 1.0

@dataclass
class PathfindingResponse:
    """Response structure for pathfinding operations"""
    agent_id: int
    agent_type: AgentType
    result: PathfindingResult
    path: List[Tuple[int, int]]
    time_steps: List[int]
    total_cost: float
    computation_time: float
    nodes_explored: int
    emergency_conflicts: List[int]  # List of emergency agents that blocked path
    message: str = ""

class EnhancedAStarPathfinder:
    """
    Enhanced A* pathfinder with emergency agent priority and dynamic obstacle awareness.
    
    Handles two-tier pathfinding:
    1. Emergency agents: Only consider static obstacles
    2. Regular agents: Consider static obstacles + emergency agent reservations
    """
    
    def __init__(self, grid_system: EnhancedMultilayerGridSystem, max_iterations: int = 10000):
        """
        Initialize the enhanced A* pathfinder.
        
        Args:
            grid_system: EnhancedMultilayerGridSystem instance
            max_iterations: Maximum iterations before timeout
        """
        self.grid_system = grid_system
        self.max_iterations = max_iterations
        
        # Cost parameters (differentiated by agent type)
        self.emergency_movement_cost = 0.8  # Lower cost for emergency agents
        self.regular_movement_cost = 1.0    # Standard cost for regular agents
        self.waiting_cost = 0.1
        self.emergency_priority_bonus = 0.5
        
        # Performance tracking
        self.cache: Dict[Tuple, List[Tuple[int, int]]] = {}
        self.emergency_cache: Dict[Tuple, List[Tuple[int, int]]] = {}
        self.cache_hits = 0
        self.total_requests = 0
        self.emergency_requests = 0
        self.regular_requests = 0
        
        # Movement directions (4-connected grid)
        self.directions = [
            (0, 1),   # North
            (1, 0),   # East  
            (0, -1),  # South
            (-1, 0)   # West
        ]
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        x, y = position
        return (0 <= x < self.grid_system.width and 
                0 <= y < self.grid_system.height)
    
    def is_position_available(self, layer: LayerType, position: Tuple[int, int], 
                            time_step: int, agent_id: int, agent_type: AgentType) -> Tuple[bool, Optional[int]]:
        """
        Check if position is available considering agent type and emergency priorities.
        
        Args:
            layer: Layer to check
            position: Position (x, y) to check
            time_step: Time step to check
            agent_id: ID of requesting agent
            agent_type: Type of requesting agent
            
        Returns:
            Tuple: (is_available, blocking_emergency_agent_id)
        """
        if not self.is_valid_position(position):
            return False, None
        
        if time_step >= self.grid_system.max_time:
            return False, None
        
        if agent_type == AgentType.EMERGENCY:
            # Emergency agents only check static obstacles
            available = self.grid_system.is_position_available_for_emergency_agent(
                layer, position, time_step, agent_id
            )
            return available, None
        else:
            # Regular agents check both static obstacles and emergency reservations
            available = self.grid_system.is_position_available_for_regular_agent(
                layer, position, time_step, agent_id
            )
            
            if not available:
                # Check if blocked by emergency agent
                is_reserved, emergency_agent_id = self.grid_system.emergency_reservations.is_position_reserved(
                    layer, position, time_step
                )
                return False, emergency_agent_id if is_reserved else None
            
            return True, None
    
    def get_neighbors(self, node: PathNode, request: PathfindingRequest) -> List[Tuple[PathNode, Optional[int]]]:
        """
        Generate valid neighbor nodes for A* expansion.
        
        Args:
            node: Current node to expand
            request: Pathfinding request with constraints
            
        Returns:
            List[Tuple[PathNode, blocking_emergency_agent_id]]: Valid neighbor nodes with conflict info
        """
        neighbors = []
        next_time = node.time_step + 1
        
        if next_time >= request.max_time_steps:
            return neighbors
        
        # Determine movement cost based on agent type
        movement_cost = (self.emergency_movement_cost if request.agent_type == AgentType.EMERGENCY 
                        else self.regular_movement_cost)
        
        # Movement to adjacent positions
        for dx, dy in self.directions:
            new_pos = (node.position[0] + dx, node.position[1] + dy)
            
            available, blocking_agent = self.is_position_available(
                request.layer, new_pos, next_time, request.agent_id, request.agent_type
            )
            
            if available:
                # Calculate costs
                g_cost = node.g_cost + movement_cost
                if request.agent_type == AgentType.EMERGENCY:
                    g_cost -= self.emergency_priority_bonus
                
                h_cost = self.manhattan_distance(new_pos, request.goal)
                
                neighbor = PathNode(
                    position=new_pos,
                    time_step=next_time,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=node
                )
                neighbors.append((neighbor, None))
            else:
                # Store information about emergency conflicts for regular agents
                if request.agent_type == AgentType.REGULAR and blocking_agent is not None:
                    # Create a high-cost node to represent the conflict
                    g_cost = node.g_cost + movement_cost * 10  # High penalty
                    h_cost = self.manhattan_distance(new_pos, request.goal)
                    
                    conflict_neighbor = PathNode(
                        position=new_pos,
                        time_step=next_time,
                        g_cost=g_cost,
                        h_cost=h_cost,
                        parent=node,
                        emergency_agent_conflict=blocking_agent
                    )
                    # Don't add conflict nodes to expansion - just track for debugging
        
        # Waiting in current position (if allowed)
        if request.allow_waiting:
            available, blocking_agent = self.is_position_available(
                request.layer, node.position, next_time, request.agent_id, request.agent_type
            )
            
            if available:
                g_cost = node.g_cost + self.waiting_cost
                if request.agent_type == AgentType.EMERGENCY:
                    g_cost -= self.emergency_priority_bonus
                
                h_cost = self.manhattan_distance(node.position, request.goal)
                
                wait_neighbor = PathNode(
                    position=node.position,
                    time_step=next_time,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=node
                )
                neighbors.append((wait_neighbor, None))
        
        return neighbors
    
    def reconstruct_path(self, goal_node: PathNode) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Reconstruct path from goal node to start using parent pointers.
        
        Args:
            goal_node: Final node with path to start
            
        Returns:
            Tuple: (path positions, time steps, emergency conflicts)
        """
        path = []
        time_steps = []
        emergency_conflicts = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            time_steps.append(current.time_step)
            if current.emergency_agent_conflict is not None:
                emergency_conflicts.append(current.emergency_agent_conflict)
            current = current.parent
        
        # Reverse to get start-to-goal order
        path.reverse()
        time_steps.reverse()
        
        # Remove duplicates from emergency conflicts
        emergency_conflicts = list(set(emergency_conflicts))
        
        return path, time_steps, emergency_conflicts
    
    def find_path(self, request: PathfindingRequest) -> PathfindingResponse:
        """
        Find optimal path using enhanced A* algorithm with emergency priority.
        
        Args:
            request: PathfindingRequest with all necessary parameters
            
        Returns:
            PathfindingResponse: Complete pathfinding result
        """
        start_time = time.time()
        self.total_requests += 1
        
        if request.agent_type == AgentType.EMERGENCY:
            self.emergency_requests += 1
        else:
            self.regular_requests += 1
        
        # Input validation
        if not self.is_valid_position(request.start):
            return PathfindingResponse(
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                result=PathfindingResult.INVALID_INPUT,
                path=[],
                time_steps=[],
                total_cost=float('inf'),
                computation_time=time.time() - start_time,
                nodes_explored=0,
                emergency_conflicts=[],
                message="Invalid start position"
            )
        
        if not self.is_valid_position(request.goal):
            return PathfindingResponse(
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                result=PathfindingResult.INVALID_INPUT,
                path=[],
                time_steps=[],
                total_cost=float('inf'),
                computation_time=time.time() - start_time,
                nodes_explored=0,
                emergency_conflicts=[],
                message="Invalid goal position"
            )
        
        # Select appropriate cache based on agent type
        cache = self.emergency_cache if request.agent_type == AgentType.EMERGENCY else self.cache
        
        # Check cache for identical requests
        cache_key = (request.start, request.goal, request.start_time, 
                    request.layer, request.agent_type)
        if cache_key in cache:
            self.cache_hits += 1
            cached_path = cache[cache_key]
            return PathfindingResponse(
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                result=PathfindingResult.SUCCESS,
                path=cached_path,
                time_steps=list(range(request.start_time, request.start_time + len(cached_path))),
                total_cost=len(cached_path) * (self.emergency_movement_cost if request.agent_type == AgentType.EMERGENCY else self.regular_movement_cost),
                computation_time=time.time() - start_time,
                nodes_explored=0,
                emergency_conflicts=[],
                message="Path retrieved from cache"
            )
        
        # Check if start position is available
        start_available, blocking_agent = self.is_position_available(
            request.layer, request.start, request.start_time, request.agent_id, request.agent_type
        )
        
        if not start_available:
            result = PathfindingResult.EMERGENCY_CONFLICT if blocking_agent else PathfindingResult.START_BLOCKED
            message = f"Start position blocked by emergency agent {blocking_agent}" if blocking_agent else "Start position is blocked"
            
            return PathfindingResponse(
                agent_id=request.agent_id,
                agent_type=request.agent_type,
                result=result,
                path=[],
                time_steps=[],
                total_cost=float('inf'),
                computation_time=time.time() - start_time,
                nodes_explored=0,
                emergency_conflicts=[blocking_agent] if blocking_agent else [],
                message=message
            )
        
        # Initialize A* algorithm
        open_set = []
        closed_set: Set[Tuple[Tuple[int, int], int]] = set()
        
        # Create start node
        start_node = PathNode(
            position=request.start,
            time_step=request.start_time,
            g_cost=0.0,
            h_cost=self.manhattan_distance(request.start, request.goal)
        )
        
        heapq.heappush(open_set, start_node)
        nodes_explored = 0
        emergency_conflicts = []
        
        # A* main loop
        while open_set and nodes_explored < self.max_iterations:
            current_node = heapq.heappop(open_set)
            nodes_explored += 1
            
            # Check if goal reached
            if current_node.position == request.goal:
                path, time_steps, path_conflicts = self.reconstruct_path(current_node)
                emergency_conflicts.extend(path_conflicts)
                
                # Cache successful path
                cache[cache_key] = path
                
                return PathfindingResponse(
                    agent_id=request.agent_id,
                    agent_type=request.agent_type,
                    result=PathfindingResult.SUCCESS,
                    path=path,
                    time_steps=time_steps,
                    total_cost=current_node.g_cost,
                    computation_time=time.time() - start_time,
                    nodes_explored=nodes_explored,
                    emergency_conflicts=list(set(emergency_conflicts)),
                    message="Path found successfully"
                )
            
            # Mark as visited
            state_key = (current_node.position, current_node.time_step)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            # Expand neighbors
            neighbors = self.get_neighbors(current_node, request)
            for neighbor, blocking_emergency in neighbors:
                neighbor_key = (neighbor.position, neighbor.time_step)
                if neighbor_key not in closed_set:
                    heapq.heappush(open_set, neighbor)
                    
                    # Track emergency conflicts for regular agents
                    if blocking_emergency and blocking_emergency not in emergency_conflicts:
                        emergency_conflicts.append(blocking_emergency)
        
        # Determine failure reason
        if nodes_explored >= self.max_iterations:
            result = PathfindingResult.TIMEOUT
            message = f"Pathfinding timeout after {self.max_iterations} iterations"
        else:
            result = PathfindingResult.NO_PATH
            message = "No valid path exists to goal"
            
            # For regular agents, check if blocked by emergency agents
            if request.agent_type == AgentType.REGULAR and emergency_conflicts:
                result = PathfindingResult.EMERGENCY_CONFLICT
                message += f" (blocked by emergency agents: {emergency_conflicts})"
        
        return PathfindingResponse(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            result=result,
            path=[],
            time_steps=[],
            total_cost=float('inf'),
            computation_time=time.time() - start_time,
            nodes_explored=nodes_explored,
            emergency_conflicts=list(set(emergency_conflicts)),
            message=message
        )
    
    def find_paths_for_emergency_agents(self, emergency_agents: List[Dict], layer: LayerType, 
                                       start_time: int, max_time_steps: int = 200) -> Dict[int, PathfindingResponse]:
        """
        Find paths for emergency agents with priority processing.
        
        Args:
            emergency_agents: List of emergency agent info dictionaries
            layer: Layer to process
            start_time: Starting time step
            max_time_steps: Maximum time steps allowed
            
        Returns:
            Dict: agent_id -> PathfindingResponse
        """
        responses = {}
        
        for agent_info in emergency_agents:
            request = PathfindingRequest(
                agent_id=agent_info['id'],
                start=agent_info['start'],
                goal=agent_info['goal'],
                start_time=start_time,
                layer=layer,
                agent_type=AgentType.EMERGENCY,
                max_time_steps=max_time_steps,
                allow_waiting=True,
                priority_weight=2.0
            )
            
            response = self.find_path(request)
            responses[agent_info['id']] = response
        
        return responses
    
    def find_paths_for_regular_agents(self, regular_agents: List[Dict], layer: LayerType, 
                                     start_time: int, max_time_steps: int = 100) -> Dict[int, PathfindingResponse]:
        """
        Find paths for regular agents considering emergency reservations.
        
        Args:
            regular_agents: List of regular agent info dictionaries
            layer: Layer to process
            start_time: Starting time step
            max_time_steps: Maximum time steps allowed
            
        Returns:
            Dict: agent_id -> PathfindingResponse
        """
        responses = {}
        
        # Sort by some priority metric if needed (e.g., by agent_id for consistency)
        sorted_agents = sorted(regular_agents, key=lambda a: a['id'])
        
        for agent_info in sorted_agents:
            request = PathfindingRequest(
                agent_id=agent_info['id'],
                start=agent_info['start'],
                goal=agent_info['goal'],
                start_time=start_time,
                layer=layer,
                agent_type=AgentType.REGULAR,
                max_time_steps=max_time_steps,
                allow_waiting=True,
                priority_weight=1.0
            )
            
            response = self.find_path(request)
            responses[agent_info['id']] = response
        
        return responses
    
    def find_path_simple(self, agent_id: int, start: Tuple[int, int], goal: Tuple[int, int],
                        start_time: int, layer: LayerType, agent_type: AgentType) -> Tuple[List[Tuple[int, int]], bool]:
        """
        Simplified pathfinding interface for basic usage.
        
        Args:
            agent_id: Agent identifier
            start: Start position (x, y)
            goal: Goal position (x, y)
            start_time: Starting time step
            layer: Layer for pathfinding
            agent_type: Type of agent (emergency or regular)
            
        Returns:
            Tuple: (path as list of positions, success flag)
        """
        request = PathfindingRequest(
            agent_id=agent_id,
            start=start,
            goal=goal,
            start_time=start_time,
            layer=layer,
            agent_type=agent_type
        )
        
        response = self.find_path(request)
        return response.path, response.result == PathfindingResult.SUCCESS
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get pathfinding performance statistics.
        
        Returns:
            Dict: Performance metrics including emergency/regular breakdown
        """
        cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        emergency_percentage = (self.emergency_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'emergency_requests': self.emergency_requests,
            'regular_requests': self.regular_requests,
            'emergency_percentage': emergency_percentage,
            'cache_hits': self.cache_hits,
            'cache_hit_rate_percent': cache_hit_rate,
            'regular_cache_size': len(self.cache),
            'emergency_cache_size': len(self.emergency_cache),
            'max_iterations': self.max_iterations
        }
    
    def clear_cache(self) -> None:
        """Clear pathfinding cache and reset statistics."""
        self.cache.clear()
        self.emergency_cache.clear()
        self.cache_hits = 0
        self.total_requests = 0
        self.emergency_requests = 0
        self.regular_requests = 0
    
    def set_cost_parameters(self, emergency_movement_cost: float = 0.8, regular_movement_cost: float = 1.0,
                          waiting_cost: float = 0.1, emergency_priority_bonus: float = 0.5) -> None:
        """
        Update cost parameters for pathfinding.
        
        Args:
            emergency_movement_cost: Movement cost for emergency agents
            regular_movement_cost: Movement cost for regular agents
            waiting_cost: Cost for waiting in place
            emergency_priority_bonus: Priority bonus for emergency agents
        """
        self.emergency_movement_cost = emergency_movement_cost
        self.regular_movement_cost = regular_movement_cost
        self.waiting_cost = waiting_cost
        self.emergency_priority_bonus = emergency_priority_bonus
    
    def validate_path(self, path: List[Tuple[int, int]], time_steps: List[int],
                     layer: LayerType, agent_id: int, agent_type: AgentType) -> bool:
        """
        Validate that a path is feasible given current grid state and agent type.
        
        Args:
            path: Path to validate
            time_steps: Time steps for each position
            layer: Layer to check
            agent_id: Agent identifier
            agent_type: Type of agent
            
        Returns:
            bool: True if path is valid
        """
        if len(path) != len(time_steps):
            return False
        
        for i, (position, time_step) in enumerate(zip(path, time_steps)):
            available, _ = self.is_position_available(layer, position, time_step, agent_id, agent_type)
            if not available:
                return False
            
            # Check movement constraints (max 1 step per time unit)
            if i > 0:
                prev_pos = path[i-1]
                distance = self.manhattan_distance(prev_pos, position)
                if distance > 1:
                    return False
        
        return True
    
    def get_path_cost(self, path: List[Tuple[int, int]], time_steps: List[int], agent_type: AgentType) -> float:
        """
        Calculate total cost for a given path based on agent type.
        
        Args:
            path: Path positions
            time_steps: Time steps for each position
            agent_type: Type of agent (affects movement cost)
            
        Returns:
            float: Total path cost
        """
        if not path or len(path) != len(time_steps):
            return float('inf')
        
        movement_cost = (self.emergency_movement_cost if agent_type == AgentType.EMERGENCY 
                        else self.regular_movement_cost)
        
        total_cost = 0.0
        
        for i in range(len(path)):
            if i == 0:
                continue
            
            # Check if agent moved or waited
            if path[i] == path[i-1]:
                total_cost += self.waiting_cost
            else:
                total_cost += movement_cost
        
        # Apply emergency bonus
        if agent_type == AgentType.EMERGENCY:
            total_cost -= self.emergency_priority_bonus * (len(path) - 1)
        
        return max(0.0, total_cost)  # Ensure non-negative cost
    
    def export_pathfinding_config(self) -> Dict[str, Union[int, float]]:
        """
        Export current pathfinding configuration.
        
        Returns:
            Dict: Configuration parameters
        """
        return {
            'max_iterations': self.max_iterations,
            'emergency_movement_cost': self.emergency_movement_cost,
            'regular_movement_cost': self.regular_movement_cost,
            'waiting_cost': self.waiting_cost,
            'emergency_priority_bonus': self.emergency_priority_bonus,
            'grid_width': self.grid_system.width,
            'grid_height': self.grid_system.height,
            'max_time': self.grid_system.max_time
        }