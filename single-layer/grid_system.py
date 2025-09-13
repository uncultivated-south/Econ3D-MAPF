"""
Grid System Module for UrbanAirspaceSim
Handles 3D airspace representation and dynamic obstacle management
REFACTORED VERSION - Addresses identified bugs and design issues
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict
from contextlib import contextmanager
import threading
import copy

class CellState(Enum):
    """Represents the state of a grid cell"""
    FREE = 0
    STATIC_OBSTACLE = 1
    EMERGENCY_OCCUPIED = 2  # Emergency agent paths (highest priority)
    RESERVED = 3  # Reserved by auction winners or CBS assignments
    
class AgentType(Enum):
    """Types of agents in the system"""
    EMERGENCY = "emergency"
    NON_EMERGENCY = "non_emergency"

class PathStatus(Enum):
    """Status of agent paths"""
    NONE = "none"
    PLANNING = "planning"  # Path is being calculated
    ASSIGNED = "assigned"  # Path is assigned and active
    COMPLETED = "completed"  # Agent reached goal

@dataclass
class Position:
    """Represents a 3D position in the grid (x, y, time)"""
    x: int
    y: int
    t: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.t))
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y and self.t == other.t
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple format (x, y, t)"""
        return (self.x, self.y, self.t)
    
    def spatial_position(self) -> Tuple[int, int]:
        """Get spatial position (x, y) without time"""
        return (self.x, self.y)
    
    def manhattan_distance_to(self, other: 'Position') -> int:
        """Calculate Manhattan distance to another position (ignoring time)"""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def is_adjacent_to(self, other: 'Position') -> bool:
        """Check if this position is spatially adjacent to another"""
        return self.manhattan_distance_to(other) <= 1
    
    def __str__(self):
        return f"({self.x}, {self.y}, {self.t})"
    
    def __repr__(self):
        return f"Position({self.x}, {self.y}, {self.t})"

@dataclass
class Agent:
    """Represents an agent in the system"""
    id: int
    agent_type: AgentType
    start: Tuple[int, int]  # (x, y) starting position
    goal: Tuple[int, int]   # (x, y) goal position
    budget: float = 0.0     # Budget for non-emergency agents
    strategy: str = "balanced"  # "conservative", "aggressive", "balanced"
    path: List[Position] = None  # Assigned path
    path_status: PathStatus = PathStatus.NONE
    priority: int = 0  # Higher values = higher priority
    
    def __post_init__(self):
        if self.path is None:
            self.path = []
        # Set default priority based on agent type
        if self.priority == 0:
            self.priority = 100 if self.agent_type == AgentType.EMERGENCY else 1
    
    def get_start_position(self, start_time: int = 0) -> Position:
        """Get start position as Position object"""
        return Position(self.start[0], self.start[1], start_time)
    
    def get_goal_position(self) -> Tuple[int, int]:
        """Get goal position as tuple"""
        return self.goal
    
    def has_path(self) -> bool:
        """Check if agent has an assigned path"""
        return bool(self.path) and self.path_status == PathStatus.ASSIGNED
    
    def is_emergency(self) -> bool:
        """Check if this is an emergency agent"""
        return self.agent_type == AgentType.EMERGENCY

class GridSystemError(Exception):
    """Custom exception for grid system errors"""
    pass

class PathReservation:
    """Manages path reservations with automatic cleanup"""
    def __init__(self, grid_system: 'GridSystem', agent_id: int, path: List[Position], 
                 reservation_type: CellState = CellState.RESERVED):
        self.grid_system = grid_system
        self.agent_id = agent_id
        self.path = path.copy()
        self.reservation_type = reservation_type
        self.is_active = False
    
    def __enter__(self):
        """Reserve the path"""
        if self._can_reserve():
            self._reserve()
            self.is_active = True
            return self
        else:
            raise GridSystemError(f"Cannot reserve path for agent {self.agent_id}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the path reservation"""
        if self.is_active:
            self._release()
            self.is_active = False
    
    def _can_reserve(self) -> bool:
        """Check if path can be reserved"""
        for pos in self.path:
            if not self.grid_system._is_position_reservable(pos, self.agent_id):
                return False
        return True
    
    def _reserve(self):
        """Reserve all positions in the path"""
        for pos in self.path:
            self.grid_system._reserve_position(pos, self.agent_id, self.reservation_type)
    
    def _release(self):
        """Release all positions in the path"""
        for pos in self.path:
            self.grid_system._release_position(pos, self.agent_id)

class GridSystem:
    """
    Main grid system class for managing 3D airspace with improved error handling and consistency
    """
    
    def __init__(self, width: int, height: int, max_time: int = 100):
        """
        Initialize the grid system
        
        Args:
            width: Grid width (x dimension)
            height: Grid height (y dimension)  
            max_time: Maximum time steps to consider
        """
        if width <= 0 or height <= 0 or max_time <= 0:
            raise ValueError("Grid dimensions must be positive")
        
        self.width = width
        self.height = height
        self.max_time = max_time
        
        # 3D grid: [x][y][t] -> CellState
        self.grid = np.full((width, height, max_time), CellState.FREE, dtype=object)
        
        # Track occupancy by agent and time with priority information
        self.occupancy_map: Dict[Position, int] = {}  # Position -> agent_id
        self.agent_priorities: Dict[int, int] = {}  # agent_id -> priority
        
        # Agent management with thread safety
        self.agents: Dict[int, Agent] = {}
        self._agent_lock = threading.RLock()
        
        # Path tracking with status
        self.agent_paths: Dict[int, List[Position]] = {}
        self.path_status: Dict[int, PathStatus] = {}
        
        # Static obstacles
        self.static_obstacles: Set[Tuple[int, int]] = set()
        
        # Conflict tracking for analysis
        self.conflict_history: List[Dict] = []
        self.conflict_density: Dict[Tuple[int, int], int] = defaultdict(int)
        
    def is_valid_position(self, x: int, y: int, t: int = 0) -> bool:
        """Check if position is within grid bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= t < self.max_time)
    
    def is_valid_spatial_position(self, x: int, y: int) -> bool:
        """Check if spatial position (x, y) is within grid bounds"""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def add_static_obstacle(self, x: int, y: int) -> bool:
        """
        Add a static obstacle at spatial position (x, y) for all time steps
        
        Args:
            x, y: Spatial coordinates
            
        Returns:
            True if obstacle was added successfully
        """
        if not self.is_valid_spatial_position(x, y):
            return False
        
        self.static_obstacles.add((x, y))
        
        # Mark all time steps as obstacles
        for t in range(self.max_time):
            if self.is_valid_position(x, y, t):
                self.grid[x, y, t] = CellState.STATIC_OBSTACLE
        
        return True
    
    def add_static_obstacle(self, x: int, y: int) -> bool:
        """
        Add a static obstacle at spatial position (x, y) for all time steps
        
        Args:
            x, y: Spatial coordinates
            
        Returns:
            True if obstacle was added successfully
        """
        if not self.is_valid_spatial_position(x, y):
            return False
        
        self.static_obstacles.add((x, y))
        
        # Mark all time steps as obstacles
        for t in range(self.max_time):
            if self.is_valid_position(x, y, t):
                self.grid[x, y, t] = CellState.STATIC_OBSTACLE
        
        return True
    
    def remove_static_obstacle(self, x: int, y: int) -> bool:
        """Remove a static obstacle at spatial position (x, y)"""
        if (x, y) not in self.static_obstacles:
            return False
        
        self.static_obstacles.remove((x, y))
        
        # Clear obstacle from all time steps (if not occupied by agents)
        for t in range(self.max_time):
            pos = Position(x, y, t)
            if pos not in self.occupancy_map:
                self.grid[x, y, t] = CellState.FREE
        
        return True
    
    def _is_position_reservable(self, pos: Position, requesting_agent_id: int) -> bool:
        """
        Check if a position can be reserved by the requesting agent
        
        Args:
            pos: Position to check
            requesting_agent_id: ID of agent requesting reservation
            
        Returns:
            True if position can be reserved
        """
        if not self.is_valid_position(pos.x, pos.y, pos.t):
            return False
        
        # Check static obstacles
        if (pos.x, pos.y) in self.static_obstacles:
            return False
        
        # Check if position is already occupied
        if pos in self.occupancy_map:
            occupying_agent_id = self.occupancy_map[pos]
            
            # Agent can always reserve its own positions
            if occupying_agent_id == requesting_agent_id:
                return True
            
            # Check priority - higher priority agents can't be displaced
            requesting_priority = self.agent_priorities.get(requesting_agent_id, 1)
            occupying_priority = self.agent_priorities.get(occupying_agent_id, 1)
            
            return requesting_priority > occupying_priority
        
        return True
    
    def _reserve_position(self, pos: Position, agent_id: int, 
                         reservation_type: CellState = CellState.RESERVED):
        """
        Reserve a position for an agent
        
        Args:
            pos: Position to reserve
            agent_id: Agent reserving the position
            reservation_type: Type of reservation
        """
        if not self.is_valid_position(pos.x, pos.y, pos.t):
            raise GridSystemError(f"Invalid position {pos}")
        
        # Update occupancy
        self.occupancy_map[pos] = agent_id
        self.grid[pos.x, pos.y, pos.t] = reservation_type
    
    def _release_position(self, pos: Position, agent_id: int):
        """Release a position reservation"""
        if pos in self.occupancy_map and self.occupancy_map[pos] == agent_id:
            del self.occupancy_map[pos]
            
            # Reset to appropriate state
            if (pos.x, pos.y) in self.static_obstacles:
                self.grid[pos.x, pos.y, pos.t] = CellState.STATIC_OBSTACLE
            else:
                self.grid[pos.x, pos.y, pos.t] = CellState.FREE
    
    def is_cell_free(self, pos: Position, ignore_agent_id: Optional[int] = None,
                    check_priority: bool = True) -> bool:
        """
        Check if a cell is free for use by an agent
        
        Args:
            pos: Position to check
            ignore_agent_id: Agent ID to ignore when checking
            check_priority: Whether to consider agent priorities
            
        Returns:
            True if cell is available for use
        """
        if not self.is_valid_position(pos.x, pos.y, pos.t):
            return False
        
        # Check static obstacles
        if (pos.x, pos.y) in self.static_obstacles:
            return False
        
        # Check occupancy
        if pos in self.occupancy_map:
            occupying_agent_id = self.occupancy_map[pos]
            
            # Ignore specified agent
            if ignore_agent_id is not None and occupying_agent_id == ignore_agent_id:
                return True
            
            # If priority checking is enabled and we have an agent to compare
            if check_priority and ignore_agent_id is not None:
                requesting_priority = self.agent_priorities.get(ignore_agent_id, 1)
                occupying_priority = self.agent_priorities.get(occupying_agent_id, 1)
                
                # Higher priority agent can use the cell
                return requesting_priority > occupying_priority
            
            return False
        
        return True
    
    @contextmanager
    def agent_transaction(self):
        """Context manager for atomic agent operations"""
        with self._agent_lock:
            yield
    
    def add_agent(self, agent: Agent) -> bool:
        """
        Add an agent to the system with validation
        
        Args:
            agent: Agent to add
            
        Returns:
            True if agent was added successfully
        """
        if not isinstance(agent, Agent):
            raise TypeError("Expected Agent instance")
        
        with self.agent_transaction():
            if agent.id in self.agents:
                return False
            
            # Validate start and goal positions
            if not (self.is_valid_spatial_position(agent.start[0], agent.start[1]) and
                    self.is_valid_spatial_position(agent.goal[0], agent.goal[1])):
                return False
            
            # Add agent
            self.agents[agent.id] = agent
            self.agent_priorities[agent.id] = agent.priority
            self.path_status[agent.id] = agent.path_status
            
            # Initialize empty path tracking
            self.agent_paths[agent.id] = []
            
            return True
    
    def remove_agent(self, agent_id: int) -> bool:
        """Remove an agent and clean up all its reservations"""
        with self.agent_transaction():
            if agent_id not in self.agents:
                return False
            
            # Clear agent's path reservations
            self.clear_agent_path(agent_id)
            
            # Remove from tracking structures
            del self.agents[agent_id]
            del self.agent_priorities[agent_id]
            del self.path_status[agent_id]
            
            if agent_id in self.agent_paths:
                del self.agent_paths[agent_id]
            
            return True
    
    def set_agent_path(self, agent_id: int, path: List[Position], 
                      validate_before_commit: bool = True) -> bool:
        """
        Set a path for an agent with atomic transaction semantics
        
        Args:
            agent_id: Agent ID
            path: List of positions representing the path
            validate_before_commit: Whether to validate entire path before committing
            
        Returns:
            True if path was set successfully
        """
        with self.agent_transaction():
            if agent_id not in self.agents:
                return False
            
            if not path:
                return self.clear_agent_path(agent_id)
            
            # Validate path continuity
            if not self.validate_path_continuity(path):
                return False
            
            agent = self.agents[agent_id]
            
            # Determine reservation type based on agent type
            reservation_type = (CellState.EMERGENCY_OCCUPIED if agent.is_emergency() 
                              else CellState.RESERVED)
            
            # Try to reserve the new path
            try:
                with PathReservation(self, agent_id, path, reservation_type) as reservation:
                    # If we get here, reservation was successful
                    # Clear old path first
                    self.clear_agent_path(agent_id)
                    
                    # Commit new path
                    self.agent_paths[agent_id] = path.copy()
                    self.agents[agent_id].path = path.copy()
                    self.agents[agent_id].path_status = PathStatus.ASSIGNED
                    self.path_status[agent_id] = PathStatus.ASSIGNED
                    
                    # The context manager will handle the actual reservation
                    reservation.is_active = False  # Prevent auto-cleanup
                    
                    # Manually reserve positions
                    for pos in path:
                        self._reserve_position(pos, agent_id, reservation_type)
                    
                    return True
                    
            except GridSystemError:
                return False
    
    def clear_agent_path(self, agent_id: int) -> bool:
        """Clear an agent's path and remove all its reservations"""
        with self.agent_transaction():
            if agent_id not in self.agents:
                return False
            
            # Release all positions occupied by this agent
            if agent_id in self.agent_paths:
                for pos in self.agent_paths[agent_id]:
                    self._release_position(pos, agent_id)
                
                # Clear path records
                del self.agent_paths[agent_id]
            
            # Update agent state
            self.agents[agent_id].path = []
            self.agents[agent_id].path_status = PathStatus.NONE
            self.path_status[agent_id] = PathStatus.NONE
            
            return True
    
    def agent_reached_goal(self, agent_id: int) -> bool:
        """
        Mark that an agent has reached its goal and release its future path
        """
        with self.agent_transaction():
            if agent_id not in self.agents:
                return False
            
            # Update status
            self.agents[agent_id].path_status = PathStatus.COMPLETED
            self.path_status[agent_id] = PathStatus.COMPLETED
            
            # For emergency agents, release the path to allow others to use the space
            if agent_id in self.agents and self.agents[agent_id].is_emergency():
                self.clear_agent_path(agent_id)
            
            return True
    
    def get_neighbors(self, pos: Position, include_wait: bool = True) -> List[Position]:
        """
        Get valid neighboring positions with optional wait action
        
        Args:
            pos: Current position
            include_wait: Whether to include waiting in place as an option
            
        Returns:
            List of valid neighboring positions
        """
        neighbors = []
        
        # Movement directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        # Add wait action if requested
        if include_wait:
            directions.insert(0, (0, 0))
        
        for dx, dy in directions:
            new_x, new_y = pos.x + dx, pos.y + dy
            new_t = pos.t + 1
            
            if self.is_valid_position(new_x, new_y, new_t):
                neighbors.append(Position(new_x, new_y, new_t))
        
        return neighbors
    
    def validate_path_continuity(self, path: List[Position]) -> bool:
        """
        Validate that a path is continuous and follows movement rules
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid
        """
        if len(path) <= 1:
            return True
        
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            # Check time progression (must advance by exactly 1)
            if curr_pos.t != prev_pos.t + 1:
                return False
            
            # Check spatial movement (max 1 cell per time step, including waiting)
            dx = abs(curr_pos.x - prev_pos.x)
            dy = abs(curr_pos.y - prev_pos.y)
            
            if dx + dy > 1:  # Manhattan distance > 1
                return False
        
        return True
    
    def validate_path_conflicts(self, path: List[Position], agent_id: int,
                              check_emergency_paths: bool = True) -> List[Position]:
        """
        Find conflicting positions in a path
        
        Args:
            path: Path to validate
            agent_id: ID of agent owning the path
            check_emergency_paths: Whether to check conflicts with emergency paths
            
        Returns:
            List of positions that have conflicts
        """
        conflicts = []
        
        for pos in path:
            if not self.is_cell_free(pos, ignore_agent_id=agent_id):
                # Check if this is a priority conflict we should report
                if pos in self.occupancy_map:
                    occupying_agent = self.occupancy_map[pos]
                    occupying_priority = self.agent_priorities.get(occupying_agent, 1)
                    requesting_priority = self.agent_priorities.get(agent_id, 1)
                    
                    # Report conflict if occupying agent has equal or higher priority
                    if occupying_priority >= requesting_priority:
                        conflicts.append(pos)
                else:
                    # Static obstacle or other blocking condition
                    conflicts.append(pos)
        
        return conflicts
    
    def get_occupancy_at_time(self, t: int) -> Dict[Tuple[int, int], int]:
        """
        Get all agent positions at a specific time
        
        Args:
            t: Time step
            
        Returns:
            Dictionary mapping (x, y) to agent_id
        """
        occupancy = {}
        
        for pos, agent_id in self.occupancy_map.items():
            if pos.t == t:
                occupancy[(pos.x, pos.y)] = agent_id
        
        return occupancy
    
    def get_grid_state_at_time(self, t: int) -> np.ndarray:
        """
        Get 2D grid state at specific time
        
        Args:
            t: Time step
            
        Returns:
            2D numpy array representing grid state at time t
        """
        if not self.is_valid_position(0, 0, t):
            raise ValueError(f"Invalid time {t}")
        
        return self.grid[:, :, t].copy()
    
    def update_conflict_density(self, conflicts: Dict[Tuple[int, int], int]):
        """
        Update conflict density map for analytics and pricing
        
        Args:
            conflicts: Dictionary mapping (x, y) to conflict count
        """
        self.conflict_density.update(conflicts)
        
        # Record in history for analysis
        self.conflict_history.append({
            'timestamp': len(self.conflict_history),
            'conflicts': conflicts.copy(),
            'total_conflicts': sum(conflicts.values())
        })
    
    def get_conflict_density(self, x: int, y: int) -> int:
        """Get conflict density for a specific grid cell"""
        return self.conflict_density.get((x, y), 0)
    
    def get_path_cost(self, path: List[Position]) -> float:
        """
        Calculate path cost with time-based weighting
        
        Args:
            path: Path to calculate cost for
            
        Returns:
            Total path cost
        """
        if not path:
            return float('inf')
        
        # Simple time-based cost (can be extended)
        return len(path)
    
    def get_system_state(self) -> Dict:
        """Get comprehensive system state for monitoring"""
        with self.agent_transaction():
            emergency_agents = [a for a in self.agents.values() if a.is_emergency()]
            non_emergency_agents = [a for a in self.agents.values() if not a.is_emergency()]
            
            return {
                'grid_size': (self.width, self.height, self.max_time),
                'total_agents': len(self.agents),
                'emergency_agents': len(emergency_agents),
                'non_emergency_agents': len(non_emergency_agents),
                'agents_with_paths': len([a for a in self.agents.values() if a.has_path()]),
                'static_obstacles': len(self.static_obstacles),
                'total_occupancy': len(self.occupancy_map),
                'conflict_density_sum': sum(self.conflict_density.values()),
                'conflict_history_length': len(self.conflict_history)
            }
    
    def reset_system(self, keep_agents: bool = True, keep_static_obstacles: bool = True):
        """
        Reset system to initial state with options to preserve certain elements
        
        Args:
            keep_agents: Whether to keep agents (but clear their paths)
            keep_static_obstacles: Whether to keep static obstacles
        """
        with self.agent_transaction():
            # Clear all occupancy
            self.occupancy_map.clear()
            
            # Reset grid state
            self.grid.fill(CellState.FREE)
            
            # Re-add static obstacles if keeping them
            if keep_static_obstacles:
                static_obstacles_copy = self.static_obstacles.copy()
                self.static_obstacles.clear()
                for x, y in static_obstacles_copy:
                    self.add_static_obstacle(x, y)
            else:
                self.static_obstacles.clear()
            
            # Clear or reset agent paths
            if keep_agents:
                for agent_id in self.agents:
                    self.agents[agent_id].path = []
                    self.agents[agent_id].path_status = PathStatus.NONE
                    self.path_status[agent_id] = PathStatus.NONE
                self.agent_paths.clear()
            else:
                self.agents.clear()
                self.agent_priorities.clear()
                self.agent_paths.clear()
                self.path_status.clear()
            
            # Clear analytics
            self.conflict_density.clear()
            self.conflict_history.clear()
    
    # Utility methods for module integration
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Get all agents of specified type"""
        with self.agent_transaction():
            return [agent for agent in self.agents.values() 
                    if agent.agent_type == agent_type]
    
    def get_agents_by_status(self, status: PathStatus) -> List[Agent]:
        """Get all agents with specified path status"""
        with self.agent_transaction():
            return [agent for agent in self.agents.values() 
                    if agent.path_status == status]
    
    def get_unassigned_agents(self, agent_type: Optional[AgentType] = None) -> List[Agent]:
        """Get agents without assigned paths"""
        with self.agent_transaction():
            agents = self.agents.values()
            if agent_type:
                agents = [a for a in agents if a.agent_type == agent_type]
            return [agent for agent in agents if not agent.has_path()]
    
    def get_emergency_paths(self) -> Dict[int, List[Position]]:
        """Get all emergency agent paths for pathfinding avoidance"""
        with self.agent_transaction():
            emergency_paths = {}
            for agent_id, agent in self.agents.items():
                if agent.is_emergency() and agent.has_path():
                    emergency_paths[agent_id] = agent.path.copy()
            return emergency_paths
    
    def export_paths_for_cbs(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Export agent paths in format suitable for CBS module"""
        with self.agent_transaction():
            return {agent_id: [pos.to_tuple() for pos in path] 
                    for agent_id, path in self.agent_paths.items() if path}
    
    def export_agent_configs_for_auction(self) -> List[Dict]:
        """Export agent configurations for auction module"""
        with self.agent_transaction():
            return [
                {
                    'id': agent.id,
                    'start': agent.start,
                    'goal': agent.goal,
                    'budget': agent.budget,
                    'strategy': agent.strategy,
                    'priority': agent.priority
                }
                for agent in self.agents.values() 
                if agent.agent_type == AgentType.NON_EMERGENCY and not agent.has_path()
            ]
    
    def import_paths_from_external(self, paths: Dict[int, List[Tuple[int, int, int]]], 
                                  validate: bool = True) -> Dict[int, bool]:
        """
        Import paths from external modules (CBS, auction system)
        
        Args:
            paths: Dictionary mapping agent_id to path tuples
            validate: Whether to validate paths before importing
            
        Returns:
            Dictionary mapping agent_id to success status
        """
        results = {}
        
        for agent_id, path_tuples in paths.items():
            # Convert tuples to Position objects
            try:
                path = [Position(x, y, t) for x, y, t in path_tuples]
                
                if validate:
                    # Validate path before setting
                    if not self.validate_path_continuity(path):
                        results[agent_id] = False
                        continue
                    
                    # Check for conflicts
                    conflicts = self.validate_path_conflicts(path, agent_id)
                    if conflicts:
                        results[agent_id] = False
                        continue
                
                # Set the path
                success = self.set_agent_path(agent_id, path)
                results[agent_id] = success
                
            except (ValueError, TypeError) as e:
                results[agent_id] = False
        
        return results