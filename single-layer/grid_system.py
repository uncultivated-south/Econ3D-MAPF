"""
Grid System Module for UrbanAirspaceSim
Handles 3D airspace representation and dynamic obstacle management
"""

from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

class CellState(Enum):
    """Represents the state of a grid cell"""
    FREE = 0
    STATIC_OBSTACLE = 1
    DYNAMIC_OBSTACLE = 2  # Emergency agent paths
    
class AgentType(Enum):
    """Types of agents in the system"""
    EMERGENCY = "emergency"
    NON_EMERGENCY = "non_emergency"

@dataclass
class Position:
    """Represents a 3D position in the grid (x, y, time)"""
    x: int
    y: int
    t: int
    
    def __hash__(self):
        return hash((self.x, self.y, self.t))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.t == other.t
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple format (x, y, t)"""
        return (self.x, self.y, self.t)

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
    
    def __post_init__(self):
        if self.path is None:
            self.path = []

class GridSystem:
    """
    Main grid system class for managing 3D airspace
    """
    
    def __init__(self, width: int, height: int, max_time: int = 100):
        """
        Initialize the grid system
        
        Args:
            width: Grid width (x dimension)
            height: Grid height (y dimension)  
            max_time: Maximum time steps to consider
        """
        self.width = width
        self.height = height
        self.max_time = max_time
        
        # 3D grid: [x][y][t] -> CellState
        self.grid = np.full((width, height, max_time), CellState.FREE, dtype=object)
        
        # Track dynamic obstacles (emergency agent paths) with their release times
        self.dynamic_obstacles: Dict[Position, int] = {}  # Position -> agent_id
        
        # Conflict density tracking for CBS analysis
        self.conflict_density: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Agent management
        self.agents: Dict[int, Agent] = {}
        self.emergency_agents: List[int] = []
        self.non_emergency_agents: List[int] = []
        
        # Path tracking
        self.agent_paths: Dict[int, List[Position]] = {}
        
    def is_valid_position(self, x: int, y: int, t: int = 0) -> bool:
        """Check if position is within grid bounds"""
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                0 <= t < self.max_time)
    
    def is_cell_free(self, pos: Position, ignore_agent_id: Optional[int] = None) -> bool:
        """
        Check if a cell is free at given time
        
        Args:
            pos: Position to check
            ignore_agent_id: Agent ID to ignore when checking (for replanning)
        """
        if not self.is_valid_position(pos.x, pos.y, pos.t):
            return False
            
        # Check static obstacles
        if self.grid[pos.x, pos.y, pos.t] == CellState.STATIC_OBSTACLE:
            return False
            
        # Check dynamic obstacles (emergency paths)
        if pos in self.dynamic_obstacles:
            if ignore_agent_id is None or self.dynamic_obstacles[pos] != ignore_agent_id:
                return False
                
        return True
    
    def add_agent(self, agent: Agent) -> bool:
        """
        Add an agent to the system
        
        Returns:
            True if agent was added successfully
        """
        if agent.id in self.agents:
            return False
            
        # Validate start and goal positions
        if not (self.is_valid_position(agent.start[0], agent.start[1]) and
                self.is_valid_position(agent.goal[0], agent.goal[1])):
            return False
            
        self.agents[agent.id] = agent
        
        if agent.agent_type == AgentType.EMERGENCY:
            self.emergency_agents.append(agent.id)
        else:
            self.non_emergency_agents.append(agent.id)
            
        return True
    
    def remove_agent(self, agent_id: int) -> bool:
        """Remove an agent and clear its path"""
        if agent_id not in self.agents:
            return False
            
        # Clear agent's path from dynamic obstacles
        self.clear_agent_path(agent_id)
        
        # Remove from agent lists
        if agent_id in self.emergency_agents:
            self.emergency_agents.remove(agent_id)
        if agent_id in self.non_emergency_agents:
            self.non_emergency_agents.remove(agent_id)
            
        # Remove from main agent dict
        del self.agents[agent_id]
        
        return True
    
    def set_agent_path(self, agent_id: int, path: List[Position]) -> bool:
        """
        Set a path for an agent and update dynamic obstacles
        
        Args:
            agent_id: Agent ID
            path: List of positions representing the path
            
        Returns:
            True if path was set successfully
        """
        if agent_id not in self.agents:
            return False
            
        # Clear existing path
        self.clear_agent_path(agent_id)
        
        # Validate new path
        for pos in path:
            if not self.is_valid_position(pos.x, pos.y, pos.t):
                return False
                
        # Set new path
        self.agent_paths[agent_id] = path.copy()
        self.agents[agent_id].path = path.copy()
        
        # Add to dynamic obstacles if emergency agent
        if self.agents[agent_id].agent_type == AgentType.EMERGENCY:
            for pos in path:
                self.dynamic_obstacles[pos] = agent_id
                
        return True
    
    def clear_agent_path(self, agent_id: int):
        """Clear an agent's path and remove from dynamic obstacles"""
        if agent_id in self.agent_paths:
            # Remove from dynamic obstacles
            for pos in self.agent_paths[agent_id]:
                if pos in self.dynamic_obstacles and self.dynamic_obstacles[pos] == agent_id:
                    del self.dynamic_obstacles[pos]
                    
            # Clear path records
            del self.agent_paths[agent_id]
            if agent_id in self.agents:
                self.agents[agent_id].path = []
    
    def agent_reached_goal(self, agent_id: int) -> bool:
        """
        Mark that an agent has reached its goal and release its path
        This is called when emergency agents complete their missions
        """
        if agent_id not in self.agents:
            return False
            
        self.clear_agent_path(agent_id)
        return True
    
    def get_neighbors(self, pos: Position) -> List[Position]:
        """
        Get valid neighboring positions (including waiting in place)
        Standard 4-connectivity + wait action
        """
        neighbors = []
        
        # Movement directions: stay, up, down, left, right
        directions = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dx, dy in directions:
            new_x, new_y = pos.x + dx, pos.y + dy
            new_t = pos.t + 1
            
            if self.is_valid_position(new_x, new_y, new_t):
                neighbors.append(Position(new_x, new_y, new_t))
                
        return neighbors
    
    def update_conflict_density(self, conflicts: Dict[Tuple[int, int], int]):
        """
        Update conflict density map based on CBS analysis
        
        Args:
            conflicts: Dictionary mapping (x, y) to conflict count
        """
        self.conflict_density = conflicts.copy()
    
    def get_conflict_density(self, x: int, y: int) -> int:
        """Get conflict density for a specific grid cell"""
        return self.conflict_density.get((x, y), 0)
    
    def get_grid_state_at_time(self, t: int) -> np.ndarray:
        """
        Get 2D grid state at specific time
        
        Returns:
            2D numpy array representing grid state at time t
        """
        if not self.is_valid_position(0, 0, t):
            return None
            
        state = np.full((self.width, self.height), CellState.FREE, dtype=object)
        
        # Add static obstacles
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y, t] == CellState.STATIC_OBSTACLE:
                    state[x, y] = CellState.STATIC_OBSTACLE
                    
        # Add dynamic obstacles
        for pos, agent_id in self.dynamic_obstacles.items():
            if pos.t == t:
                state[pos.x, pos.y] = CellState.DYNAMIC_OBSTACLE
                
        return state
    
    def get_path_cost(self, path: List[Position]) -> int:
        """Calculate path cost (time-based)"""
        if not path:
            return 0
        return len(path)
    
    def validate_path_continuity(self, path: List[Position]) -> bool:
        """
        Validate that a path is continuous (each step is reachable from previous)
        """
        if len(path) <= 1:
            return True
            
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            # Check time progression
            if curr_pos.t != prev_pos.t + 1:
                return False
                
            # Check spatial movement (max 1 cell per time step)
            dx = abs(curr_pos.x - prev_pos.x)
            dy = abs(curr_pos.y - prev_pos.y)
            
            if dx + dy > 1:  # Manhattan distance > 1
                return False
                
        return True
    
    def get_system_state(self) -> Dict:
        """
        Get current system state for monitoring/debugging
        """
        return {
            'grid_size': (self.width, self.height, self.max_time),
            'total_agents': len(self.agents),
            'emergency_agents': len(self.emergency_agents),
            'non_emergency_agents': len(self.non_emergency_agents),
            'dynamic_obstacles': len(self.dynamic_obstacles),
            'conflict_density_sum': sum(self.conflict_density.values()),
            'agents_with_paths': len(self.agent_paths)
        }
    
    def reset_system(self):
        """Reset system to initial state (keep agents but clear paths)"""
        self.dynamic_obstacles.clear()
        self.conflict_density.clear()
        self.agent_paths.clear()
        
        for agent in self.agents.values():
            agent.path = []
    
    # Utility methods for interfacing with other modules
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Get all agents of specified type"""
        return [agent for agent in self.agents.values() 
                if agent.agent_type == agent_type]
    
    def get_unassigned_agents(self, agent_type: Optional[AgentType] = None) -> List[Agent]:
        """Get agents without assigned paths"""
        agents = self.agents.values()
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        return [agent for agent in agents if not agent.path]
    
    def export_paths_for_cbs(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Export agent paths in format suitable for CBS module"""
        return {agent_id: [pos.to_tuple() for pos in path] 
                for agent_id, path in self.agent_paths.items()}
    
    def export_agent_configs_for_auction(self) -> List[Dict]:
        """Export agent configurations for auction module"""
        return [
            {
                'id': agent.id,
                'start': agent.start,
                'goal': agent.goal,
                'budget': agent.budget,
                'strategy': agent.strategy
            }
            for agent in self.agents.values() 
            if agent.agent_type == AgentType.NON_EMERGENCY and not agent.path
        ]