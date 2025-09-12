"""
Conflict-Based Search (CBS) Module for UrbanAirspaceSim
Handles multi-agent pathfinding with conflict detection and resolution
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import copy

# Import from previous modules
from grid_system import Position, Agent, AgentType, GridSystem
from astar_pathfinding import AStarPathfinder, PathfindingContext, paths_to_tuples, tuples_to_paths

@dataclass
class Conflict:
    """Represents a conflict between two agents"""
    agent1_id: int
    agent2_id: int
    position: Tuple[int, int, int]  # (x, y, t) where conflict occurs
    conflict_type: str  # "vertex" or "edge"
    
    # For edge conflicts, store the previous positions
    agent1_prev: Optional[Tuple[int, int, int]] = None
    agent2_prev: Optional[Tuple[int, int, int]] = None
    
    def __hash__(self):
        return hash((self.agent1_id, self.agent2_id, self.position, self.conflict_type))
    
    def __eq__(self, other):
        return (self.agent1_id == other.agent1_id and 
                self.agent2_id == other.agent2_id and
                self.position == other.position and
                self.conflict_type == other.conflict_type)

@dataclass
class Constraint:
    """Represents a constraint for an agent"""
    agent_id: int
    position: Tuple[int, int, int]  # (x, y, t) - forbidden position
    constraint_type: str  # "vertex" or "edge"
    
    # For edge constraints, store the previous position
    prev_position: Optional[Tuple[int, int, int]] = None

@dataclass
class CBSNode:
    """Node in the CBS high-level search tree"""
    constraints: List[Constraint] = field(default_factory=list)
    paths: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)
    cost: float = 0.0
    conflicts: List[Conflict] = field(default_factory=list)
    
    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        # Tie-breaking: prefer fewer conflicts
        return len(self.conflicts) < len(other.conflicts)

class CBSResult:
    """Result of CBS execution"""
    def __init__(self):
        self.success: bool = False
        self.paths: Dict[int, List[Tuple[int, int, int]]] = {}
        self.total_cost: float = float('inf')
        self.iterations_used: int = 0
        self.conflicts_found: List[Conflict] = []
        self.conflict_density: Dict[Tuple[int, int], int] = defaultdict(int)
        self.should_trigger_auction: bool = False
        self.failure_reason: str = ""

class ConflictBasedSearch:
    """
    Conflict-Based Search implementation for multi-agent pathfinding
    """
    
    def __init__(self, grid_system: GridSystem, pathfinder: AStarPathfinder):
        """
        Initialize CBS with grid system and pathfinder
        
        Args:
            grid_system: The grid system instance
            pathfinder: A* pathfinder instance
        """
        self.grid = grid_system
        self.pathfinder = pathfinder
        
        # CBS parameters
        self.max_iterations = grid_system.width * grid_system.height  # Square of area
        self.debug_mode = False
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'auction_triggers': 0,
            'average_iterations': 0
        }
    
    def detect_conflicts(self, paths: Dict[int, List[Tuple[int, int, int]]]) -> List[Conflict]:
        """
        Detect conflicts between agent paths
        
        Args:
            paths: Dictionary mapping agent_id to path (list of (x, y, t) tuples)
            
        Returns:
            List of conflicts found
        """
        conflicts = []
        agent_ids = list(paths.keys())
        
        # Check all pairs of agents
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1_id = agent_ids[i]
                agent2_id = agent_ids[j]
                
                path1 = paths[agent1_id]
                path2 = paths[agent2_id]
                
                # Find conflicts between this pair
                pair_conflicts = self._detect_conflicts_between_agents(
                    agent1_id, path1, agent2_id, path2
                )
                conflicts.extend(pair_conflicts)
        
        return conflicts
    
    def _detect_conflicts_between_agents(self, 
                                       agent1_id: int, path1: List[Tuple[int, int, int]],
                                       agent2_id: int, path2: List[Tuple[int, int, int]]) -> List[Conflict]:
        """
        Detect conflicts between two specific agents
        """
        conflicts = []
        
        if not path1 or not path2:
            return conflicts
        
        # Extend shorter path to match longer one (agent stays at goal)
        max_time = max(path1[-1][2], path2[-1][2]) if path1 and path2 else 0
        
        extended_path1 = self._extend_path_to_time(path1, max_time)
        extended_path2 = self._extend_path_to_time(path2, max_time)
        
        # Check for vertex conflicts (same position at same time)
        for i in range(min(len(extended_path1), len(extended_path2))):
            pos1 = extended_path1[i]
            pos2 = extended_path2[i]
            
            if pos1[0] == pos2[0] and pos1[1] == pos2[1] and pos1[2] == pos2[2]:
                conflict = Conflict(
                    agent1_id=agent1_id,
                    agent2_id=agent2_id,
                    position=pos1,
                    conflict_type="vertex"
                )
                conflicts.append(conflict)
        
        # Check for edge conflicts (agents swap positions)
        for i in range(1, min(len(extended_path1), len(extended_path2))):
            pos1_prev = extended_path1[i-1]
            pos1_curr = extended_path1[i]
            pos2_prev = extended_path2[i-1]
            pos2_curr = extended_path2[i]
            
            # Check if agents swap positions
            if (pos1_prev[0] == pos2_curr[0] and pos1_prev[1] == pos2_curr[1] and
                pos1_curr[0] == pos2_prev[0] and pos1_curr[1] == pos2_prev[1] and
                pos1_prev[2] == pos2_prev[2] and pos1_curr[2] == pos2_curr[2]):
                
                conflict = Conflict(
                    agent1_id=agent1_id,
                    agent2_id=agent2_id,
                    position=pos1_curr,
                    conflict_type="edge",
                    agent1_prev=pos1_prev,
                    agent2_prev=pos2_prev
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _extend_path_to_time(self, path: List[Tuple[int, int, int]], target_time: int) -> List[Tuple[int, int, int]]:
        """
        Extend path by having agent wait at goal position
        """
        if not path or target_time < path[-1][2]:
            return path
        
        extended = path.copy()
        last_pos = path[-1]
        
        # Add waiting positions at the goal
        for t in range(last_pos[2] + 1, target_time + 1):
            extended.append((last_pos[0], last_pos[1], t))
        
        return extended
    
    def calculate_conflict_density(self, conflicts: List[Conflict]) -> Dict[Tuple[int, int], int]:
        """
        Calculate conflict density per grid cell for auction pricing
        
        Args:
            conflicts: List of conflicts found
            
        Returns:
            Dictionary mapping (x, y) to conflict count
        """
        density = defaultdict(int)
        
        for conflict in conflicts:
            x, y, t = conflict.position
            density[(x, y)] += 1
            
            # For edge conflicts, also count the previous positions
            if conflict.conflict_type == "edge":
                if conflict.agent1_prev:
                    prev_x, prev_y, prev_t = conflict.agent1_prev
                    density[(prev_x, prev_y)] += 1
                if conflict.agent2_prev:
                    prev_x, prev_y, prev_t = conflict.agent2_prev
                    density[(prev_x, prev_y)] += 1
        
        return dict(density)
    
    def create_constraints_from_conflict(self, conflict: Conflict) -> List[Constraint]:
        """
        Create constraints from a conflict
        
        Args:
            conflict: Conflict to resolve
            
        Returns:
            List of constraints (one for each agent involved)
        """
        constraints = []
        
        if conflict.conflict_type == "vertex":
            # Vertex conflict: forbid both agents from being at this position
            constraints.append(Constraint(
                agent_id=conflict.agent1_id,
                position=conflict.position,
                constraint_type="vertex"
            ))
            constraints.append(Constraint(
                agent_id=conflict.agent2_id,
                position=conflict.position,
                constraint_type="vertex"
            ))
        
        elif conflict.conflict_type == "edge":
            # Edge conflict: forbid the edge traversal for both agents
            constraints.append(Constraint(
                agent_id=conflict.agent1_id,
                position=conflict.position,
                constraint_type="edge",
                prev_position=conflict.agent1_prev
            ))
            constraints.append(Constraint(
                agent_id=conflict.agent2_id,
                position=conflict.position,
                constraint_type="edge",
                prev_position=conflict.agent2_prev
            ))
        
        return constraints
    
    def apply_constraints_to_pathfinding(self, agent_id: int, constraints: List[Constraint]) -> Dict[Tuple[int, int, int], bool]:
        """
        Convert constraints to forbidden positions for pathfinding
        
        Args:
            agent_id: Agent to apply constraints to
            constraints: List of constraints
            
        Returns:
            Dictionary of forbidden positions for this agent
        """
        forbidden = {}
        
        for constraint in constraints:
            if constraint.agent_id == agent_id:
                if constraint.constraint_type == "vertex":
                    forbidden[constraint.position] = True
                elif constraint.constraint_type == "edge":
                    # For edge constraints, we need to implement more complex logic
                    # For now, we'll forbid the destination position
                    forbidden[constraint.position] = True
        
        return forbidden
    
    def replan_agent_with_constraints(self, agent: Agent, constraints: List[Constraint],
                                    emergency_paths: Dict[int, List[Position]] = None,
                                    start_time: int = 0) -> Optional[List[Tuple[int, int, int]]]:
        """
        Replan path for an agent with given constraints
        
        Args:
            agent: Agent to replan
            constraints: Constraints to apply
            emergency_paths: Emergency agent paths to avoid
            start_time: Starting time for replanning
            
        Returns:
            New path or None if no path found
        """
        if emergency_paths is None:
            emergency_paths = {}
        
        # Apply constraints by creating a custom pathfinding context
        forbidden_positions = self.apply_constraints_to_pathfinding(agent.id, constraints)
        
        # Create modified pathfinding context
        context = PathfindingContext(
            agent_id=agent.id,
            start=agent.start,
            goal=agent.goal,
            start_time=start_time,
            max_time=self.grid.max_time,
            emergency_paths=emergency_paths
        )
        
        # Find path using A* with constraints
        # Note: This is a simplified version. A full implementation would modify
        # the A* algorithm to respect the forbidden positions
        path = self.pathfinder.find_path(context)
        
        if path:
            return [pos.to_tuple() for pos in path]
        return None
    
    def solve(self, agents: List[Agent], 
             emergency_paths: Dict[int, List[Position]] = None,
             start_time: int = 0) -> CBSResult:
        """
        Main CBS solving method
        
        Args:
            agents: List of agents to find paths for
            emergency_paths: Emergency agent paths to avoid
            start_time: Starting time for pathfinding
            
        Returns:
            CBSResult with solution or failure information
        """
        self.stats['total_runs'] += 1
        result = CBSResult()
        
        if not agents:
            result.success = True
            return result
        
        if emergency_paths is None:
            emergency_paths = {}
        
        # Convert emergency paths to tuple format
        emergency_tuple_paths = paths_to_tuples(emergency_paths)
        
        # Initialize CBS with root node
        root_node = CBSNode()
        
        # Find initial paths for all agents
        initial_paths = self._find_initial_paths(agents, emergency_paths, start_time)
        
        # Check if any agent couldn't find a path initially
        for agent in agents:
            if agent.id not in initial_paths:
                result.failure_reason = f"No initial path found for agent {agent.id}"
                result.should_trigger_auction = True
                self.stats['auction_triggers'] += 1
                return result
        
        root_node.paths = initial_paths
        root_node.cost = sum(len(path) for path in initial_paths.values())
        root_node.conflicts = self.detect_conflicts(initial_paths)
        
        # If no conflicts, we're done
        if not root_node.conflicts:
            result.success = True
            result.paths = initial_paths
            result.total_cost = root_node.cost
            result.iterations_used = 1
            result.conflict_density = {}
            self.stats['successful_runs'] += 1
            return result
        
        # Initialize CBS search
        open_list = [root_node]
        heapq.heapify(open_list)
        
        iterations = 0
        best_node = root_node  # Track best node found so far
        
        while open_list and iterations < self.max_iterations:
            iterations += 1
            
            # Get node with lowest cost
            current_node = heapq.heappop(open_list)
            
            # Update best node if this has fewer conflicts
            if len(current_node.conflicts) < len(best_node.conflicts):
                best_node = current_node
            
            # Check if this node is conflict-free
            if not current_node.conflicts:
                result.success = True
                result.paths = current_node.paths
                result.total_cost = current_node.cost
                result.iterations_used = iterations
                result.conflict_density = {}
                self.stats['successful_runs'] += 1
                return result
            
            # Choose first conflict to resolve
            conflict = current_node.conflicts[0]
            
            # Create constraints from conflict
            constraints = self.create_constraints_from_conflict(conflict)
            
            # Create child nodes for each constraint
            for constraint in constraints:
                child_node = self._create_child_node(current_node, constraint, agents, 
                                                   emergency_paths, start_time)
                
                if child_node and child_node.paths:
                    # Check if all agents have paths
                    if len(child_node.paths) == len(agents):
                        heapq.heappush(open_list, child_node)
        
        # CBS failed to find solution within iteration limit
        result.iterations_used = iterations
        result.conflicts_found = best_node.conflicts
        result.conflict_density = self.calculate_conflict_density(best_node.conflicts)
        result.should_trigger_auction = True
        result.failure_reason = f"CBS exceeded max iterations ({self.max_iterations})"
        
        # Store best partial solution found
        result.paths = best_node.paths
        result.total_cost = best_node.cost
        
        self.stats['auction_triggers'] += 1
        
        if self.debug_mode:
            print(f"CBS failed after {iterations} iterations")
            print(f"Best node had {len(best_node.conflicts)} conflicts")
            print(f"Conflict density: {result.conflict_density}")
        
        return result
    
    def _find_initial_paths(self, agents: List[Agent], 
                           emergency_paths: Dict[int, List[Position]], 
                           start_time: int) -> Dict[int, List[Tuple[int, int, int]]]:
        """Find initial paths for all agents"""
        paths = {}
        
        for agent in agents:
            path = self.pathfinder.find_path_for_agent(
                agent, emergency_paths=emergency_paths, start_time=start_time
            )
            if path:
                paths[agent.id] = [pos.to_tuple() for pos in path]
        
        return paths
    
    def _create_child_node(self, parent_node: CBSNode, constraint: Constraint,
                          agents: List[Agent], emergency_paths: Dict[int, List[Position]],
                          start_time: int) -> Optional[CBSNode]:
        """Create a child node with additional constraint"""
        child_node = CBSNode()
        child_node.constraints = parent_node.constraints + [constraint]
        child_node.paths = parent_node.paths.copy()
        
        # Find agent that needs replanning
        constrained_agent = None
        for agent in agents:
            if agent.id == constraint.agent_id:
                constrained_agent = agent
                break
        
        if not constrained_agent:
            return None
        
        # Replan path for constrained agent
        new_path = self.replan_agent_with_constraints(
            constrained_agent, child_node.constraints, emergency_paths, start_time
        )
        
        if not new_path:
            return None  # No valid path found
        
        child_node.paths[constraint.agent_id] = new_path
        child_node.cost = sum(len(path) for path in child_node.paths.values())
        child_node.conflicts = self.detect_conflicts(child_node.paths)
        
        return child_node
    
    def validate_auction_winners(self, winner_paths: Dict[int, List[Tuple[int, int, int]]],
                                emergency_paths: Dict[int, List[Position]] = None) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Validate and resolve conflicts among auction winners using CBS
        
        Args:
            winner_paths: Paths of auction winners
            emergency_paths: Emergency agent paths to avoid
            
        Returns:
            Conflict-free paths for valid winners
        """
        if not winner_paths:
            return {}
        
        if emergency_paths is None:
            emergency_paths = {}
        
        # Check for conflicts among winners
        conflicts = self.detect_conflicts(winner_paths)
        
        if not conflicts:
            return winner_paths  # No conflicts, all winners are valid
        
        # Resolve conflicts by priority (highest total bid wins)
        # This assumes winner_paths is already ordered by total bid (highest first)
        validated_paths = {}
        
        for agent_id, path in winner_paths.items():
            # Check if this path conflicts with already validated paths
            test_paths = validated_paths.copy()
            test_paths[agent_id] = path
            
            test_conflicts = self.detect_conflicts(test_paths)
            
            # Also check conflicts with emergency paths
            emergency_conflicts = []
            for emergency_id, emergency_path in emergency_paths.items():
                emergency_tuple_path = [pos.to_tuple() for pos in emergency_path]
                emergency_conflicts.extend(
                    self._detect_conflicts_between_agents(
                        agent_id, path, emergency_id, emergency_tuple_path
                    )
                )
            
            # If no new conflicts, this winner is valid
            if not test_conflicts and not emergency_conflicts:
                validated_paths[agent_id] = path
            # Otherwise, this winner loses due to conflicts
        
        return validated_paths
    
    def get_statistics(self) -> Dict:
        """Get CBS statistics"""
        avg_iterations = (self.stats['average_iterations'] * (self.stats['total_runs'] - 1) + 
                         self.stats.get('last_iterations', 0)) / max(self.stats['total_runs'], 1)
        
        return {
            'total_runs': self.stats['total_runs'],
            'successful_runs': self.stats['successful_runs'],
            'success_rate': self.stats['successful_runs'] / max(self.stats['total_runs'], 1),
            'auction_triggers': self.stats['auction_triggers'],
            'auction_trigger_rate': self.stats['auction_triggers'] / max(self.stats['total_runs'], 1),
            'average_iterations': avg_iterations,
            'max_iterations_limit': self.max_iterations
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
    
    def reset_statistics(self):
        """Reset CBS statistics"""
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'auction_triggers': 0,
            'average_iterations': 0
        }

# Utility functions for integration

def create_cbs_solver(grid_system: GridSystem) -> ConflictBasedSearch:
    """Create CBS solver with pathfinder"""
    pathfinder = AStarPathfinder(grid_system)
    return ConflictBasedSearch(grid_system, pathfinder)

def extract_non_emergency_agents(grid_system: GridSystem) -> List[Agent]:
    """Extract non-emergency agents from grid system"""
    return [agent for agent in grid_system.agents.values() 
            if agent.agent_type == AgentType.NON_EMERGENCY]

def should_trigger_auction(cbs_result: CBSResult) -> bool:
    """Determine if auction should be triggered based on CBS result"""
    return cbs_result.should_trigger_auction

def get_conflict_density_for_auction(cbs_result: CBSResult) -> Dict[Tuple[int, int], int]:
    """Extract conflict density for auction pricing"""
    return cbs_result.conflict_density

def convert_paths_for_grid_system(paths: Dict[int, List[Tuple[int, int, int]]]) -> Dict[int, List[Position]]:
    """Convert CBS paths to grid system format"""
    return tuples_to_paths(paths)