"""
Conflict-Based Search (CBS) Module for UrbanAirspaceSim
Handles multi-agent pathfinding with conflict detection and resolution
REFACTORED VERSION - Addresses identified bugs and integrates with improved modules
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import copy
import time
import threading
from enum import Enum

# Import from refactored modules
from grid_system import Position, Agent, AgentType, GridSystem
from astar_pathfinding import (
    AStarPathfinder, PathfindingConfig, PathfindingResult, 
    Constraint, ConstraintType, create_conservative_config
)

class ConflictType(Enum):
    """Types of conflicts between agents"""
    VERTEX = "vertex"     # Same position at same time
    EDGE = "edge"         # Agents swap positions
    FOLLOWING = "following"  # Agent follows too closely

@dataclass
class Conflict:
    """Represents a conflict between two agents"""
    conflict_type: ConflictType
    agent1_id: int
    agent2_id: int
    position: Position  # Where conflict occurs
    time: int          # When conflict occurs
    
    # For edge conflicts
    agent1_prev_position: Optional[Position] = None
    agent2_prev_position: Optional[Position] = None
    
    # For following conflicts
    min_separation: int = 1
    
    def __hash__(self):
        return hash((self.conflict_type, self.agent1_id, self.agent2_id, 
                    self.position, self.time))
    
    def __eq__(self, other):
        return (isinstance(other, Conflict) and
                self.conflict_type == other.conflict_type and
                self.agent1_id == other.agent1_id and 
                self.agent2_id == other.agent2_id and
                self.position == other.position and
                self.time == other.time)
    
    def get_involved_agents(self) -> Set[int]:
        """Get set of agent IDs involved in this conflict"""
        return {self.agent1_id, self.agent2_id}
    
    def involves_agent(self, agent_id: int) -> bool:
        """Check if conflict involves specific agent"""
        return agent_id in {self.agent1_id, self.agent2_id}

@dataclass
class CBSConstraint:
    """CBS-specific constraint derived from conflicts"""
    base_constraint: Constraint  # The actual pathfinding constraint
    source_conflict: Conflict   # Conflict that generated this constraint
    target_agent_id: int        # Agent this constraint applies to
    priority: int = 1          # Higher = more important constraint
    
    def __hash__(self):
        return hash((self.base_constraint, self.target_agent_id))

@dataclass 
class CBSNode:
    """Node in the CBS high-level search tree"""
    # Core CBS data
    constraints: List[CBSConstraint] = field(default_factory=list)
    paths: Dict[int, List[Position]] = field(default_factory=dict)
    solution_cost: float = 0.0
    
    # Conflict information
    conflicts: List[Conflict] = field(default_factory=list)
    unresolved_agents: Set[int] = field(default_factory=set)
    
    # Search metadata
    depth: int = 0
    parent: Optional['CBSNode'] = None
    creation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Calculate total cost from paths
        if self.paths and self.solution_cost == 0.0:
            self.solution_cost = sum(len(path) for path in self.paths.values())
    
    def __lt__(self, other):
        # Primary ordering: solution cost
        if abs(self.solution_cost - other.solution_cost) > 1e-9:
            return self.solution_cost < other.solution_cost
        
        # Secondary: fewer conflicts (more resolved)
        if len(self.conflicts) != len(other.conflicts):
            return len(self.conflicts) < len(other.conflicts)
        
        # Tertiary: fewer unresolved agents
        if len(self.unresolved_agents) != len(other.unresolved_agents):
            return len(self.unresolved_agents) < len(other.unresolved_agents)
        
        # Final: shallower depth (simpler solution)
        return self.depth < other.depth
    
    def is_solution(self) -> bool:
        """Check if this node represents a complete solution"""
        return len(self.conflicts) == 0 and len(self.unresolved_agents) == 0
    
    def get_constraint_count_for_agent(self, agent_id: int) -> int:
        """Get number of constraints affecting specific agent"""
        return sum(1 for c in self.constraints if c.target_agent_id == agent_id)

class CBSResult:
    """Result of CBS execution with comprehensive information"""
    
    def __init__(self):
        # Core results
        self.success: bool = False
        self.paths: Dict[int, List[Position]] = {}
        self.total_cost: float = float('inf')
        
        # Search statistics
        self.iterations_used: int = 0
        self.nodes_expanded: int = 0
        self.computation_time: float = 0.0
        
        # Conflict analysis
        self.conflicts_found: List[Conflict] = []
        self.conflicts_resolved: List[Conflict] = []
        self.final_conflicts: List[Conflict] = []
        self.conflict_density: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Solution quality
        self.solution_depth: int = 0
        self.constraints_generated: int = 0
        self.agents_replanned: Set[int] = set()
        
        # Integration flags
        self.should_trigger_auction: bool = False
        self.auction_candidates: List[int] = []  # Agents that couldn't be planned
        
        # Failure information
        self.failure_reason: str = ""
        self.partial_solution: Dict[int, List[Position]] = {}
        
        # Performance metrics
        self.pathfinding_calls: int = 0
        self.constraint_generation_time: float = 0.0
        self.conflict_detection_time: float = 0.0

class ConflictBasedSearch:
    """
    Enhanced Conflict-Based Search implementation with proper constraint handling
    """
    
    def __init__(self, grid_system: GridSystem, pathfinder: AStarPathfinder):
        """
        Initialize CBS with grid system and pathfinder
        
        Args:
            grid_system: The grid system instance
            pathfinder: A* pathfinder instance with constraint support
        """
        self.grid = grid_system
        self.pathfinder = pathfinder
        
        # CBS parameters
        self.max_iterations = 1000
        self.max_computation_time = 300.0  # 5 minutes
        self.max_conflicts_per_node = 100  # Prevent explosive conflict growth
        
        # Conflict detection parameters
        self.detect_edge_conflicts = True
        self.detect_following_conflicts = True
        self.min_agent_separation = 1
        
        # Performance optimization
        self.early_termination = True
        self.constraint_pruning = True
        self.conflict_prioritization = True
        
        # Thread safety
        self._search_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'average_iterations': 0.0,
            'average_computation_time': 0.0,
            'total_conflicts_detected': 0,
            'total_constraints_generated': 0,
            'auction_triggers': 0
        }
        
        self.debug_mode = False
    
    def detect_conflicts(self, paths: Dict[int, List[Position]]) -> List[Conflict]:
        """
        Detect all conflicts between agent paths with comprehensive conflict types
        
        Args:
            paths: Dictionary mapping agent_id to path
            
        Returns:
            List of conflicts found
        """
        if not paths:
            return []
        
        conflicts = []
        agent_ids = list(paths.keys())
        
        # Check all pairs of agents
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1_id = agent_ids[i]
                agent2_id = agent_ids[j]
                
                path1 = paths[agent1_id]
                path2 = paths[agent2_id]
                
                if not path1 or not path2:
                    continue
                
                # Find conflicts between this pair
                pair_conflicts = self._detect_conflicts_between_agents(
                    agent1_id, path1, agent2_id, path2
                )
                conflicts.extend(pair_conflicts)
        
        # Sort conflicts by time (earlier conflicts first)
        conflicts.sort(key=lambda c: c.time)
        
        return conflicts
    
    def _detect_conflicts_between_agents(self, agent1_id: int, path1: List[Position],
                                       agent2_id: int, path2: List[Position]) -> List[Conflict]:
        """
        Detect conflicts between two specific agents with multiple conflict types
        """
        conflicts = []
        
        # Extend paths to same length (agents wait at goal)
        extended_path1 = self._extend_path_to_time(path1, 
            max(path1[-1].t if path1 else 0, path2[-1].t if path2 else 0))
        extended_path2 = self._extend_path_to_time(path2,
            max(path1[-1].t if path1 else 0, path2[-1].t if path2 else 0))
        
        min_length = min(len(extended_path1), len(extended_path2))
        
        # Check vertex conflicts (same position at same time)
        for i in range(min_length):
            pos1 = extended_path1[i]
            pos2 = extended_path2[i]
            
            if pos1 == pos2:
                conflict = Conflict(
                    conflict_type=ConflictType.VERTEX,
                    agent1_id=agent1_id,
                    agent2_id=agent2_id,
                    position=pos1,
                    time=pos1.t
                )
                conflicts.append(conflict)
        
        # Check edge conflicts (agents swap positions)
        if self.detect_edge_conflicts:
            for i in range(1, min_length):
                pos1_prev = extended_path1[i-1]
                pos1_curr = extended_path1[i]
                pos2_prev = extended_path2[i-1]
                pos2_curr = extended_path2[i]
                
                # Check if agents swap positions
                if (pos1_prev == pos2_curr and pos1_curr == pos2_prev and 
                    pos1_prev != pos1_curr):  # Ensure actual movement
                    
                    conflict = Conflict(
                        conflict_type=ConflictType.EDGE,
                        agent1_id=agent1_id,
                        agent2_id=agent2_id,
                        position=pos1_curr,
                        time=pos1_curr.t,
                        agent1_prev_position=pos1_prev,
                        agent2_prev_position=pos2_prev
                    )
                    conflicts.append(conflict)
        
        # Check following conflicts (agents too close)
        if self.detect_following_conflicts and self.min_agent_separation > 1:
            conflicts.extend(self._detect_following_conflicts(
                agent1_id, extended_path1, agent2_id, extended_path2
            ))
        
        return conflicts
    
    def _detect_following_conflicts(self, agent1_id: int, path1: List[Position],
                                  agent2_id: int, path2: List[Position]) -> List[Conflict]:
        """Detect following conflicts (agents too close spatially)"""
        conflicts = []
        min_length = min(len(path1), len(path2))
        
        for i in range(min_length):
            pos1 = path1[i]
            pos2 = path2[i]
            
            # Calculate spatial distance
            distance = pos1.manhattan_distance_to(pos2)
            
            if 0 < distance < self.min_agent_separation:
                # Determine which agent is following
                following_agent = agent2_id if i == 0 else (
                    agent1_id if path1[i-1].manhattan_distance_to(pos1) > 
                    path2[i-1].manhattan_distance_to(pos2) else agent2_id
                )
                
                conflict = Conflict(
                    conflict_type=ConflictType.FOLLOWING,
                    agent1_id=agent1_id,
                    agent2_id=agent2_id,
                    position=pos1 if following_agent == agent1_id else pos2,
                    time=pos1.t,
                    min_separation=self.min_agent_separation
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _extend_path_to_time(self, path: List[Position], target_time: int) -> List[Position]:
        """
        Extend path by having agent wait at goal position
        
        Args:
            path: Original path
            target_time: Time to extend to
            
        Returns:
            Extended path
        """
        if not path or target_time <= path[-1].t:
            return path
        
        extended = path.copy()
        last_pos = path[-1]
        
        # Add waiting positions at the goal
        for t in range(last_pos.t + 1, target_time + 1):
            extended.append(Position(last_pos.x, last_pos.y, t))
        
        return extended
    
    def _generate_constraints_from_conflict(self, conflict: Conflict) -> List[CBSConstraint]:
        """
        Generate CBS constraints from a conflict
        
        Args:
            conflict: Conflict to resolve
            
        Returns:
            List of CBS constraints (typically 2, one for each agent)
        """
        constraints = []
        
        if conflict.conflict_type == ConflictType.VERTEX:
            # Vertex conflict: forbid both agents from being at this position
            for agent_id in [conflict.agent1_id, conflict.agent2_id]:
                base_constraint = Constraint(
                    constraint_type=ConstraintType.VERTEX,
                    position=conflict.position,
                    agent_id=agent_id
                )
                
                cbs_constraint = CBSConstraint(
                    base_constraint=base_constraint,
                    source_conflict=conflict,
                    target_agent_id=agent_id,
                    priority=self._calculate_constraint_priority(conflict, agent_id)
                )
                constraints.append(cbs_constraint)
        
        elif conflict.conflict_type == ConflictType.EDGE:
            # Edge conflict: forbid the edge traversal for both agents
            if conflict.agent1_prev_position and conflict.agent2_prev_position:
                # Constraint for agent 1
                base_constraint1 = Constraint(
                    constraint_type=ConstraintType.EDGE,
                    position=conflict.position,
                    prev_position=conflict.agent1_prev_position,
                    agent_id=conflict.agent1_id
                )
                
                cbs_constraint1 = CBSConstraint(
                    base_constraint=base_constraint1,
                    source_conflict=conflict,
                    target_agent_id=conflict.agent1_id,
                    priority=self._calculate_constraint_priority(conflict, conflict.agent1_id)
                )
                constraints.append(cbs_constraint1)
                
                # Constraint for agent 2
                base_constraint2 = Constraint(
                    constraint_type=ConstraintType.EDGE,
                    position=conflict.agent2_prev_position,
                    prev_position=conflict.position,
                    agent_id=conflict.agent2_id
                )
                
                cbs_constraint2 = CBSConstraint(
                    base_constraint=base_constraint2,
                    source_conflict=conflict,
                    target_agent_id=conflict.agent2_id,
                    priority=self._calculate_constraint_priority(conflict, conflict.agent2_id)
                )
                constraints.append(cbs_constraint2)
        
        elif conflict.conflict_type == ConflictType.FOLLOWING:
            # Following conflict: add temporal constraint for following agent
            # Create a temporal constraint to maintain separation
            start_time = max(0, conflict.time - conflict.min_separation)
            end_time = conflict.time + conflict.min_separation
            
            for agent_id in [conflict.agent1_id, conflict.agent2_id]:
                base_constraint = Constraint(
                    constraint_type=ConstraintType.TEMPORAL,
                    position=conflict.position,
                    time_range=(start_time, end_time),
                    agent_id=agent_id
                )
                
                cbs_constraint = CBSConstraint(
                    base_constraint=base_constraint,
                    source_conflict=conflict,
                    target_agent_id=agent_id,
                    priority=self._calculate_constraint_priority(conflict, agent_id)
                )
                constraints.append(cbs_constraint)
        
        return constraints
    
    def _calculate_constraint_priority(self, conflict: Conflict, agent_id: int) -> int:
        """Calculate priority for a constraint based on conflict and agent"""
        # Base priority from conflict type
        base_priority = {
            ConflictType.VERTEX: 10,
            ConflictType.EDGE: 8,
            ConflictType.FOLLOWING: 5
        }.get(conflict.conflict_type, 1)
        
        # Adjust based on agent priority if available
        if agent_id in self.grid.agents:
            agent_priority = self.grid.agents[agent_id].priority
            base_priority += agent_priority // 10  # Scale down agent priority
        
        # Earlier conflicts have higher priority
        time_factor = max(1, 100 - conflict.time)
        
        return base_priority + time_factor
    
    def _apply_constraints_to_pathfinder(self, constraints: List[CBSConstraint], 
                                       target_agent_id: int):
        """
        Apply CBS constraints to the pathfinder for a specific agent
        
        Args:
            constraints: List of CBS constraints
            target_agent_id: Agent to apply constraints to
        """
        # Clear existing constraints for this agent
        self.pathfinder.remove_constraints_for_agent(target_agent_id)
        
        # Add new constraints
        for cbs_constraint in constraints:
            if cbs_constraint.target_agent_id == target_agent_id:
                self.pathfinder.add_constraint(cbs_constraint.base_constraint, target_agent_id)
    
    def _find_path_with_constraints(self, agent: Agent, constraints: List[CBSConstraint],
                                  start_time: int = 0) -> PathfindingResult:
        """
        Find path for agent with given constraints applied
        
        Args:
            agent: Agent to plan for
            constraints: Constraints to apply
            start_time: Starting time
            
        Returns:
            Pathfinding result
        """
        # Apply constraints to pathfinder
        self._apply_constraints_to_pathfinder(constraints, agent.id)
        
        # Find path
        result = self.pathfinder.find_path(agent, start_time)
        
        # Clean up constraints
        self.pathfinder.remove_constraints_for_agent(agent.id)
        
        return result
    
    def _calculate_conflict_density(self, conflicts: List[Conflict]) -> Dict[Tuple[int, int], int]:
        """Calculate conflict density per spatial grid cell"""
        density = defaultdict(int)
        
        for conflict in conflicts:
            spatial_pos = conflict.position.spatial_position()
            density[spatial_pos] += 1
            
            # For edge conflicts, also count the source positions
            if (conflict.conflict_type == ConflictType.EDGE and 
                conflict.agent1_prev_position and conflict.agent2_prev_position):
                
                density[conflict.agent1_prev_position.spatial_position()] += 1
                density[conflict.agent2_prev_position.spatial_position()] += 1
        
        return dict(density)
    
    def solve(self, agents: List[Agent], start_time: int = 0) -> CBSResult:
        """
        Main CBS solving method with comprehensive conflict resolution
        
        Args:
            agents: List of agents to find paths for
            start_time: Starting time for all agents
            
        Returns:
            CBSResult with solution or detailed failure information
        """
        with self._search_lock:
            start_solve_time = time.time()
            self.stats['total_runs'] += 1
            
            result = CBSResult()
            result.pathfinding_calls = 0
            
            if not agents:
                result.success = True
                result.computation_time = time.time() - start_solve_time
                return result
            
            if self.debug_mode:
                print(f"\nCBS: Starting search with {len(agents)} agents")
            
            # Initialize root node with unconstrained paths
            root_node = self._create_root_node(agents, start_time, result)
            
            if not root_node:
                result.failure_reason = "Could not create initial solution"
                result.should_trigger_auction = True
                result.auction_candidates = [a.id for a in agents]
                result.computation_time = time.time() - start_solve_time
                return result
            
            # Check if root is already a solution
            if root_node.is_solution():
                result.success = True
                result.paths = root_node.paths
                result.total_cost = root_node.solution_cost
                result.iterations_used = 1
                result.computation_time = time.time() - start_solve_time
                self.stats['successful_runs'] += 1
                return result
            
            # CBS high-level search
            open_list = [root_node]
            heapq.heapify(open_list)
            
            iterations = 0
            best_node = root_node
            
            while (open_list and 
                   iterations < self.max_iterations and 
                   time.time() - start_solve_time < self.max_computation_time):
                
                iterations += 1
                
                # Get best node
                current_node = heapq.heappop(open_list)
                
                if self.debug_mode and iterations % 100 == 0:
                    print(f"CBS: Iteration {iterations}, conflicts: {len(current_node.conflicts)}")
                
                # Track best partial solution
                if (len(current_node.conflicts) < len(best_node.conflicts) or
                    (len(current_node.conflicts) == len(best_node.conflicts) and
                     current_node.solution_cost < best_node.solution_cost)):
                    best_node = current_node
                
                # Check if solution found
                if current_node.is_solution():
                    result.success = True
                    result.paths = current_node.paths
                    result.total_cost = current_node.solution_cost
                    result.iterations_used = iterations
                    result.nodes_expanded = iterations
                    result.solution_depth = current_node.depth
                    result.computation_time = time.time() - start_solve_time
                    result.constraints_generated = len(current_node.constraints)
                    
                    self.stats['successful_runs'] += 1
                    self._update_stats(result)
                    
                    if self.debug_mode:
                        print(f"CBS: Solution found at iteration {iterations}")
                    
                    return result
                
                # Select conflict to resolve
                if not current_node.conflicts:
                    continue
                
                conflict = self._select_conflict_to_resolve(current_node.conflicts)
                
                # Generate child nodes
                child_nodes = self._generate_child_nodes(current_node, conflict, agents, 
                                                       start_time, result)
                
                # Add valid children to open list
                for child in child_nodes:
                    if child and len(child.conflicts) <= self.max_conflicts_per_node:
                        heapq.heappush(open_list, child)
                
                # Early termination check
                if (self.early_termination and 
                    len(open_list) > self.max_iterations // 2 and
                    time.time() - start_solve_time > self.max_computation_time / 2):
                    break
            
            # Search failed - prepare result with best partial solution
            result.iterations_used = iterations
            result.nodes_expanded = iterations
            result.computation_time = time.time() - start_solve_time
            result.final_conflicts = best_node.conflicts
            result.conflict_density = self._calculate_conflict_density(best_node.conflicts)
            result.partial_solution = best_node.paths
            result.constraints_generated = len(best_node.constraints)
            
            # Determine failure reason and auction candidates
            if iterations >= self.max_iterations:
                result.failure_reason = f"Maximum iterations ({self.max_iterations}) exceeded"
            elif time.time() - start_solve_time >= self.max_computation_time:
                result.failure_reason = f"Time limit ({self.max_computation_time}s) exceeded"
            else:
                result.failure_reason = "Search space exhausted"
            
            # Identify agents that should go to auction
            result.should_trigger_auction = True
            result.auction_candidates = list(best_node.unresolved_agents)
            
            # If no specific unresolved agents, use agents involved in conflicts
            if not result.auction_candidates:
                involved_agents = set()
                for conflict in best_node.conflicts:
                    involved_agents.update(conflict.get_involved_agents())
                result.auction_candidates = list(involved_agents)
            
            self._update_stats(result)
            
            if self.debug_mode:
                print(f"CBS: Search failed after {iterations} iterations")
                print(f"CBS: Best solution had {len(best_node.conflicts)} conflicts")
                print(f"CBS: Auction candidates: {result.auction_candidates}")
            
            return result
    
    def _create_root_node(self, agents: List[Agent], start_time: int, 
                         result: CBSResult) -> Optional[CBSNode]:
        """Create root node with initial unconstrained paths"""
        root_node = CBSNode()
        
        # Find initial paths for all agents
        for agent in agents:
            path_result = self.pathfinder.find_path(agent, start_time)
            result.pathfinding_calls += 1
            
            if path_result.success:
                root_node.paths[agent.id] = path_result.path
            else:
                root_node.unresolved_agents.add(agent.id)
        
        # If no agents got paths, return None
        if not root_node.paths:
            return None
        
        # Detect conflicts in initial solution
        conflict_start = time.time()
        root_node.conflicts = self.detect_conflicts(root_node.paths)
        result.conflict_detection_time += time.time() - conflict_start
        
        result.conflicts_found.extend(root_node.conflicts)
        root_node.solution_cost = sum(len(path) for path in root_node.paths.values())
        
        return root_node
    
    def _select_conflict_to_resolve(self, conflicts: List[Conflict]) -> Conflict:
        """Select which conflict to resolve next"""
        if not conflicts:
            return None
        
        if not self.conflict_prioritization:
            return conflicts[0]
        
        # Prioritize by conflict type and time
        def conflict_priority(conflict):
            type_priority = {
                ConflictType.VERTEX: 3,
                ConflictType.EDGE: 2, 
                ConflictType.FOLLOWING: 1
            }.get(conflict.conflict_type, 0)
            
            # Earlier conflicts are more important
            time_priority = 1000 - conflict.time
            
            return type_priority * 1000 + time_priority
        
        return max(conflicts, key=conflict_priority)
    
    def _generate_child_nodes(self, parent_node: CBSNode, conflict: Conflict,
                            agents: List[Agent], start_time: int, 
                            result: CBSResult) -> List[CBSNode]:
        """Generate child nodes by resolving the given conflict"""
        child_nodes = []
        
        # Generate constraints from conflict
        constraint_start = time.time()
        new_constraints = self._generate_constraints_from_conflict(conflict)
        result.constraint_generation_time += time.time() - constraint_start
        result.constraints_generated += len(new_constraints)
        
        # Group constraints by target agent
        constraints_by_agent = defaultdict(list)
        for constraint in new_constraints:
            constraints_by_agent[constraint.target_agent_id].append(constraint)
        
        # Create one child node for each agent involved in the conflict
        for agent_id, agent_constraints in constraints_by_agent.items():
            child_node = self._create_child_node(parent_node, agent_constraints, 
                                               agents, start_time, result)
            if child_node:
                child_nodes.append(child_node)
        
        return child_nodes
    
    def _create_child_node(self, parent_node: CBSNode, new_constraints: List[CBSConstraint],
                          agents: List[Agent], start_time: int, 
                          result: CBSResult) -> Optional[CBSNode]:
        """Create a child node with additional constraints"""
        # Find the target agent
        if not new_constraints:
            return None
        
        target_agent_id = new_constraints[0].target_agent_id
        target_agent = next((a for a in agents if a.id == target_agent_id), None)
        
        if not target_agent:
            return None
        
        # Create child node
        child_node = CBSNode(
            constraints=parent_node.constraints + new_constraints,
            paths=parent_node.paths.copy(),
            depth=parent_node.depth + 1,
            parent=parent_node,
            unresolved_agents=parent_node.unresolved_agents.copy()
        )
        
        # Replan path for target agent with new constraints
        all_constraints = child_node.constraints
        path_result = self._find_path_with_constraints(target_agent, all_constraints, start_time)
        result.pathfinding_calls += 1
        result.agents_replanned.add(target_agent_id)
        
        if path_result.success:
            child_node.paths[target_agent_id] = path_result.path
            child_node.unresolved_agents.discard(target_agent_id)
        else:
            # Agent couldn't find path with constraints
            child_node.unresolved_agents.add(target_agent_id)
            if target_agent_id in child_node.paths:
                del child_node.paths[target_agent_id]
        
        # Detect conflicts in new solution
        conflict_start = time.time()
        child_node.conflicts = self.detect_conflicts(child_node.paths)
        result.conflict_detection_time += time.time() - conflict_start
        
        # Calculate solution cost
        child_node.solution_cost = sum(len(path) for path in child_node.paths.values())
        
        return child_node
    
    def _update_stats(self, result: CBSResult):
        """Update CBS statistics"""
        self.stats['average_iterations'] = (
            (self.stats['average_iterations'] * (self.stats['total_runs'] - 1) + 
             result.iterations_used) / self.stats['total_runs']
        )
        
        self.stats['average_computation_time'] = (
            (self.stats['average_computation_time'] * (self.stats['total_runs'] - 1) + 
             result.computation_time) / self.stats['total_runs']
        )
        
        self.stats['total_conflicts_detected'] += len(result.conflicts_found)
        self.stats['total_constraints_generated'] += result.constraints_generated
        
        if result.should_trigger_auction:
            self.stats['auction_triggers'] += 1
    
    def validate_auction_winners(self, winner_paths: Dict[int, List[Position]],
                                emergency_paths: Dict[int, List[Position]] = None) -> Dict[int, List[Position]]:
        """
        Validate and resolve conflicts among auction winners using CBS
        
        Args:
            winner_paths: Paths of auction winners (priority-ordered)
            emergency_paths: Emergency agent paths to avoid
            
        Returns:
            Conflict-free paths for valid winners
        """
        if not winner_paths:
            return {}
        
        if emergency_paths is None:
            emergency_paths = {}
        
        # Add emergency paths as immutable constraints
        all_paths = emergency_paths.copy()
        all_paths.update(winner_paths)
        
        # Detect conflicts
        conflicts = self.detect_conflicts(all_paths)
        
        if not conflicts:
            return winner_paths  # No conflicts
        
        # Filter conflicts to only those involving auction winners
        winner_conflicts = [
            c for c in conflicts 
            if c.agent1_id in winner_paths or c.agent2_id in winner_paths
        ]
        
        if not winner_conflicts:
            return winner_paths
        
        # Resolve conflicts by priority (assuming winner_paths is ordered by priority)
        validated_paths = {}
        winner_ids = list(winner_paths.keys())
        
        # Add emergency paths as fixed constraints
        for agent_id, path in emergency_paths.items():
            for pos in path:
                constraint = Constraint(
                    constraint_type=ConstraintType.VERTEX,
                    position=pos
                )
                self.pathfinder.add_constraint(constraint)
        
        try:
            # Process winners in priority order
            for winner_id in winner_ids:
                path = winner_paths[winner_id]
                
                # Check if this path conflicts with already validated paths
                test_paths = validated_paths.copy()
                test_paths[winner_id] = path
                
                test_conflicts = self.detect_conflicts(test_paths)
                
                # If no new conflicts, accept this winner
                if not test_conflicts:
                    validated_paths[winner_id] = path
                
                # Add this winner's path as constraints for subsequent winners
                for pos in path:
                    constraint = Constraint(
                        constraint_type=ConstraintType.VERTEX,
                        position=pos
                    )
                    self.pathfinder.add_constraint(constraint)
        
        finally:
            # Clean up all constraints
            self.pathfinder.clear_all_constraints()
        
        return validated_paths
    
    def get_statistics(self) -> Dict:
        """Get comprehensive CBS statistics"""
        return {
            'total_runs': self.stats['total_runs'],
            'successful_runs': self.stats['successful_runs'],
            'success_rate': self.stats['successful_runs'] / max(self.stats['total_runs'], 1),
            'average_iterations': self.stats['average_iterations'],
            'average_computation_time': self.stats['average_computation_time'],
            'total_conflicts_detected': self.stats['total_conflicts_detected'],
            'total_constraints_generated': self.stats['total_constraints_generated'],
            'auction_triggers': self.stats['auction_triggers'],
            'auction_trigger_rate': self.stats['auction_triggers'] / max(self.stats['total_runs'], 1),
            'max_iterations_limit': self.max_iterations,
            'max_time_limit': self.max_computation_time
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug output"""
        self.debug_mode = enabled
    
    def reset_statistics(self):
        """Reset CBS statistics"""
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'average_iterations': 0.0,
            'average_computation_time': 0.0,
            'total_conflicts_detected': 0,
            'total_constraints_generated': 0,
            'auction_triggers': 0
        }
    
    def configure_search(self, max_iterations: int = None, max_time: float = None,
                        detect_edge_conflicts: bool = None, 
                        detect_following_conflicts: bool = None):
        """Configure CBS search parameters"""
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if max_time is not None:
            self.max_computation_time = max_time
        if detect_edge_conflicts is not None:
            self.detect_edge_conflicts = detect_edge_conflicts
        if detect_following_conflicts is not None:
            self.detect_following_conflicts = detect_following_conflicts

# Utility functions for integration with other modules

def create_cbs_solver(grid_system: GridSystem, pathfinder: AStarPathfinder = None) -> ConflictBasedSearch:
    """
    Create CBS solver with optional custom pathfinder
    
    Args:
        grid_system: Grid system instance
        pathfinder: Optional pathfinder (creates default if None)
        
    Returns:
        Configured CBS solver
    """
    if pathfinder is None:
        # Create pathfinder with conservative config for CBS
        config = create_conservative_config()
        pathfinder = AStarPathfinder(grid_system, config)
    
    return ConflictBasedSearch(grid_system, pathfinder)

def extract_non_emergency_agents(grid_system: GridSystem) -> List[Agent]:
    """Extract non-emergency agents that need path planning"""
    return grid_system.get_agents_by_type(AgentType.NON_EMERGENCY)

def extract_agents_without_paths(grid_system: GridSystem, 
                                agent_type: AgentType = None) -> List[Agent]:
    """Extract agents that don't have assigned paths"""
    return grid_system.get_unassigned_agents(agent_type)

def should_trigger_auction(cbs_result: CBSResult, threshold: float = 0.5) -> bool:
    """
    Determine if auction should be triggered based on CBS result
    
    Args:
        cbs_result: Result from CBS
        threshold: Minimum success rate threshold (agents with paths / total agents)
        
    Returns:
        True if auction should be triggered
    """
    if cbs_result.should_trigger_auction:
        return True
    
    # Check if too many agents couldn't get paths
    if cbs_result.auction_candidates:
        total_agents = len(cbs_result.paths) + len(cbs_result.auction_candidates)
        success_rate = len(cbs_result.paths) / total_agents if total_agents > 0 else 0
        return success_rate < threshold
    
    return False

def get_conflict_density_for_auction(cbs_result: CBSResult) -> Dict[Tuple[int, int], int]:
    """Extract conflict density for auction pricing"""
    return dict(cbs_result.conflict_density)

def convert_cbs_paths_to_grid_format(cbs_paths: Dict[int, List[Position]]) -> Dict[int, List[Position]]:
    """Convert CBS paths to format expected by grid system (already in correct format)"""
    return cbs_paths

def convert_tuple_paths_to_positions(tuple_paths: Dict[int, List[Tuple[int, int, int]]]) -> Dict[int, List[Position]]:
    """Convert tuple format paths to Position objects"""
    return {
        agent_id: [Position(x, y, t) for x, y, t in path]
        for agent_id, path in tuple_paths.items()
    }

def convert_positions_to_tuple_paths(position_paths: Dict[int, List[Position]]) -> Dict[int, List[Tuple[int, int, int]]]:
    """Convert Position paths to tuple format"""
    return {
        agent_id: [pos.to_tuple() for pos in path]
        for agent_id, path in position_paths.items()
    }

def analyze_cbs_performance(cbs_result: CBSResult) -> Dict:
    """
    Analyze CBS performance and provide insights
    
    Args:
        cbs_result: Result from CBS execution
        
    Returns:
        Performance analysis dictionary
    """
    analysis = {
        'solution_quality': 'optimal' if cbs_result.success else 'partial',
        'efficiency_rating': 'high',  # Will be calculated
        'bottlenecks': [],
        'recommendations': []
    }
    
    # Calculate efficiency rating
    if cbs_result.computation_time > 60:
        analysis['efficiency_rating'] = 'low'
        analysis['bottlenecks'].append('high_computation_time')
        analysis['recommendations'].append('Consider reducing max_iterations or using faster pathfinding config')
    elif cbs_result.computation_time > 10:
        analysis['efficiency_rating'] = 'medium'
    
    # Check iteration efficiency
    if cbs_result.iterations_used > 500:
        analysis['bottlenecks'].append('high_iteration_count')
        analysis['recommendations'].append('Consider enabling early termination or conflict prioritization')
    
    # Check pathfinding efficiency
    avg_pathfinding_time = cbs_result.computation_time / max(cbs_result.pathfinding_calls, 1)
    if avg_pathfinding_time > 1.0:
        analysis['bottlenecks'].append('slow_pathfinding')
        analysis['recommendations'].append('Consider using faster pathfinding configuration')
    
    # Check conflict density
    if cbs_result.conflict_density:
        max_density = max(cbs_result.conflict_density.values())
        if max_density > 10:
            analysis['bottlenecks'].append('high_conflict_density')
            analysis['recommendations'].append('Consider using auction system for high-conflict areas')
    
    # Success rate analysis
    if not cbs_result.success and cbs_result.auction_candidates:
        analysis['recommendations'].append(f'Send {len(cbs_result.auction_candidates)} agents to auction')
    
    return analysis

class CBSProfiler:
    """Profiler for CBS performance analysis"""
    
    def __init__(self):
        self.results_history: List[CBSResult] = []
    
    def record_result(self, result: CBSResult):
        """Record a CBS result for analysis"""
        self.results_history.append(result)
    
    def get_performance_summary(self) -> Dict:
        """Get summary of CBS performance over time"""
        if not self.results_history:
            return {}
        
        successful_runs = [r for r in self.results_history if r.success]
        
        return {
            'total_runs': len(self.results_history),
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / len(self.results_history),
            'average_computation_time': sum(r.computation_time for r in self.results_history) / len(self.results_history),
            'average_iterations': sum(r.iterations_used for r in self.results_history) / len(self.results_history),
            'total_conflicts_detected': sum(len(r.conflicts_found) for r in self.results_history),
            'auction_trigger_rate': sum(1 for r in self.results_history if r.should_trigger_auction) / len(self.results_history)
        }
    
    def clear_history(self):
        """Clear performance history"""
        self.results_history.clear()