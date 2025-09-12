"""
Enhanced Conflict-Based Search (CBS) Module with Emergency Agent Priority
Implements CBS algorithm with two-tier processing: emergency agents first,
then regular agents with emergency paths as dynamic obstacles.
"""

import heapq
from typing import Dict, List, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
import time

# Import required modules
from enhanced_multilayer_grid_system import (
    EnhancedMultilayerGridSystem, LayerType, AgentState, AgentType, ProcessingPhase
)
from enhanced_astar_pathfinding import (
    EnhancedAStarPathfinder, PathfindingRequest, PathfindingResponse, 
    PathfindingResult
)

class ConflictType(Enum):
    """Types of conflicts between agents"""
    VERTEX = "vertex"      # Same position at same time
    EDGE = "edge"         # Swapping positions
    FOLLOWING = "following"  # Following too closely
    EMERGENCY = "emergency"  # Conflict with emergency agent path

class CBSResult(Enum):
    """CBS algorithm results"""
    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    PATHFINDING_FAILED = "pathfinding_failed"
    EMERGENCY_CONFLICT = "emergency_conflict"
    PARTIAL_SUCCESS = "partial_success"  # Some agents got paths

@dataclass
class Conflict:
    """Representation of a conflict between agents or with emergency paths"""
    type: ConflictType
    agent1: int
    agent2: Optional[int]  # None for emergency conflicts
    position1: Tuple[int, int]
    position2: Optional[Tuple[int, int]]
    time_step: int
    emergency_agent_id: Optional[int] = None  # ID of blocking emergency agent
    
    def __str__(self) -> str:
        if self.type == ConflictType.EMERGENCY:
            return f"Emergency conflict: agent {self.agent1} blocked by emergency agent {self.emergency_agent_id} at {self.position1} at time {self.time_step}"
        elif self.type == ConflictType.VERTEX:
            return f"Vertex conflict: agents {self.agent1} and {self.agent2} at {self.position1} at time {self.time_step}"
        elif self.type == ConflictType.EDGE:
            return f"Edge conflict: agents {self.agent1} and {self.agent2} swapping {self.position1}<->{self.position2} at time {self.time_step}"
        else:
            return f"Following conflict: agents {self.agent1} and {self.agent2} at time {self.time_step}"

@dataclass
class Constraint:
    """Constraint to prevent specific agent actions"""
    agent_id: int
    position: Tuple[int, int]
    time_step: int
    constraint_type: ConflictType
    emergency_source: Optional[int] = None  # Emergency agent causing constraint
    
    def __str__(self) -> str:
        if self.emergency_source:
            return f"Agent {self.agent_id}: avoid {self.position} at time {self.time_step} (emergency agent {self.emergency_source})"
        return f"Agent {self.agent_id}: avoid {self.position} at time {self.time_step} ({self.constraint_type.value})"

@dataclass
class CBSNode:
    """Node in the CBS search tree"""
    constraints: List[Constraint] = field(default_factory=list)
    solution: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    cost: float = 0.0
    conflicts: List[Conflict] = field(default_factory=list)
    emergency_conflicts: List[Conflict] = field(default_factory=list)  # Separate tracking
    
    def __lt__(self, other):
        """Comparison for priority queue (lowest cost first)"""
        # Prioritize nodes with fewer emergency conflicts
        if len(self.emergency_conflicts) != len(other.emergency_conflicts):
            return len(self.emergency_conflicts) < len(other.emergency_conflicts)
        return self.cost < other.cost

@dataclass
class CBSRequest:
    """Request structure for CBS operations"""
    agents: List[Dict]  # Agent info: id, start, goal, agent_type, etc.
    layer: LayerType
    agent_type: AgentType  # EMERGENCY or REGULAR
    start_time: int
    max_time_steps: int = 100
    max_iterations: int = 400
    timeout_seconds: float = 30.0
    enable_prioritization: bool = True
    ignore_emergency_conflicts: bool = False  # For emergency agent processing

@dataclass
class CBSResponse:
    """Response structure for CBS operations"""
    result: CBSResult
    agent_type: AgentType
    solution: Dict[int, List[Tuple[int, int]]]  # agent_id -> path
    time_steps: Dict[int, List[int]]  # agent_id -> time_steps
    total_cost: float
    computation_time: float
    iterations_used: int
    conflicts_resolved: int
    emergency_conflicts: List[Conflict]  # Conflicts with emergency agents
    regular_conflicts: List[Conflict]   # Conflicts between regular agents
    failed_agents: List[int]  # Agents that couldn't get paths
    successful_agents: List[int]  # Agents that got paths
    message: str = ""

class EnhancedCBS:
    """
    Enhanced Conflict-Based Search with emergency agent priority.
    
    Supports two-phase processing:
    1. Emergency agents: Process with only static obstacles
    2. Regular agents: Process with emergency paths as dynamic obstacles
    """
    
    def __init__(self, grid_system: EnhancedMultilayerGridSystem, pathfinder: EnhancedAStarPathfinder):
        """
        Initialize enhanced CBS with required components.
        
        Args:
            grid_system: EnhancedMultilayerGridSystem for spatial queries
            pathfinder: EnhancedAStarPathfinder for individual agent pathfinding
        """
        self.grid_system = grid_system
        self.pathfinder = pathfinder
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.emergency_requests = 0
        self.regular_requests = 0
        self.total_conflicts_detected = 0
        self.total_conflicts_resolved = 0
        self.emergency_conflicts_detected = 0
    
    def detect_conflicts(self, solution: Dict[int, List[Tuple[int, int]]], 
                        agent_type: AgentType, layer: LayerType, start_time: int) -> Tuple[List[Conflict], List[Conflict]]:
        """
        Detect conflicts in a given solution, separating emergency and regular conflicts.
        
        Args:
            solution: Dictionary mapping agent_id to path
            agent_type: Type of agents in the solution
            layer: Layer being processed
            start_time: Starting time step
            
        Returns:
            Tuple: (regular_conflicts, emergency_conflicts)
        """
        regular_conflicts = []
        emergency_conflicts = []
        agent_ids = list(solution.keys())
        
        # For regular agents, check conflicts with emergency reservations
        if agent_type == AgentType.REGULAR:
            for agent_id, path in solution.items():
                emergency_path_conflicts = self._detect_emergency_conflicts(
                    agent_id, path, layer, start_time
                )
                emergency_conflicts.extend(emergency_path_conflicts)
        
        # Check conflicts between agents of the same type
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1, agent2 = agent_ids[i], agent_ids[j]
                path1, path2 = solution[agent1], solution[agent2]
                
                # Find conflicts between this pair
                pair_conflicts = self._detect_conflicts_between_agents(
                    agent1, path1, agent2, path2
                )
                regular_conflicts.extend(pair_conflicts)
        
        return regular_conflicts, emergency_conflicts
    
    def _detect_emergency_conflicts(self, agent_id: int, path: List[Tuple[int, int]], 
                                   layer: LayerType, start_time: int) -> List[Conflict]:
        """
        Detect conflicts between a regular agent and emergency reservations.
        
        Args:
            agent_id: Regular agent ID
            path: Agent's planned path
            layer: Layer being checked
            start_time: Starting time for the path
            
        Returns:
            List[Conflict]: Conflicts with emergency agents
        """
        conflicts = []
        
        for t, position in enumerate(path):
            time_step = start_time + t
            
            # Check if position is reserved by emergency agent
            is_reserved, emergency_agent_id = self.grid_system.emergency_reservations.is_position_reserved(
                layer, position, time_step
            )
            
            if is_reserved:
                conflict = Conflict(
                    type=ConflictType.EMERGENCY,
                    agent1=agent_id,
                    agent2=None,
                    position1=position,
                    position2=None,
                    time_step=time_step,
                    emergency_agent_id=emergency_agent_id
                )
                conflicts.append(conflict)
                self.emergency_conflicts_detected += 1
        
        return conflicts
    
    def _detect_conflicts_between_agents(self, agent1: int, path1: List[Tuple[int, int]],
                                       agent2: int, path2: List[Tuple[int, int]]) -> List[Conflict]:
        """
        Detect conflicts between two specific agents of the same type.
        
        Args:
            agent1: First agent ID
            path1: First agent's path
            agent2: Second agent ID  
            path2: Second agent's path
            
        Returns:
            List[Conflict]: Conflicts between the two agents
        """
        conflicts = []
        max_len = max(len(path1), len(path2))
        
        # Extend shorter path (agent stays at goal)
        extended_path1 = path1 + [path1[-1]] * (max_len - len(path1)) if path1 else []
        extended_path2 = path2 + [path2[-1]] * (max_len - len(path2)) if path2 else []
        
        for t in range(max_len):
            pos1 = extended_path1[t] if t < len(extended_path1) else extended_path1[-1]
            pos2 = extended_path2[t] if t < len(extended_path2) else extended_path2[-1]
            
            # Vertex conflict: same position at same time
            if pos1 == pos2:
                conflict = Conflict(
                    type=ConflictType.VERTEX,
                    agent1=agent1,
                    agent2=agent2,
                    position1=pos1,
                    position2=pos2,
                    time_step=t
                )
                conflicts.append(conflict)
            
            # Edge conflict: agents swapping positions
            elif t > 0:
                prev_pos1 = extended_path1[t-1] if t-1 < len(extended_path1) else extended_path1[-1]
                prev_pos2 = extended_path2[t-1] if t-1 < len(extended_path2) else extended_path2[-1]
                
                if pos1 == prev_pos2 and pos2 == prev_pos1:
                    conflict = Conflict(
                        type=ConflictType.EDGE,
                        agent1=agent1,
                        agent2=agent2,
                        position1=prev_pos1,
                        position2=prev_pos2,
                        time_step=t
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def create_constraints_from_conflict(self, conflict: Conflict) -> List[Constraint]:
        """
        Create constraints to resolve a given conflict.
        
        Args:
            conflict: Conflict to resolve
            
        Returns:
            List[Constraint]: Constraints for agents involved
        """
        constraints = []
        
        if conflict.type == ConflictType.EMERGENCY:
            # Only constrain the regular agent (can't constrain emergency agents)
            constraints.append(Constraint(
                agent_id=conflict.agent1,
                position=conflict.position1,
                time_step=conflict.time_step,
                constraint_type=ConflictType.EMERGENCY,
                emergency_source=conflict.emergency_agent_id
            ))
        
        elif conflict.type == ConflictType.VERTEX:
            # Both agents cannot be at the conflicting position at the conflicting time
            constraints.append(Constraint(
                agent_id=conflict.agent1,
                position=conflict.position1,
                time_step=conflict.time_step,
                constraint_type=ConflictType.VERTEX
            ))
            constraints.append(Constraint(
                agent_id=conflict.agent2,
                position=conflict.position2,
                time_step=conflict.time_step,
                constraint_type=ConflictType.VERTEX
            ))
        
        elif conflict.type == ConflictType.EDGE:
            # Prevent the edge traversal for both agents
            constraints.append(Constraint(
                agent_id=conflict.agent1,
                position=conflict.position1,
                time_step=conflict.time_step,
                constraint_type=ConflictType.EDGE
            ))
            constraints.append(Constraint(
                agent_id=conflict.agent2,
                position=conflict.position2,
                time_step=conflict.time_step,
                constraint_type=ConflictType.EDGE
            ))
        
        return constraints
    
    def find_path_with_constraints(self, agent_info: Dict, constraints: List[Constraint],
                                 layer: LayerType, start_time: int, max_time_steps: int) -> PathfindingResponse:
        """
        Find path for an agent considering given constraints.
        
        Args:
            agent_info: Agent information dictionary
            constraints: List of constraints to respect
            layer: Layer for pathfinding
            start_time: Starting time step
            max_time_steps: Maximum time steps allowed
            
        Returns:
            PathfindingResponse: Pathfinding result
        """
        # Filter constraints for this agent
        agent_constraints = [c for c in constraints if c.agent_id == agent_info['id']]
        
        # Temporarily mark constrained positions as occupied in grid
        # This is a simplified approach - ideally, we'd modify the pathfinder directly
        original_occupancy = {}
        for constraint in agent_constraints:
            pos = constraint.position
            time = constraint.time_step
            if 0 <= time < self.grid_system.max_time:
                x, y = pos
                if 0 <= x < self.grid_system.width and 0 <= y < self.grid_system.height:
                    # Store original state
                    original_state = self.grid_system.grid_occupancy[layer][time, x, y]
                    original_occupancy[(pos, time)] = original_state
                    # Mark as occupied
                    self.grid_system.grid_occupancy[layer][time, x, y] = True
        
        # Create pathfinding request
        request = PathfindingRequest(
            agent_id=agent_info['id'],
            start=agent_info['start'],
            goal=agent_info['goal'],
            start_time=start_time,
            layer=layer,
            agent_type=agent_info.get('agent_type', AgentType.REGULAR),
            max_time_steps=max_time_steps,
            allow_waiting=True
        )
        
        # Find path
        response = self.pathfinder.find_path(request)
        
        # Restore original occupancy
        for (pos, time), original_state in original_occupancy.items():
            x, y = pos
            self.grid_system.grid_occupancy[layer][time, x, y] = original_state
        
        return response
    
    def solve_cbs(self, request: CBSRequest) -> CBSResponse:
        """
        Main enhanced CBS algorithm implementation.
        
        Args:
            request: CBS request with all parameters
            
        Returns:
            CBSResponse: Complete CBS solution result
        """
        start_time = time.time()
        self.total_requests += 1
        
        if request.agent_type == AgentType.EMERGENCY:
            self.emergency_requests += 1
        else:
            self.regular_requests += 1
        
        # Input validation
        if not request.agents:
            return CBSResponse(
                result=CBSResult.INVALID_INPUT,
                agent_type=request.agent_type,
                solution={},
                time_steps={},
                total_cost=float('inf'),
                computation_time=time.time() - start_time,
                iterations_used=0,
                conflicts_resolved=0,
                emergency_conflicts=[],
                regular_conflicts=[],
                failed_agents=[],
                successful_agents=[],
                message="No agents provided"
            )
        
        # Initialize CBS
        open_list = []
        iterations = 0
        conflicts_resolved = 0
        
        # Create root CBS node
        root_node = CBSNode()
        
        # Find initial paths for all agents
        for agent_info in request.agents:
            path_response = self.find_path_with_constraints(
                agent_info, [], request.layer, request.start_time, request.max_time_steps
            )
            
            if path_response.result != PathfindingResult.SUCCESS:
                # For partial success handling
                if request.agent_type == AgentType.REGULAR:
                    continue  # Skip failed agents, might still solve others
                else:
                    return CBSResponse(
                        result=CBSResult.PATHFINDING_FAILED,
                        agent_type=request.agent_type,
                        solution={},
                        time_steps={},
                        total_cost=float('inf'),
                        computation_time=time.time() - start_time,
                        iterations_used=iterations,
                        conflicts_resolved=conflicts_resolved,
                        emergency_conflicts=[],
                        regular_conflicts=[],
                        failed_agents=[agent_info['id']],
                        successful_agents=[],
                        message=f"Initial pathfinding failed for agent {agent_info['id']}: {path_response.message}"
                    )
            
            root_node.solution[agent_info['id']] = path_response.path
            root_node.cost += path_response.total_cost
        
        # Detect conflicts in initial solution
        regular_conflicts, emergency_conflicts = self.detect_conflicts(
            root_node.solution, request.agent_type, request.layer, request.start_time
        )
        
        root_node.conflicts = regular_conflicts
        root_node.emergency_conflicts = emergency_conflicts
        
        # For regular agents, emergency conflicts are often irresolvable
        if request.agent_type == AgentType.REGULAR and emergency_conflicts and not request.ignore_emergency_conflicts:
            # Return partial success if some agents got paths
            successful_agents = list(root_node.solution.keys())
            failed_agents = [a['id'] for a in request.agents if a['id'] not in successful_agents]
            
            if successful_agents:
                time_steps = {}
                for agent_id, path in root_node.solution.items():
                    time_steps[agent_id] = list(range(request.start_time, request.start_time + len(path)))
                
                return CBSResponse(
                    result=CBSResult.PARTIAL_SUCCESS,
                    agent_type=request.agent_type,
                    solution=root_node.solution,
                    time_steps=time_steps,
                    total_cost=root_node.cost,
                    computation_time=time.time() - start_time,
                    iterations_used=iterations,
                    conflicts_resolved=conflicts_resolved,
                    emergency_conflicts=emergency_conflicts,
                    regular_conflicts=regular_conflicts,
                    failed_agents=failed_agents,
                    successful_agents=successful_agents,
                    message=f"Partial success: {len(successful_agents)}/{len(request.agents)} agents got paths"
                )
        
        # If no conflicts, we're done
        if not root_node.conflicts and not root_node.emergency_conflicts:
            time_steps = {}
            for agent_id, path in root_node.solution.items():
                time_steps[agent_id] = list(range(request.start_time, request.start_time + len(path)))
            
            self.successful_requests += 1
            return CBSResponse(
                result=CBSResult.SUCCESS,
                agent_type=request.agent_type,
                solution=root_node.solution,
                time_steps=time_steps,
                total_cost=root_node.cost,
                computation_time=time.time() - start_time,
                iterations_used=iterations,
                conflicts_resolved=conflicts_resolved,
                emergency_conflicts=[],
                regular_conflicts=[],
                failed_agents=[],
                successful_agents=list(root_node.solution.keys()),
                message="Solution found without conflicts"
            )
        
        heapq.heappush(open_list, root_node)
        
        # Main CBS loop
        while open_list and iterations < request.max_iterations:
            # Check timeout
            if time.time() - start_time > request.timeout_seconds:
                return CBSResponse(
                    result=CBSResult.TIMEOUT,
                    agent_type=request.agent_type,
                    solution={},
                    time_steps={},
                    total_cost=float('inf'),
                    computation_time=time.time() - start_time,
                    iterations_used=iterations,
                    conflicts_resolved=conflicts_resolved,
                    emergency_conflicts=[],
                    regular_conflicts=[],
                    failed_agents=[],
                    successful_agents=[],
                    message=f"CBS timeout after {request.timeout_seconds} seconds"
                )
            
            current_node = heapq.heappop(open_list)
            iterations += 1
            
            # If no conflicts, solution found
            if not current_node.conflicts and not current_node.emergency_conflicts:
                time_steps = {}
                for agent_id, path in current_node.solution.items():
                    time_steps[agent_id] = list(range(request.start_time, request.start_time + len(path)))
                
                self.successful_requests += 1
                self.total_conflicts_resolved += conflicts_resolved
                
                return CBSResponse(
                    result=CBSResult.SUCCESS,
                    agent_type=request.agent_type,
                    solution=current_node.solution,
                    time_steps=time_steps,
                    total_cost=current_node.cost,
                    computation_time=time.time() - start_time,
                    iterations_used=iterations,
                    conflicts_resolved=conflicts_resolved,
                    emergency_conflicts=[],
                    regular_conflicts=[],
                    failed_agents=[],
                    successful_agents=list(current_node.solution.keys()),
                    message="CBS solution found successfully"
                )
            
            # Select conflict to resolve (prioritize regular conflicts over emergency conflicts)
            conflict = None
            if current_node.conflicts:
                conflict = current_node.conflicts[0]
            elif current_node.emergency_conflicts and request.ignore_emergency_conflicts:
                conflict = current_node.emergency_conflicts[0]
            
            if conflict is None:
                continue
            
            self.total_conflicts_detected += 1
            
            # Create constraints for this conflict
            constraints = self.create_constraints_from_conflict(conflict)
            
            # Create child nodes for each constraint
            for constraint in constraints:
                child_node = CBSNode()
                child_node.constraints = current_node.constraints + [constraint]
                child_node.solution = copy.deepcopy(current_node.solution)
                
                # Find which agent to replan
                agent_to_replan = constraint.agent_id
                agent_info = next((a for a in request.agents if a['id'] == agent_to_replan), None)
                
                if agent_info is None:
                    continue
                
                # Replan path with new constraints
                path_response = self.find_path_with_constraints(
                    agent_info, child_node.constraints, request.layer, 
                    request.start_time, request.max_time_steps
                )
                
                if path_response.result == PathfindingResult.SUCCESS:
                    child_node.solution[agent_to_replan] = path_response.path
                    
                    # Calculate total cost
                    child_node.cost = sum(
                        self.pathfinder.get_path_cost(path, list(range(len(path))), 
                                                    agent_info.get('agent_type', AgentType.REGULAR))
                        for path in child_node.solution.values()
                    )
                    
                    # Detect conflicts in new solution
                    regular_conflicts, emergency_conflicts = self.detect_conflicts(
                        child_node.solution, request.agent_type, request.layer, request.start_time
                    )
                    child_node.conflicts = regular_conflicts
                    child_node.emergency_conflicts = emergency_conflicts
                    
                    # Add to open list
                    heapq.heappush(open_list, child_node)
            
            conflicts_resolved += 1
        
        # No solution found within limits
        result = CBSResult.TIMEOUT if iterations >= request.max_iterations else CBSResult.NO_SOLUTION
        message = f"CBS exceeded {request.max_iterations} iterations" if result == CBSResult.TIMEOUT else "No valid CBS solution exists"
        
        return CBSResponse(
            result=result,
            agent_type=request.agent_type,
            solution={},
            time_steps={},
            total_cost=float('inf'),
            computation_time=time.time() - start_time,
            iterations_used=iterations,
            conflicts_resolved=conflicts_resolved,
            emergency_conflicts=[],
            regular_conflicts=[],
            failed_agents=[],
            successful_agents=[],
            message=message
        )
    
    def solve_emergency_agents(self, layer: LayerType, emergency_agents: List[Dict], 
                              start_time: int) -> CBSResponse:
        """
        Solve CBS for emergency agents with highest priority.
        
        Args:
            layer: Layer to solve
            emergency_agents: List of emergency agent information
            start_time: Starting time step
            
        Returns:
            CBSResponse: CBS solution for emergency agents
        """
        request = CBSRequest(
            agents=emergency_agents,
            layer=layer,
            agent_type=AgentType.EMERGENCY,
            start_time=start_time,
            max_time_steps=200,  # More time for emergency agents
            max_iterations=800,  # More iterations allowed
            timeout_seconds=60.0,  # More time allowed
            enable_prioritization=True,
            ignore_emergency_conflicts=True  # Emergency agents don't conflict with emergency paths
        )
        
        return self.solve_cbs(request)
    
    def solve_regular_agents(self, layer: LayerType, regular_agents: List[Dict], 
                           start_time: int) -> CBSResponse:
        """
        Solve CBS for regular agents considering emergency reservations.
        
        Args:
            layer: Layer to solve
            regular_agents: List of regular agent information
            start_time: Starting time step
            
        Returns:
            CBSResponse: CBS solution for regular agents
        """
        request = CBSRequest(
            agents=regular_agents,
            layer=layer,
            agent_type=AgentType.REGULAR,
            start_time=start_time,
            max_time_steps=100,
            max_iterations=400,
            timeout_seconds=30.0,
            enable_prioritization=True,
            ignore_emergency_conflicts=False  # Regular agents must respect emergency paths
        )
        
        return self.solve_cbs(request)
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get enhanced CBS performance statistics.
        
        Returns:
            Dict: Performance metrics including emergency/regular breakdown
        """
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        emergency_percentage = (self.emergency_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        avg_conflicts_per_request = (self.total_conflicts_detected / self.total_requests) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'emergency_requests': self.emergency_requests,
            'regular_requests': self.regular_requests,
            'success_rate_percent': success_rate,
            'emergency_percentage': emergency_percentage,
            'total_conflicts_detected': self.total_conflicts_detected,
            'total_conflicts_resolved': self.total_conflicts_resolved,
            'emergency_conflicts_detected': self.emergency_conflicts_detected,
            'avg_conflicts_per_request': avg_conflicts_per_request
        }
    
    def validate_solution(self, solution: Dict[int, List[Tuple[int, int]]], 
                         agent_type: AgentType, layer: LayerType, start_time: int) -> Tuple[bool, List[str]]:
        """
        Validate a CBS solution for correctness considering agent type.
        
        Args:
            solution: Solution to validate
            agent_type: Type of agents in solution
            layer: Layer the solution applies to
            start_time: Starting time step
            
        Returns:
            Tuple: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for conflicts
        regular_conflicts, emergency_conflicts = self.detect_conflicts(
            solution, agent_type, layer, start_time
        )
        
        if regular_conflicts:
            issues.extend([f"Regular conflict: {str(conflict)}" for conflict in regular_conflicts])
        
        if emergency_conflicts and agent_type == AgentType.REGULAR:
            issues.extend([f"Emergency conflict: {str(conflict)}" for conflict in emergency_conflicts])
        
        # Validate each path individually
        for agent_id, path in solution.items():
            time_steps = list(range(start_time, start_time + len(path)))
            
            if not self.pathfinder.validate_path(path, time_steps, layer, agent_id, agent_type):
                issues.append(f"Invalid path for agent {agent_id}")
        
        return len(issues) == 0, issues
    
    def export_cbs_config(self) -> Dict[str, Union[int, float]]:
        """
        Export current enhanced CBS configuration.
        
        Returns:
            Dict: Configuration parameters
        """
        pathfinder_config = self.pathfinder.export_pathfinding_config()
        
        return {
            **pathfinder_config,
            'cbs_total_requests': self.total_requests,
            'cbs_successful_requests': self.successful_requests,
            'cbs_emergency_requests': self.emergency_requests,
            'cbs_regular_requests': self.regular_requests,
            'cbs_conflicts_detected': self.total_conflicts_detected,
            'cbs_conflicts_resolved': self.total_conflicts_resolved,
            'cbs_emergency_conflicts_detected': self.emergency_conflicts_detected
        }
    
    def reset_stats(self) -> None:
        """Reset CBS performance statistics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.emergency_requests = 0
        self.regular_requests = 0
        self.total_conflicts_detected = 0
        self.total_conflicts_resolved = 0
        self.emergency_conflicts_detected = 0
    
    def analyze_emergency_impact(self, layer: LayerType, time_range: Tuple[int, int]) -> Dict[str, Union[int, float]]:
        """
        Analyze the impact of emergency agent reservations on available airspace.
        
        Args:
            layer: Layer to analyze
            time_range: (start_time, end_time) for analysis
            
        Returns:
            Dict: Analysis results including occupancy rates and blocked positions
        """
        start_time, end_time = time_range
        total_positions = self.grid_system.width * self.grid_system.height * (end_time - start_time)
        
        # Count emergency reservations in the time range
        emergency_occupied = 0
        blocked_positions = set()
        
        for (res_layer, x, y, t), agent_id in self.grid_system.emergency_reservations.occupancy_map.items():
            if res_layer == layer and start_time <= t < end_time:
                emergency_occupied += 1
                blocked_positions.add((x, y, t))
        
        occupancy_rate = (emergency_occupied / total_positions * 100) if total_positions > 0 else 0
        
        # Count unique spatial positions blocked
        unique_spatial_positions = len(set((x, y) for x, y, t in blocked_positions))
        spatial_coverage = (unique_spatial_positions / (self.grid_system.width * self.grid_system.height) * 100)
        
        return {
            'layer': layer.name,
            'time_range': time_range,
            'total_positions': total_positions,
            'emergency_occupied_positions': emergency_occupied,
            'occupancy_rate_percent': occupancy_rate,
            'unique_spatial_positions_blocked': unique_spatial_positions,
            'spatial_coverage_percent': spatial_coverage,
            'emergency_agents_in_layer': len(self.grid_system.emergency_reservations.get_emergency_agents_in_layer(layer))
        }
    
    def get_conflict_statistics(self, conflicts: List[Conflict]) -> Dict[str, int]:
        """
        Get detailed statistics about conflicts.
        
        Args:
            conflicts: List of conflicts to analyze
            
        Returns:
            Dict: Conflict statistics by type
        """
        stats = {
            'total_conflicts': len(conflicts),
            'vertex_conflicts': 0,
            'edge_conflicts': 0,
            'emergency_conflicts': 0,
            'following_conflicts': 0,
            'unique_agents_involved': set(),
            'unique_emergency_agents_involved': set()
        }
        
        for conflict in conflicts:
            stats['unique_agents_involved'].add(conflict.agent1)
            if conflict.agent2 is not None:
                stats['unique_agents_involved'].add(conflict.agent2)
            
            if conflict.type == ConflictType.VERTEX:
                stats['vertex_conflicts'] += 1
            elif conflict.type == ConflictType.EDGE:
                stats['edge_conflicts'] += 1
            elif conflict.type == ConflictType.EMERGENCY:
                stats['emergency_conflicts'] += 1
                if conflict.emergency_agent_id:
                    stats['unique_emergency_agents_involved'].add(conflict.emergency_agent_id)
            elif conflict.type == ConflictType.FOLLOWING:
                stats['following_conflicts'] += 1
        
        # Convert sets to counts
        stats['unique_agents_count'] = len(stats['unique_agents_involved'])
        stats['unique_emergency_agents_count'] = len(stats['unique_emergency_agents_involved'])
        del stats['unique_agents_involved']
        del stats['unique_emergency_agents_involved']
        
        return stats
    
    def export_solution_summary(self, response: CBSResponse) -> Dict[str, Union[str, int, float, List]]:
        """
        Export a comprehensive summary of a CBS solution.
        
        Args:
            response: CBSResponse to summarize
            
        Returns:
            Dict: Detailed solution summary
        """
        # Calculate path lengths and costs
        path_lengths = {agent_id: len(path) for agent_id, path in response.solution.items()}
        avg_path_length = sum(path_lengths.values()) / len(path_lengths) if path_lengths else 0
        
        # Analyze conflicts
        regular_conflict_stats = self.get_conflict_statistics(response.regular_conflicts)
        emergency_conflict_stats = self.get_conflict_statistics(response.emergency_conflicts)
        
        return {
            'result': response.result.value,
            'agent_type': response.agent_type.value,
            'computation_time_seconds': response.computation_time,
            'iterations_used': response.iterations_used,
            'conflicts_resolved': response.conflicts_resolved,
            'total_cost': response.total_cost,
            'successful_agents': len(response.successful_agents),
            'failed_agents': len(response.failed_agents),
            'success_rate_percent': (len(response.successful_agents) / (len(response.successful_agents) + len(response.failed_agents)) * 100) if (response.successful_agents or response.failed_agents) else 0,
            'path_lengths': path_lengths,
            'average_path_length': avg_path_length,
            'regular_conflict_stats': regular_conflict_stats,
            'emergency_conflict_stats': emergency_conflict_stats,
            'message': response.message
        }
    
    def create_cbs_request_from_grid_agents(self, layer: LayerType, agent_type: AgentType, 
                                           start_time: int, **kwargs) -> CBSRequest:
        """
        Create a CBS request from agents stored in the grid system.
        
        Args:
            layer: Layer to process
            agent_type: Type of agents to process
            start_time: Starting time step
            **kwargs: Additional CBS request parameters
            
        Returns:
            CBSRequest: Ready-to-use CBS request
        """
        if agent_type == AgentType.EMERGENCY:
            agents = self.grid_system.get_emergency_agents_for_processing(layer)
        else:
            agents = self.grid_system.get_regular_agents_for_processing(layer)
        
        # Set default parameters based on agent type
        defaults = {
            'max_time_steps': 200 if agent_type == AgentType.EMERGENCY else 100,
            'max_iterations': 800 if agent_type == AgentType.EMERGENCY else 400,
            'timeout_seconds': 60.0 if agent_type == AgentType.EMERGENCY else 30.0,
            'enable_prioritization': True,
            'ignore_emergency_conflicts': (agent_type == AgentType.EMERGENCY)
        }
        
        # Override with provided kwargs
        defaults.update(kwargs)
        
        return CBSRequest(
            agents=agents,
            layer=layer,
            agent_type=agent_type,
            start_time=start_time,
            **defaults
        )
    
    def process_layer_with_priority(self, layer: LayerType, start_time: int) -> Tuple[CBSResponse, CBSResponse]:
        """
        Process a layer with emergency priority: emergency agents first, then regular agents.
        
        Args:
            layer: Layer to process
            start_time: Starting time step
            
        Returns:
            Tuple: (emergency_response, regular_response)
        """
        # Phase 1: Process emergency agents
        emergency_response = CBSResponse(
            result=CBSResult.SUCCESS,
            agent_type=AgentType.EMERGENCY,
            solution={},
            time_steps={},
            total_cost=0.0,
            computation_time=0.0,
            iterations_used=0,
            conflicts_resolved=0,
            emergency_conflicts=[],
            regular_conflicts=[],
            failed_agents=[],
            successful_agents=[],
            message="No emergency agents to process"
        )
        
        emergency_agents = self.grid_system.get_emergency_agents_for_processing(layer)
        if emergency_agents:
            emergency_response = self.solve_emergency_agents(layer, emergency_agents, start_time)
            
            # If successful, assign emergency paths to grid system
            if emergency_response.result == CBSResult.SUCCESS:
                self.grid_system.assign_emergency_paths(layer, emergency_response.solution, start_time)
        
        # Phase 2: Process regular agents (with emergency paths as dynamic obstacles)
        regular_response = CBSResponse(
            result=CBSResult.SUCCESS,
            agent_type=AgentType.REGULAR,
            solution={},
            time_steps={},
            total_cost=0.0,
            computation_time=0.0,
            iterations_used=0,
            conflicts_resolved=0,
            emergency_conflicts=[],
            regular_conflicts=[],
            failed_agents=[],
            successful_agents=[],
            message="No regular agents to process"
        )
        
        regular_agents = self.grid_system.get_regular_agents_for_processing(layer)
        if regular_agents:
            regular_response = self.solve_regular_agents(layer, regular_agents, start_time)
            
            # Update grid system with successful regular agent paths
            if regular_response.solution:
                self.grid_system.update_regular_agent_paths(layer, regular_response.solution)
        
        return emergency_response, regular_response