"""
Double-Layer Airspace System for UrbanAirspaceSim
Implements aviation-inspired flight level assignment with dual-layer coordination
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy
import time
import threading
import logging

# Import from existing modules
from grid_system import Position, Agent, AgentType, GridSystem, PathStatus
from astar_pathfinding import AStarPathfinder, PathfindingConfig, PathfindingResult
from cbs_module import ConflictBasedSearch, CBSResult
from auction_system import AuctionSystem, AuctionResult

class FlightLevel(Enum):
    """Flight levels for double-layer airspace"""
    GROUND = 0      # Ground level - takeoff position
    LOWER = 1       # Lower airspace layer
    HIGHER = 2      # Higher airspace layer

class FlightDirection(Enum):
    """Primary flight directions for layer assignment"""
    EASTWARD = "eastward"     # Positive x-direction
    WESTWARD = "westward"     # Negative x-direction  
    NORTHWARD = "northward"   # Positive y-direction
    SOUTHWARD = "southward"   # Negative y-direction
    STATIONARY = "stationary" # No primary direction

@dataclass
class LayerAssignment:
    """Assignment of agent to specific flight level"""
    agent_id: int
    assigned_layer: FlightLevel
    primary_direction: FlightDirection
    rationale: str  # Explanation for assignment decision
    
    def __hash__(self):
        return hash(self.agent_id)

@dataclass
class VerticalMovementPhase:
    """Represents a phase in the vertical movement protocol"""
    time_step: int
    source_layer: FlightLevel
    target_layer: FlightLevel
    description: str

class FlightLevelAssignmentSystem:
    """
    Assigns aircraft to flight levels based on aviation rules
    """
    
    def __init__(self):
        self.assignment_rules = {
            # Primary rules based on dominant direction
            FlightDirection.EASTWARD: FlightLevel.LOWER,
            FlightDirection.WESTWARD: FlightLevel.HIGHER,
            FlightDirection.NORTHWARD: FlightLevel.HIGHER,
            FlightDirection.SOUTHWARD: FlightLevel.LOWER,
            FlightDirection.STATIONARY: FlightLevel.LOWER  # Default to lower
        }
        
        self.logger = logging.getLogger(__name__ + ".FlightLevelAssignment")
    
    def determine_primary_direction(self, agent: Agent) -> FlightDirection:
        """
        Determine agent's primary flight direction based on start and goal positions
        
        Args:
            agent: Agent to analyze
            
        Returns:
            Primary flight direction
        """
        start_x, start_y = agent.start
        goal_x, goal_y = agent.goal
        
        dx = goal_x - start_x
        dy = goal_y - start_y
        
        # Handle stationary case
        if dx == 0 and dy == 0:
            return FlightDirection.STATIONARY
        
        # Determine dominant direction
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        if abs_dx > abs_dy:
            # Primary movement is horizontal
            if dx > 0:
                return FlightDirection.EASTWARD
            else:
                return FlightDirection.WESTWARD
        elif abs_dy > abs_dx:
            # Primary movement is vertical
            if dy > 0:
                return FlightDirection.NORTHWARD
            else:
                return FlightDirection.SOUTHWARD
        else:
            # Equal horizontal and vertical movement
            # Prioritize horizontal movement for tie-breaking
            if dx > 0:
                return FlightDirection.EASTWARD
            elif dx < 0:
                return FlightDirection.WESTWARD
            elif dy > 0:
                return FlightDirection.NORTHWARD
            else:
                return FlightDirection.SOUTHWARD
    
    def assign_flight_level(self, agent: Agent, 
                           emergency_priority: bool = False) -> LayerAssignment:
        """
        Assign flight level to an agent based on aviation rules
        
        Args:
            agent: Agent to assign
            emergency_priority: Whether emergency agents get special handling
            
        Returns:
            Layer assignment with rationale
        """
        primary_direction = self.determine_primary_direction(agent)
        
        # Emergency agents can override normal rules if needed
        if agent.is_emergency() and emergency_priority:
            # For now, emergency agents follow same rules but get priority within layer
            assigned_layer = self.assignment_rules[primary_direction]
            rationale = f"Emergency agent: {primary_direction.value} → {assigned_layer.name}"
        else:
            assigned_layer = self.assignment_rules[primary_direction]
            rationale = f"Standard assignment: {primary_direction.value} → {assigned_layer.name}"
        
        return LayerAssignment(
            agent_id=agent.id,
            assigned_layer=assigned_layer,
            primary_direction=primary_direction,
            rationale=rationale
        )
    
    def assign_agents_to_layers(self, agents: List[Agent]) -> Dict[FlightLevel, List[Agent]]:
        """
        Assign all agents to appropriate flight levels
        
        Args:
            agents: List of agents to assign
            
        Returns:
            Dictionary mapping flight levels to lists of agents
        """
        layer_assignments = {}
        agents_by_layer = defaultdict(list)
        
        for agent in agents:
            assignment = self.assign_flight_level(agent)
            layer_assignments[agent.id] = assignment
            agents_by_layer[assignment.assigned_layer].append(agent)
        
        # Convert to regular dict for cleaner interface
        return {
            FlightLevel.GROUND: [],  # No agents stay on ground permanently
            FlightLevel.LOWER: agents_by_layer[FlightLevel.LOWER],
            FlightLevel.HIGHER: agents_by_layer[FlightLevel.HIGHER]
        }

class VerticalMovementProtocol:
    """
    Handles the synchronized takeoff and vertical movement phases
    """
    
    def __init__(self):
        # Standard vertical movement sequence
        self.movement_phases = [
            VerticalMovementPhase(0, FlightLevel.GROUND, FlightLevel.GROUND,
                                "All aircraft start on ground"),
            VerticalMovementPhase(1, FlightLevel.GROUND, FlightLevel.LOWER,
                                "All aircraft move to lower layer"),
            VerticalMovementPhase(2, FlightLevel.LOWER, FlightLevel.HIGHER,
                                "Higher-layer aircraft transition up"),
            # Horizontal movement begins at t=3
        ]
        
        self.horizontal_movement_start_time = 3
        
    def generate_vertical_movement_path(self, agent: Agent, 
                                      assigned_layer: FlightLevel) -> List[Position]:
        """
        Generate the mandatory vertical movement path for an agent
        
        Args:
            agent: Agent to generate path for
            assigned_layer: Agent's assigned final layer
            
        Returns:
            List of positions for vertical movement phases
        """
        start_x, start_y = agent.start
        vertical_path = []
        
        if assigned_layer == FlightLevel.LOWER:
            # Lower layer agents: Ground -> Lower
            vertical_path = [
                Position(start_x, start_y, 0),  # t=0: Ground
                Position(start_x, start_y, 1),  # t=1: Lower layer
                Position(start_x, start_y, 2),  # t=2: Remain in lower layer
            ]
        
        elif assigned_layer == FlightLevel.HIGHER:
            # Higher layer agents: Ground -> Lower -> Higher
            vertical_path = [
                Position(start_x, start_y, 0),  # t=0: Ground
                Position(start_x, start_y, 1),  # t=1: Lower layer (transition)
                Position(start_x, start_y, 2),  # t=2: Higher layer
            ]
        
        return vertical_path
    
    def combine_vertical_and_horizontal_paths(self, 
                                            vertical_path: List[Position],
                                            horizontal_path: List[Position]) -> List[Position]:
        """
        Combine vertical movement phases with horizontal pathfinding results
        
        Args:
            vertical_path: Mandatory vertical movement positions
            horizontal_path: Pathfinding result for horizontal movement
            
        Returns:
            Complete path including vertical and horizontal phases
        """
        if not horizontal_path:
            return vertical_path
        
        # Adjust horizontal path timestamps to start after vertical movement
        adjusted_horizontal_path = []
        
        for i, pos in enumerate(horizontal_path):
            new_time = self.horizontal_movement_start_time + i
            adjusted_pos = Position(pos.x, pos.y, new_time)
            adjusted_horizontal_path.append(adjusted_pos)
        
        # Combine paths, removing duplicate transition point
        combined_path = vertical_path.copy()
        
        # Ensure smooth transition
        if (adjusted_horizontal_path and 
            vertical_path and
            vertical_path[-1].spatial_position() == adjusted_horizontal_path[0].spatial_position()):
            # Remove duplicate position at transition
            combined_path.extend(adjusted_horizontal_path[1:])
        else:
            combined_path.extend(adjusted_horizontal_path)
        
        return combined_path

class TwoLayerGridCoordinator:
    """
    Coordinates two separate grid systems for double-layer airspace
    """
    
    def __init__(self, width: int, height: int, max_time: int = 100):
        """
        Initialize coordinator with two grid layers
        
        Args:
            width: Grid width
            height: Grid height
            max_time: Maximum time steps
        """
        self.width = width
        self.height = height
        self.max_time = max_time
        
        # Create separate grid instances for each layer
        self.lower_grid = GridSystem(width, height, max_time)
        self.higher_grid = GridSystem(width, height, max_time)
        
        # Track agent-to-layer mappings
        self.agent_layer_assignments: Dict[int, FlightLevel] = {}
        
        # Components for each layer
        self.lower_pathfinder: Optional[AStarPathfinder] = None
        self.higher_pathfinder: Optional[AStarPathfinder] = None
        self.lower_cbs: Optional[ConflictBasedSearch] = None
        self.higher_cbs: Optional[ConflictBasedSearch] = None
        self.lower_auction: Optional[AuctionSystem] = None
        self.higher_auction: Optional[AuctionSystem] = None
        
        self.logger = logging.getLogger(__name__ + ".TwoLayerCoordinator")
    
    def initialize_layer_components(self, 
                                   pathfinder_config: PathfindingConfig = None):
        """Initialize pathfinding components for both layers"""
        
        # Create pathfinders
        self.lower_pathfinder = AStarPathfinder(self.lower_grid, pathfinder_config)
        self.higher_pathfinder = AStarPathfinder(self.higher_grid, pathfinder_config)
        
        # Create CBS solvers
        self.lower_cbs = ConflictBasedSearch(self.lower_grid, self.lower_pathfinder)
        self.higher_cbs = ConflictBasedSearch(self.higher_grid, self.higher_pathfinder)
        
        # Create auction systems
        self.lower_auction = AuctionSystem(self.lower_grid, self.lower_cbs, self.lower_pathfinder)
        self.higher_auction = AuctionSystem(self.higher_grid, self.higher_cbs, self.higher_pathfinder)
    
    def get_grid_for_layer(self, layer: FlightLevel) -> Optional[GridSystem]:
        """Get the grid system for a specific layer"""
        if layer == FlightLevel.LOWER:
            return self.lower_grid
        elif layer == FlightLevel.HIGHER:
            return self.higher_grid
        else:
            return None  # Ground level doesn't have persistent grid
    
    def get_pathfinder_for_layer(self, layer: FlightLevel) -> Optional[AStarPathfinder]:
        """Get the pathfinder for a specific layer"""
        if layer == FlightLevel.LOWER:
            return self.lower_pathfinder
        elif layer == FlightLevel.HIGHER:
            return self.higher_pathfinder
        else:
            return None
    
    def get_cbs_for_layer(self, layer: FlightLevel) -> Optional[ConflictBasedSearch]:
        """Get the CBS solver for a specific layer"""
        if layer == FlightLevel.LOWER:
            return self.lower_cbs
        elif layer == FlightLevel.HIGHER:
            return self.higher_cbs
        else:
            return None
    
    def get_auction_for_layer(self, layer: FlightLevel) -> Optional[AuctionSystem]:
        """Get the auction system for a specific layer"""
        if layer == FlightLevel.LOWER:
            return self.lower_auction
        elif layer == FlightLevel.HIGHER:
            return self.higher_auction
        else:
            return None
    
    def add_agents_to_layer(self, agents: List[Agent], layer: FlightLevel) -> bool:
        """Add agents to the specified layer's grid"""
        grid = self.get_grid_for_layer(layer)
        if not grid:
            return False
        
        success = True
        for agent in agents:
            if grid.add_agent(agent):
                self.agent_layer_assignments[agent.id] = layer
            else:
                success = False
                self.logger.warning(f"Failed to add agent {agent.id} to layer {layer}")
        
        return success
    
    def add_static_obstacles_to_all_layers(self, obstacles: List[Tuple[int, int]]) -> bool:
        """Add static obstacles to both layers"""
        success = True
        
        for x, y in obstacles:
            if not self.lower_grid.add_static_obstacle(x, y):
                success = False
            if not self.higher_grid.add_static_obstacle(x, y):
                success = False
        
        return success
    
    def get_system_state(self) -> Dict:
        """Get comprehensive state of both layers"""
        return {
            'lower_layer': self.lower_grid.get_system_state(),
            'higher_layer': self.higher_grid.get_system_state(),
            'agent_assignments': dict(self.agent_layer_assignments),
            'total_agents': len(self.agent_layer_assignments)
        }

@dataclass
class DoubleLayerResult:
    """Result of double-layer airspace coordination"""
    # Core results
    success: bool = False
    agent_paths: Dict[int, List[Position]] = field(default_factory=dict)
    layer_assignments: Dict[int, FlightLevel] = field(default_factory=dict)
    
    # Layer-specific results
    lower_layer_result: Optional[Union[CBSResult, AuctionResult]] = None
    higher_layer_result: Optional[Union[CBSResult, AuctionResult]] = None
    
    # Performance metrics
    total_duration: float = 0.0
    assignment_time: float = 0.0
    lower_layer_time: float = 0.0
    higher_layer_time: float = 0.0
    integration_time: float = 0.0
    
    # Statistics
    agents_by_layer: Dict[FlightLevel, int] = field(default_factory=dict)
    conflicts_eliminated_by_layering: int = 0
    
    # Failure information
    failure_reason: str = ""
    unassigned_agents: List[int] = field(default_factory=list)

class DoubleLayerAirspaceSystem:
    """
    Master coordination system for double-layer airspace management
    """
    
    def __init__(self, width: int, height: int, max_time: int = 100):
        """
        Initialize the double-layer airspace system
        
        Args:
            width: Airspace width
            height: Airspace height
            max_time: Maximum simulation time
        """
        self.width = width
        self.height = height
        self.max_time = max_time
        
        # Core components
        self.flight_level_system = FlightLevelAssignmentSystem()
        self.vertical_movement_protocol = VerticalMovementProtocol()
        self.grid_coordinator = TwoLayerGridCoordinator(width, height, max_time)
        
        # Configuration
        self.use_cbs_first = True  # Try CBS before auctions
        self.debug_mode = False
        
        self.logger = logging.getLogger(__name__ + ".DoubleLayerSystem")
    
    def initialize(self, pathfinder_config: PathfindingConfig = None,
                   static_obstacles: List[Tuple[int, int]] = None):
        """
        Initialize the system with configuration
        
        Args:
            pathfinder_config: Configuration for pathfinding
            static_obstacles: Static obstacles to add to all layers
        """
        self.grid_coordinator.initialize_layer_components(pathfinder_config)
        
        if static_obstacles:
            self.grid_coordinator.add_static_obstacles_to_all_layers(static_obstacles)
    
    def coordinate_agents(self, agents: List[Agent]) -> DoubleLayerResult:
        """
        Main coordination method implementing the seven-step process
        
        Args:
            agents: List of agents to coordinate
            
        Returns:
            Comprehensive result of coordination
        """
        start_time = time.time()
        result = DoubleLayerResult()
        
        try:
            # Step 1: Assign flight levels
            assignment_start = time.time()
            agents_by_layer = self.flight_level_system.assign_agents_to_layers(agents)
            result.assignment_time = time.time() - assignment_start
            
            if self.debug_mode:
                self.logger.info(f"Layer assignments: Lower={len(agents_by_layer[FlightLevel.LOWER])}, "
                               f"Higher={len(agents_by_layer[FlightLevel.HIGHER])}")
            
            # Step 2: Separate agents by layer and add to grids
            for layer, layer_agents in agents_by_layer.items():
                if layer != FlightLevel.GROUND and layer_agents:
                    self.grid_coordinator.add_agents_to_layer(layer_agents, layer)
                    result.agents_by_layer[layer] = len(layer_agents)
                    
                    # Track assignments
                    for agent in layer_agents:
                        result.layer_assignments[agent.id] = layer
            
            # Step 3 & 4: Handle emergency agents and create constraints
            emergency_constraints_lower = self._handle_emergency_agents(FlightLevel.LOWER)
            emergency_constraints_higher = self._handle_emergency_agents(FlightLevel.HIGHER)
            
            # Step 5: Run CBS or auctions for regular agents in each layer
            lower_start = time.time()
            result.lower_layer_result = self._coordinate_layer(
                FlightLevel.LOWER, emergency_constraints_lower
            )
            result.lower_layer_time = time.time() - lower_start
            
            higher_start = time.time()
            result.higher_layer_result = self._coordinate_layer(
                FlightLevel.HIGHER, emergency_constraints_higher
            )
            result.higher_layer_time = time.time() - higher_start
            
            # Step 6: Integrate results with vertical movement
            integration_start = time.time()
            success = self._integrate_results(result, agents_by_layer)
            result.integration_time = time.time() - integration_start
            
            result.success = success
            
        except Exception as e:
            result.failure_reason = f"Coordination failed: {e}"
            self.logger.error(f"Double-layer coordination error: {e}")
        
        finally:
            result.total_duration = time.time() - start_time
        
        return result
    
    def _handle_emergency_agents(self, layer: FlightLevel) -> Dict[int, List[Position]]:
        """
        Handle emergency agents in a layer and return their paths as constraints
        
        Args:
            layer: Flight layer to handle
            
        Returns:
            Dictionary of emergency agent paths
        """
        grid = self.grid_coordinator.get_grid_for_layer(layer)
        pathfinder = self.grid_coordinator.get_pathfinder_for_layer(layer)
        
        if not grid or not pathfinder:
            return {}
        
        emergency_agents = grid.get_agents_by_type(AgentType.EMERGENCY)
        emergency_paths = {}
        
        for agent in emergency_agents:
            # Generate horizontal movement path (vertical will be added later)
            result = pathfinder.find_path(agent, 
                                        start_time=0)  # Will be adjusted for vertical movement
            
            if result.success:
                # Set path in grid
                if grid.set_agent_path(agent.id, result.path):
                    emergency_paths[agent.id] = result.path
                    
                    if self.debug_mode:
                        self.logger.info(f"Emergency agent {agent.id} path set in {layer.name} layer")
        
        return emergency_paths
    
    def _coordinate_layer(self, layer: FlightLevel, 
                         emergency_constraints: Dict[int, List[Position]]) -> Optional[Union[CBSResult, AuctionResult]]:
        """
        Coordinate agents within a single layer
        
        Args:
            layer: Flight layer to coordinate
            emergency_constraints: Emergency agent paths to avoid
            
        Returns:
            Result of coordination (CBS or Auction)
        """
        grid = self.grid_coordinator.get_grid_for_layer(layer)
        cbs = self.grid_coordinator.get_cbs_for_layer(layer)
        auction = self.grid_coordinator.get_auction_for_layer(layer)
        
        if not all([grid, cbs, auction]):
            return None
        
        # Get non-emergency agents that need paths
        regular_agents = grid.get_unassigned_agents(AgentType.NON_EMERGENCY)
        
        if not regular_agents:
            return None  # No agents to coordinate
        
        if self.use_cbs_first:
            # Try CBS first
            cbs_result = cbs.solve(regular_agents)
            
            if cbs_result.success:
                # CBS succeeded - set paths in grid
                for agent_id, path in cbs_result.paths.items():
                    grid.set_agent_path(agent_id, path)
                
                if self.debug_mode:
                    self.logger.info(f"CBS succeeded for {len(cbs_result.paths)} agents in {layer.name}")
                
                return cbs_result
            
            elif cbs_result.should_trigger_auction and cbs_result.auction_candidates:
                # CBS failed, try auction for remaining agents
                auction_agents = [agent for agent in regular_agents 
                                if agent.id in cbs_result.auction_candidates]
                
                if auction_agents:
                    auction_result = auction.run_auction(
                        auction_agents,
                        dict(cbs_result.conflict_density),
                        emergency_constraints
                    )
                    
                    # Set successful auction paths in grid
                    if auction_result.success:
                        for agent_id, path in auction_result.final_winners.items():
                            grid.set_agent_path(agent_id, path)
                    
                    if self.debug_mode:
                        self.logger.info(f"Auction completed for {len(auction_result.final_winners)} agents in {layer.name}")
                    
                    return auction_result
        
        else:
            # Use auction directly
            # Extract conflict density (would need to be calculated)
            conflict_density = {}  # Placeholder
            
            auction_result = auction.run_auction(
                regular_agents,
                conflict_density,
                emergency_constraints
            )
            
            # Set successful auction paths in grid
            if auction_result.success:
                for agent_id, path in auction_result.final_winners.items():
                    grid.set_agent_path(agent_id, path)
            
            return auction_result
        
        return None
    
    def _integrate_results(self, result: DoubleLayerResult, 
                          agents_by_layer: Dict[FlightLevel, List[Agent]]) -> bool:
        """
        Integrate layer results with vertical movement protocol
        
        Args:
            result: Result object to populate
            agents_by_layer: Agents organized by layer
            
        Returns:
            True if integration successful
        """
        success = True
        
        for layer in [FlightLevel.LOWER, FlightLevel.HIGHER]:
            grid = self.grid_coordinator.get_grid_for_layer(layer)
            if not grid:
                continue
            
            layer_agents = agents_by_layer[layer]
            
            for agent in layer_agents:
                # Get horizontal path from grid (if any)
                horizontal_path = []
                if agent.has_path():
                    horizontal_path = agent.path
                
                # Generate vertical movement path
                vertical_path = self.vertical_movement_protocol.generate_vertical_movement_path(
                    agent, layer
                )
                
                # Combine vertical and horizontal paths
                complete_path = self.vertical_movement_protocol.combine_vertical_and_horizontal_paths(
                    vertical_path, horizontal_path
                )
                
                if complete_path:
                    result.agent_paths[agent.id] = complete_path
                else:
                    result.unassigned_agents.append(agent.id)
                    success = False
        
        return success
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            'grid_coordinator_state': self.grid_coordinator.get_system_state(),
            'lower_layer_stats': (self.grid_coordinator.lower_cbs.get_statistics() 
                                if self.grid_coordinator.lower_cbs else {}),
            'higher_layer_stats': (self.grid_coordinator.higher_cbs.get_statistics() 
                                 if self.grid_coordinator.higher_cbs else {}),
            'lower_auction_stats': (self.grid_coordinator.lower_auction.get_statistics() 
                                  if self.grid_coordinator.lower_auction else {}),
            'higher_auction_stats': (self.grid_coordinator.higher_auction.get_statistics() 
                                   if self.grid_coordinator.higher_auction else {})
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode across all components"""
        self.debug_mode = enabled
        
        if self.grid_coordinator.lower_cbs:
            self.grid_coordinator.lower_cbs.set_debug_mode(enabled)
        if self.grid_coordinator.higher_cbs:
            self.grid_coordinator.higher_cbs.set_debug_mode(enabled)
        if self.grid_coordinator.lower_auction:
            self.grid_coordinator.lower_auction.set_debug_mode(enabled)
        if self.grid_coordinator.higher_auction:
            self.grid_coordinator.higher_auction.set_debug_mode(enabled)

# Utility functions for integration and testing

def create_double_layer_system(width: int, height: int, max_time: int = 100,
                              pathfinder_config: PathfindingConfig = None,
                              static_obstacles: List[Tuple[int, int]] = None) -> DoubleLayerAirspaceSystem:
    """
    Create and initialize a complete double-layer airspace system
    
    Args:
        width: Airspace width
        height: Airspace height  
        max_time: Maximum simulation time
        pathfinder_config: Pathfinding configuration
        static_obstacles: Static obstacles to add
        
    Returns:
        Initialized double-layer system
    """
    system = DoubleLayerAirspaceSystem(width, height, max_time)
    system.initialize(pathfinder_config, static_obstacles)
    return system

def analyze_layer_separation_benefit(original_conflicts: int, 
                                   lower_layer_conflicts: int,
                                   higher_layer_conflicts: int) -> Dict:
    """
    Analyze the benefit of layer separation in conflict reduction
    
    Args:
        original_conflicts: Conflicts in single-layer system
        lower_layer_conflicts: Conflicts in lower layer
        higher_layer_conflicts: Conflicts in higher layer
        
    Returns:
        Analysis of separation benefits
    """
    total_layered_conflicts = lower_layer_conflicts + higher_layer_conflicts
    conflicts_eliminated = max(0, original_conflicts - total_layered_conflicts)
    
    reduction_percentage = (conflicts_eliminated / max(original_conflicts, 1)) * 100
    
    return {
        'original_conflicts': original_conflicts,
        'lower_layer_conflicts': lower_layer_conflicts,
        'higher_layer_conflicts': higher_layer_conflicts,
        'total_layered_conflicts': total_layered_conflicts,
        'conflicts_eliminated': conflicts_eliminated,
        'reduction_percentage': reduction_percentage,
        'separation_effective': conflicts_eliminated > 0
    }

def validate_double_layer_paths(result: DoubleLayerResult) -> Dict:
    """
    Validate that double-layer paths follow the vertical movement protocol
    
    Args:
        result: Double-layer coordination result
        
    Returns:
        Validation results
    """
    validation_results = {
        'valid_paths': [],
        'invalid_paths': [],
        'protocol_violations': []
    }
    
    for agent_id, path in result.agent_paths.items():
        if not path:
            validation_results['invalid_paths'].append(agent_id)
            continue
        
        layer = result.layer_assignments.get(agent_id)
        path_valid = True
        violations = []
        
        # Check vertical movement protocol compliance
        if len(path) < 3:
            violations.append("Path too short for vertical movement protocol")
            path_valid = False
        else:
            # Check t=0 (ground)
            if path[0].t != 0:
                violations.append("Path doesn't start at t=0")
                path_valid = False
            
            # Check t=1 (lower layer)
            if path[1].t != 1:
                violations.append("Path doesn't transition at t=1")
                path_valid = False
            
            # Check t=2 based on layer assignment
            if layer == FlightLevel.HIGHER and len(path) > 2:
                if path[2].t != 2:
                    violations.append("Higher layer path doesn't transition at t=2")
                    path_valid = False
        
        if path_valid:
            validation_results['valid_paths'].append(agent_id)
        else:
            validation_results['invalid_paths'].append(agent_id)
            validation_results['protocol_violations'].extend(violations)
    
    return validation_results