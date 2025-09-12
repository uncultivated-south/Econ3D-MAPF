"""
Enhanced Multilayer Grid System with Emergency Agent Priority
Handles emergency agent route pre-assignment and dynamic obstacle management
for regular agent processing in urban airspace simulation.
"""

from typing import Dict, List, Tuple, Set, Optional, Union
from enum import Enum
import numpy as np

class LayerType(Enum):
    """Enumeration for airspace layers"""
    LOWER = 0
    UPPER = 1

class AgentState(Enum):
    """Enumeration for agent processing states"""
    GROUND = 0          # Before entering airspace
    LOWER_LAYER = 1     # Operating in lower layer
    UPPER_LAYER = 2     # Operating in upper layer  
    CLIMBING = 3        # Transitioning from lower to upper (t=2 only)
    DESCENDING = 4      # Transitioning from upper to lower
    LANDED = 5          # Exited the system

class AgentType(Enum):
    """Enumeration for agent types"""
    EMERGENCY = 0       # Emergency agents with highest priority
    REGULAR = 1         # Regular agents subject to budgets and auctions

class ProcessingPhase(Enum):
    """Enumeration for processing phases"""
    EMERGENCY_ASSIGNMENT = 0    # Emergency agents get direct path assignment
    REGULAR_CBS = 1            # Regular agents processed with CBS
    REGULAR_AUCTION = 2        # Regular agents processed with auction (if CBS fails)

class EmergencyPathReservation:
    """Container for emergency agent path reservations"""
    
    def __init__(self):
        # agent_id -> {layer, path, time_steps}
        self.reservations: Dict[int, Dict] = {}
        # (layer, x, y, time) -> agent_id mapping for quick lookup
        self.occupancy_map: Dict[Tuple[LayerType, int, int, int], int] = {}
        # layer -> set of emergency agent IDs
        self.agents_by_layer: Dict[LayerType, Set[int]] = {
            LayerType.LOWER: set(),
            LayerType.UPPER: set()
        }
    
    def add_reservation(self, agent_id: int, layer: LayerType, 
                       path: List[Tuple[int, int]], time_steps: List[int]) -> bool:
        """
        Add emergency agent path reservation.
        
        Args:
            agent_id: Emergency agent ID
            layer: Layer for the path
            path: List of (x, y) positions
            time_steps: Corresponding time steps
            
        Returns:
            bool: True if reservation added successfully
        """
        if agent_id in self.reservations:
            return False
        
        if len(path) != len(time_steps):
            return False
        
        # Store reservation
        self.reservations[agent_id] = {
            'layer': layer,
            'path': path,
            'time_steps': time_steps
        }
        
        # Update occupancy map
        for pos, time_step in zip(path, time_steps):
            x, y = pos
            occupancy_key = (layer, x, y, time_step)
            self.occupancy_map[occupancy_key] = agent_id
        
        # Update layer tracking
        self.agents_by_layer[layer].add(agent_id)
        
        return True
    
    def is_position_reserved(self, layer: LayerType, position: Tuple[int, int], 
                           time_step: int) -> Tuple[bool, Optional[int]]:
        """
        Check if position is reserved by emergency agent.
        
        Args:
            layer: Layer to check
            position: Position (x, y) to check
            time_step: Time step to check
            
        Returns:
            Tuple: (is_reserved, agent_id if reserved)
        """
        x, y = position
        occupancy_key = (layer, x, y, time_step)
        
        if occupancy_key in self.occupancy_map:
            return True, self.occupancy_map[occupancy_key]
        return False, None
    
    def get_emergency_agents_in_layer(self, layer: LayerType) -> Set[int]:
        """Get all emergency agent IDs in a specific layer."""
        return self.agents_by_layer[layer].copy()
    
    def get_reservation(self, agent_id: int) -> Optional[Dict]:
        """Get reservation details for an agent."""
        return self.reservations.get(agent_id)
    
    def clear_reservations(self) -> None:
        """Clear all reservations."""
        self.reservations.clear()
        self.occupancy_map.clear()
        for layer in LayerType:
            self.agents_by_layer[layer].clear()

class EnhancedMultilayerGridSystem:
    """
    Enhanced multilayer grid system with emergency agent priority management.
    
    Attributes:
        width (int): Grid width
        height (int): Grid height
        max_time (int): Maximum simulation time steps
        current_time (int): Current simulation time
        emergency_agents (Dict[int, Dict]): Emergency agent storage
        regular_agents (Dict[int, Dict]): Regular agent storage
        grid_occupancy (Dict[LayerType, np.ndarray]): Base occupancy grids per layer
        emergency_reservations (EmergencyPathReservation): Emergency path reservations
        layer_assignments (Dict[int, LayerType]): Agent to layer assignments
        agent_states (Dict[int, AgentState]): Agent state tracking
        processing_phase (ProcessingPhase): Current processing phase
    """
    
    def __init__(self, width: int, height: int, max_time: int = 100):
        """
        Initialize the enhanced multilayer grid system.
        
        Args:
            width: Grid width (x-dimension)
            height: Grid height (y-dimension) 
            max_time: Maximum time steps for simulation
        """
        self.width = width
        self.height = height
        self.max_time = max_time
        self.current_time = 0
        
        # Separate storage for different agent types
        self.emergency_agents: Dict[int, Dict] = {}
        self.regular_agents: Dict[int, Dict] = {}
        
        # Base grid occupancy (static obstacles, infrastructure)
        self.grid_occupancy: Dict[LayerType, np.ndarray] = {
            LayerType.LOWER: np.zeros((max_time, width, height), dtype=bool),
            LayerType.UPPER: np.zeros((max_time, width, height), dtype=bool)
        }
        
        # Emergency agent path reservations (dynamic obstacles for regular agents)
        self.emergency_reservations = EmergencyPathReservation()
        
        # Agent management
        self.layer_assignments: Dict[int, LayerType] = {}
        self.agent_states: Dict[int, AgentState] = {}
        
        # Processing phase tracking
        self.processing_phase = ProcessingPhase.EMERGENCY_ASSIGNMENT
        
    def determine_layer_assignment(self, start: Tuple[int, int], goal: Tuple[int, int]) -> LayerType:
        """
        Determine layer assignment based on East/West track split convention.
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            LayerType: LOWER or UPPER layer assignment
            
        Raises:
            ValueError: If start and goal positions are identical
        """
        delta_x = goal[0] - start[0]
        delta_y = goal[1] - start[1]
        
        # Reject case where both deltas are zero
        if delta_x == 0 and delta_y == 0:
            raise ValueError("Start and goal positions cannot be identical")
        
        # Primary rule: x-direction increment
        if delta_x > 0:
            return LayerType.LOWER
        elif delta_x < 0:
            return LayerType.UPPER
        
        # Secondary rule: y-direction increment (when delta_x = 0)
        elif delta_y < 0:
            return LayerType.LOWER
        else:  # delta_y > 0
            return LayerType.UPPER
    
    def add_emergency_agent(self, agent_id: int, start: Tuple[int, int], 
                           goal: Tuple[int, int]) -> bool:
        """
        Add an emergency agent to the system.
        
        Args:
            agent_id: Unique agent identifier
            start: Starting position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            bool: True if agent added successfully
        """
        if agent_id in self.emergency_agents or agent_id in self.regular_agents:
            return False
        
        try:
            assigned_layer = self.determine_layer_assignment(start, goal)
        except ValueError:
            return False
        
        self.emergency_agents[agent_id] = {
            'start': start,
            'goal': goal,
            'layer': assigned_layer,
            'current_pos': None,
            'path': [],
            'agent_type': AgentType.EMERGENCY
        }
        
        self.layer_assignments[agent_id] = assigned_layer
        self.agent_states[agent_id] = AgentState.GROUND
        
        return True
    
    def add_regular_agent(self, agent_id: int, start: Tuple[int, int], goal: Tuple[int, int],
                         budget: float = 50.0, strategy: str = "balanced") -> bool:
        """
        Add a regular agent to the system.
        
        Args:
            agent_id: Unique agent identifier
            start: Starting position (x, y)
            goal: Goal position (x, y)
            budget: Agent's bidding budget (for auction system)
            strategy: Bidding strategy ("conservative", "aggressive", "balanced")
            
        Returns:
            bool: True if agent added successfully
        """
        if agent_id in self.emergency_agents or agent_id in self.regular_agents:
            return False
        
        try:
            assigned_layer = self.determine_layer_assignment(start, goal)
        except ValueError:
            return False
        
        self.regular_agents[agent_id] = {
            'start': start,
            'goal': goal,
            'layer': assigned_layer,
            'current_pos': None,
            'path': [],
            'budget': budget,
            'strategy': strategy,
            'agent_type': AgentType.REGULAR
        }
        
        self.layer_assignments[agent_id] = assigned_layer
        self.agent_states[agent_id] = AgentState.GROUND
        
        return True
    
    def get_emergency_agents_for_processing(self, layer: LayerType) -> List[Dict]:
        """
        Get emergency agents ready for direct path assignment.
        
        Args:
            layer: Target layer
            
        Returns:
            List[Dict]: Emergency agent information for pathfinding
        """
        emergency_agents = []
        
        for agent_id, agent_data in self.emergency_agents.items():
            if self.layer_assignments[agent_id] == layer:
                agent_info = {
                    'id': agent_id,
                    'start': agent_data['start'],
                    'goal': agent_data['goal'],
                    'current_pos': agent_data['current_pos'],
                    'is_emergency': True,
                    'agent_type': AgentType.EMERGENCY
                }
                emergency_agents.append(agent_info)
        
        return emergency_agents
    
    def get_regular_agents_for_processing(self, layer: LayerType) -> List[Dict]:
        """
        Get regular agents for CBS/auction processing (after emergency agent assignment).
        
        Args:
            layer: Target layer
            
        Returns:
            List[Dict]: Regular agent information for pathfinding/auction
        """
        regular_agents = []
        
        for agent_id, agent_data in self.regular_agents.items():
            if self.layer_assignments[agent_id] == layer:
                agent_info = {
                    'id': agent_id,
                    'start': agent_data['start'],
                    'goal': agent_data['goal'],
                    'current_pos': agent_data['current_pos'],
                    'budget': agent_data['budget'],
                    'strategy': agent_data['strategy'],
                    'is_emergency': False,
                    'agent_type': AgentType.REGULAR
                }
                regular_agents.append(agent_info)
        
        return regular_agents
    
    def assign_emergency_paths(self, layer: LayerType, emergency_paths: Dict[int, List[Tuple[int, int]]],
                              start_time: int) -> bool:
        """
        Assign paths to emergency agents and create reservations.
        
        Args:
            layer: Layer being processed
            emergency_paths: Dictionary mapping agent_id to path
            start_time: Starting time for the paths
            
        Returns:
            bool: True if all paths assigned successfully
        """
        for agent_id, path in emergency_paths.items():
            if agent_id not in self.emergency_agents:
                return False
            
            # Update agent path
            self.emergency_agents[agent_id]['path'] = path
            
            # Create time steps for the path
            time_steps = list(range(start_time, start_time + len(path)))
            
            # Add reservation (this creates dynamic obstacles for regular agents)
            success = self.emergency_reservations.add_reservation(
                agent_id, layer, path, time_steps
            )
            
            if not success:
                return False
        
        return True
    
    def is_position_available_for_regular_agent(self, layer: LayerType, position: Tuple[int, int], 
                                              time_step: int, agent_id: int) -> bool:
        """
        Check if position is available for regular agents (considering emergency reservations).
        
        Args:
            layer: Layer to check
            position: Position (x, y) to check
            time_step: Time step to check
            agent_id: ID of requesting regular agent
            
        Returns:
            bool: True if position is available for regular agents
        """
        x, y = position
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        if not (0 <= time_step < self.max_time):
            return False
        
        # Check static occupancy (infrastructure, obstacles)
        if self.grid_occupancy[layer][time_step, x, y]:
            return False
        
        # Check emergency agent reservations (dynamic obstacles)
        is_reserved, _ = self.emergency_reservations.is_position_reserved(
            layer, position, time_step
        )
        
        return not is_reserved
    
    def is_position_available_for_emergency_agent(self, layer: LayerType, position: Tuple[int, int], 
                                                 time_step: int, agent_id: int) -> bool:
        """
        Check if position is available for emergency agents (only static obstacles matter).
        
        Args:
            layer: Layer to check
            position: Position (x, y) to check
            time_step: Time step to check
            agent_id: ID of requesting emergency agent
            
        Returns:
            bool: True if position is available for emergency agents
        """
        x, y = position
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        if not (0 <= time_step < self.max_time):
            return False
        
        # Emergency agents only care about static obstacles
        return not self.grid_occupancy[layer][time_step, x, y]
    
    def update_regular_agent_paths(self, layer: LayerType, agent_paths: Dict[int, List[Tuple[int, int]]]) -> None:
        """
        Update regular agent paths after CBS/auction processing.
        
        Args:
            layer: Layer that was processed
            agent_paths: Dictionary mapping agent_id to path
        """
        for agent_id, path in agent_paths.items():
            if agent_id in self.regular_agents and self.layer_assignments[agent_id] == layer:
                self.regular_agents[agent_id]['path'] = path
    
    def execute_temporal_phase(self, time_step: int) -> Dict[str, List[int]]:
        """
        Execute temporal processing phase with emergency priority.
        
        Args:
            time_step: Current time step
            
        Returns:
            Dict with lists of agents in different phases
        """
        phase_agents = {
            'entering_lower_emergency': [],
            'entering_lower_regular': [],
            'climbing_emergency': [],
            'climbing_regular': [],
            'staying_lower_emergency': [],
            'staying_lower_regular': [],
            'independent_lower_emergency': [],
            'independent_lower_regular': [],
            'independent_upper_emergency': [],
            'independent_upper_regular': [],
            'descending_emergency': [],
            'descending_regular': []
        }
        
        if time_step == 1:
            # All agents enter lower layer (emergency first)
            for agent_id in self.emergency_agents:
                if self.agent_states[agent_id] == AgentState.GROUND:
                    self.agent_states[agent_id] = AgentState.LOWER_LAYER
                    start_pos = self.emergency_agents[agent_id]['start']
                    self.emergency_agents[agent_id]['current_pos'] = start_pos
                    phase_agents['entering_lower_emergency'].append(agent_id)
            
            for agent_id in self.regular_agents:
                if self.agent_states[agent_id] == AgentState.GROUND:
                    self.agent_states[agent_id] = AgentState.LOWER_LAYER
                    start_pos = self.regular_agents[agent_id]['start']
                    self.regular_agents[agent_id]['current_pos'] = start_pos
                    phase_agents['entering_lower_regular'].append(agent_id)
                    
        elif time_step == 2:
            # Layer transitions (emergency agents processed first)
            for agent_id in self.emergency_agents:
                if self.agent_states[agent_id] == AgentState.LOWER_LAYER:
                    assigned_layer = self.layer_assignments[agent_id]
                    
                    if assigned_layer == LayerType.UPPER:
                        self.agent_states[agent_id] = AgentState.CLIMBING
                        phase_agents['climbing_emergency'].append(agent_id)
                    else:
                        phase_agents['staying_lower_emergency'].append(agent_id)
            
            for agent_id in self.regular_agents:
                if self.agent_states[agent_id] == AgentState.LOWER_LAYER:
                    assigned_layer = self.layer_assignments[agent_id]
                    
                    if assigned_layer == LayerType.UPPER:
                        self.agent_states[agent_id] = AgentState.CLIMBING
                        phase_agents['climbing_regular'].append(agent_id)
                    else:
                        phase_agents['staying_lower_regular'].append(agent_id)
                        
        else:  # time_step >= 3
            # Independent layer processing (emergency agents processed first)
            for agent_id in self.emergency_agents:
                agent_state = self.agent_states[agent_id]
                
                if agent_state == AgentState.CLIMBING:
                    self.agent_states[agent_id] = AgentState.UPPER_LAYER
                    phase_agents['independent_upper_emergency'].append(agent_id)
                elif agent_state == AgentState.LOWER_LAYER:
                    phase_agents['independent_lower_emergency'].append(agent_id)
                elif agent_state == AgentState.UPPER_LAYER:
                    if self.check_descent_condition(agent_id, time_step):
                        self.agent_states[agent_id] = AgentState.DESCENDING
                        phase_agents['descending_emergency'].append(agent_id)
                    else:
                        phase_agents['independent_upper_emergency'].append(agent_id)
            
            for agent_id in self.regular_agents:
                agent_state = self.agent_states[agent_id]
                
                if agent_state == AgentState.CLIMBING:
                    self.agent_states[agent_id] = AgentState.UPPER_LAYER
                    phase_agents['independent_upper_regular'].append(agent_id)
                elif agent_state == AgentState.LOWER_LAYER:
                    phase_agents['independent_lower_regular'].append(agent_id)
                elif agent_state == AgentState.UPPER_LAYER:
                    if self.check_descent_condition(agent_id, time_step):
                        self.agent_states[agent_id] = AgentState.DESCENDING
                        phase_agents['descending_regular'].append(agent_id)
                    else:
                        phase_agents['independent_upper_regular'].append(agent_id)
        
        self.current_time = time_step
        return phase_agents
    
    def check_descent_condition(self, agent_id: int, current_time: int) -> bool:
        """
        Check if an upper layer agent can descend (3-step lookahead).
        Enhanced to consider emergency agent reservations.
        
        Args:
            agent_id: Agent identifier
            current_time: Current simulation time
            
        Returns:
            bool: True if descent is possible
        """
        # Get agent information from appropriate storage
        agent = None
        if agent_id in self.emergency_agents:
            agent = self.emergency_agents[agent_id]
        elif agent_id in self.regular_agents:
            agent = self.regular_agents[agent_id]
        else:
            return False
        
        current_pos = agent['current_pos']
        if current_pos is None or self.agent_states[agent_id] != AgentState.UPPER_LAYER:
            return False
        
        # Check if we have enough time steps remaining
        if current_time + 2 >= self.max_time:
            return False
        
        x, y = current_pos
        
        # Emergency agents have priority - only check static obstacles
        if agent_id in self.emergency_agents:
            # Check t+1: Upper layer position must be free of static obstacles
            if self.grid_occupancy[LayerType.UPPER][current_time + 1, x, y]:
                return False
            
            # Check t+2: Lower layer position must be free of static obstacles
            if self.grid_occupancy[LayerType.LOWER][current_time + 2, x, y]:
                return False
        else:
            # Regular agents must consider both static obstacles and emergency reservations
            if not self.is_position_available_for_regular_agent(
                LayerType.UPPER, (x, y), current_time + 1, agent_id
            ):
                return False
            
            if not self.is_position_available_for_regular_agent(
                LayerType.LOWER, (x, y), current_time + 2, agent_id
            ):
                return False
        
        return True
    
    def set_processing_phase(self, phase: ProcessingPhase) -> None:
        """Set the current processing phase."""
        self.processing_phase = phase
    
    def get_processing_phase(self) -> ProcessingPhase:
        """Get the current processing phase."""
        return self.processing_phase
    
    def get_system_state(self) -> Dict:
        """
        Get comprehensive system state for monitoring and analysis.
        
        Returns:
            Dict: Complete system state information
        """
        emergency_by_layer = {
            'lower': len([aid for aid, layer in self.layer_assignments.items() 
                         if layer == LayerType.LOWER and aid in self.emergency_agents]),
            'upper': len([aid for aid, layer in self.layer_assignments.items() 
                         if layer == LayerType.UPPER and aid in self.emergency_agents])
        }
        
        regular_by_layer = {
            'lower': len([aid for aid, layer in self.layer_assignments.items() 
                         if layer == LayerType.LOWER and aid in self.regular_agents]),
            'upper': len([aid for aid, layer in self.layer_assignments.items() 
                         if layer == LayerType.UPPER and aid in self.regular_agents])
        }
        
        return {
            'current_time': self.current_time,
            'processing_phase': self.processing_phase.name,
            'emergency_agents': len(self.emergency_agents),
            'regular_agents': len(self.regular_agents),
            'emergency_by_layer': emergency_by_layer,
            'regular_by_layer': regular_by_layer,
            'emergency_reservations': len(self.emergency_reservations.reservations),
            'grid_dimensions': (self.width, self.height),
            'max_time': self.max_time
        }
    
    def get_emergency_dynamic_obstacles(self, layer: LayerType) -> Dict[Tuple[int, int, int], int]:
        """
        Get emergency agent positions as dynamic obstacles for regular agent processing.
        
        Args:
            layer: Layer to get obstacles for
            
        Returns:
            Dict: Mapping (x, y, time) -> emergency_agent_id
        """
        obstacles = {}
        
        for (res_layer, x, y, time), agent_id in self.emergency_reservations.occupancy_map.items():
            if res_layer == layer:
                obstacles[(x, y, time)] = agent_id
        
        return obstacles
    
    def export_scenario_data(self) -> Dict:
        """
        Export scenario data including emergency priorities.
        
        Returns:
            Dict: Complete scenario data in JSON-serializable format
        """
        return {
            'grid_dimensions': {'width': self.width, 'height': self.height},
            'max_time': self.max_time,
            'current_time': self.current_time,
            'processing_phase': self.processing_phase.name,
            'emergency_agents': {
                str(agent_id): {
                    'start': agent_data['start'],
                    'goal': agent_data['goal'],
                    'layer': agent_data['layer'].name,
                    'agent_type': 'emergency'
                }
                for agent_id, agent_data in self.emergency_agents.items()
            },
            'regular_agents': {
                str(agent_id): {
                    'start': agent_data['start'],
                    'goal': agent_data['goal'],
                    'layer': agent_data['layer'].name,
                    'budget': agent_data['budget'],
                    'strategy': agent_data['strategy'],
                    'agent_type': 'regular'
                }
                for agent_id, agent_data in self.regular_agents.items()
            },
            'emergency_reservations': {
                str(agent_id): {
                    'layer': res_data['layer'].name,
                    'path': res_data['path'],
                    'time_steps': res_data['time_steps']
                }
                for agent_id, res_data in self.emergency_reservations.reservations.items()
            }
        }
    
    def reset_system(self) -> None:
        """Reset the system to initial state."""
        self.current_time = 0
        self.emergency_agents.clear()
        self.regular_agents.clear()
        self.layer_assignments.clear()
        self.agent_states.clear()
        self.emergency_reservations.clear_reservations()
        
        # Reset occupancy grids
        self.grid_occupancy[LayerType.LOWER].fill(False)
        self.grid_occupancy[LayerType.UPPER].fill(False)
        
        self.processing_phase = ProcessingPhase.EMERGENCY_ASSIGNMENT