"""
Main Orchestration Module for Multilayer Urban Airspace Simulation
Integrates all components: enhanced grid system, pathfinding, CBS, and auction
for comprehensive two-layer airspace management with emergency priority.
"""

import random
import time
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import all enhanced modules
from enhanced_multilayer_grid_system import (
    EnhancedMultilayerGridSystem, LayerType, AgentType, AgentState, ProcessingPhase
)
from enhanced_astar_pathfinding import (
    EnhancedAStarPathfinder, PathfindingResult
)
from enhanced_cbs import (
    EnhancedCBS, CBSResult
)
from enhanced_multilayer_auction import (
    EnhancedAuctionSystem, BiddingStrategy, AuctionResult,
    create_enhanced_auction_system, prepare_regular_agents_for_auction
)

class ProcessingMode(Enum):
    """Different processing modes for system evaluation"""
    HYBRID = "hybrid"                    # Emergency->CBS->Auction (recommended)
    EMERGENCY_CBS_ONLY = "emergency_cbs_only"    # Emergency->CBS, no auction
    EMERGENCY_AUCTION_ONLY = "emergency_auction_only"  # Emergency->Auction, no CBS
    CBS_ONLY = "cbs_only"               # CBS only (no emergency priority)
    AUCTION_ONLY = "auction_only"       # Auction only (no CBS)

@dataclass
class SimulationMetrics:
    """Comprehensive metrics for simulation evaluation"""
    # Overall performance
    success_rate: float
    total_computation_time: float
    
    # Agent breakdown
    total_agents: int
    emergency_agents: int
    regular_agents: int
    
    # Layer distribution
    lower_layer_agents: int
    upper_layer_agents: int
    
    # Success by type
    emergency_success_count: int
    regular_success_count: int
    emergency_success_rate: float
    regular_success_rate: float
    
    # Processing results
    cbs_successful: bool
    cbs_computation_time: float
    auction_successful: bool
    auction_computation_time: float
    auction_total_revenue: float
    
    # Conflict analysis
    emergency_conflicts_detected: int
    regular_conflicts_resolved: int
    
    # Descent analysis
    successful_descents: int
    blocked_descents: int
    
    # Fields with default values must come last
    # Strategy performance (for regular agents)
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Layer-specific metrics
    lower_layer_success_rate: float = 0.0
    upper_layer_success_rate: float = 0.0

@dataclass
class SimulationResult:
    """Complete result of a simulation run"""
    success: bool
    processing_mode: ProcessingMode
    metrics: SimulationMetrics
    agent_paths: Dict[int, List[Tuple[int, int]]]  # agent_id -> path
    failure_reasons: List[str] = field(default_factory=list)
    message: str = ""

class MultilayerAirspaceSimulator:
    """
    Main simulator class integrating all multilayer airspace components.
    
    Handles complete simulation workflow:
    1. Emergency agent processing (direct CBS assignment)
    2. Regular agent processing (CBS and/or auction)
    3. Temporal phase management
    4. Descent condition handling
    5. Comprehensive evaluation and metrics
    """
    
    def __init__(self, width: int = 20, height: int = 20, max_time: int = 100):
        """
        Initialize the multilayer airspace simulator.
        
        Args:
            width: Grid width
            height: Grid height
            max_time: Maximum simulation time steps
        """
        self.width = width
        self.height = height
        self.max_time = max_time
        
        # Initialize core components
        self.grid_system = EnhancedMultilayerGridSystem(width, height, max_time)
        self.pathfinder = EnhancedAStarPathfinder(self.grid_system)
        self.cbs_solver = EnhancedCBS(self.grid_system, self.pathfinder)
        self.auction_system = create_enhanced_auction_system(
            self.grid_system, self.pathfinder, self.cbs_solver
        )
        
        # Default scenario parameters
        self.default_emergency_agents = 4
        self.default_regular_agents = 24
        self.default_budget_range = (10.0, 100.0)
        self.default_strategy_distribution = {
            'conservative': 0.3,
            'aggressive': 0.3,
            'balanced': 0.4
        }
        
        # Performance tracking
        self.simulation_count = 0
        self.debug_mode = False
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Enable debug mode for all components."""
        self.debug_mode = enabled
        self.auction_system.set_debug_mode(enabled)
    
    def generate_unique_positions(self, count: int) -> List[Tuple[int, int]]:
        """Generate unique random positions within grid bounds."""
        positions = set()
        attempts = 0
        max_attempts = count * 10
        
        while len(positions) < count and attempts < max_attempts:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            positions.add((x, y))
            attempts += 1
        
        if len(positions) < count:
            raise ValueError(f"Could not generate {count} unique positions in {self.width}x{self.height} grid")
        
        return list(positions)
    
    def create_scenario(self, emergency_agents: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                       regular_agents: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                       regular_budgets: Optional[Dict[int, float]] = None,
                       regular_strategies: Optional[Dict[int, str]] = None) -> bool:
        """
        Create a simulation scenario with specified agents.
        
        Args:
            emergency_agents: List of (start, goal) for emergency agents
            regular_agents: List of (start, goal) for regular agents  
            regular_budgets: Optional budget assignment for regular agents
            regular_strategies: Optional strategy assignment for regular agents
            
        Returns:
            bool: True if scenario created successfully
        """
        # Reset system
        self.grid_system.reset_system()
        
        # Add emergency agents
        emergency_agent_id = 1
        for start, goal in emergency_agents:
            success = self.grid_system.add_emergency_agent(emergency_agent_id, start, goal)
            if not success:
                if self.debug_mode:
                    print(f"Failed to add emergency agent {emergency_agent_id}")
                return False
            emergency_agent_id += 1
        
        # Add regular agents
        regular_agent_id = 1000  # Start regular agents at 1000 to avoid conflicts
        for start, goal in regular_agents:
            # Determine budget
            budget = 50.0  # default
            if regular_budgets and regular_agent_id in regular_budgets:
                budget = regular_budgets[regular_agent_id]
            else:
                budget = random.uniform(self.default_budget_range[0], self.default_budget_range[1])
            
            # Determine strategy
            strategy = "aggressive"  # default
            if regular_strategies and regular_agent_id in regular_strategies:
                strategy = regular_strategies[regular_agent_id]
            else:
                strategy = random.choices(
                    list(self.default_strategy_distribution.keys()),
                    weights=list(self.default_strategy_distribution.values())
                )[0]
            
            success = self.grid_system.add_regular_agent(regular_agent_id, start, goal, budget, strategy)
            if not success:
                if self.debug_mode:
                    print(f"Failed to add regular agent {regular_agent_id}")
                return False
            regular_agent_id += 1
        
        return True
    
    def create_default_scenario(self) -> bool:
        """Create a default test scenario."""
        # Generate unique positions for all agents
        total_agents = self.default_emergency_agents + self.default_regular_agents
        positions = self.generate_unique_positions(total_agents * 2)  # *2 for start and goal
        
        # Create emergency agents
        emergency_agents = []
        for i in range(self.default_emergency_agents):
            start = positions[i * 2]
            goal = positions[i * 2 + 1]
            emergency_agents.append((start, goal))
        
        # Create regular agents
        regular_agents = []
        start_idx = self.default_emergency_agents * 2
        for i in range(self.default_regular_agents):
            start = positions[start_idx + i * 2]
            goal = positions[start_idx + i * 2 + 1]
            regular_agents.append((start, goal))
        
        return self.create_scenario(emergency_agents, regular_agents)
    
    def process_emergency_agents(self) -> Tuple[bool, Dict[str, float]]:
        """
        Process emergency agents with direct CBS assignment.
        
        Returns:
            Tuple: (success, metrics)
        """
        start_time = time.time()
        metrics = {'computation_time': 0.0, 'success_count': 0, 'total_count': 0}
        
        # Process each layer separately
        for layer in [LayerType.LOWER, LayerType.UPPER]:
            emergency_agents = self.grid_system.get_emergency_agents_for_processing(layer)
            metrics['total_count'] += len(emergency_agents)
            
            if not emergency_agents:
                continue
            
            if self.debug_mode:
                print(f"Processing {len(emergency_agents)} emergency agents in {layer.name} layer")
            
            # Use CBS to assign paths to emergency agents
            cbs_response = self.cbs_solver.solve_emergency_agents(layer, emergency_agents, 1)
            
            if cbs_response.result == CBSResult.SUCCESS:
                # Assign emergency paths to grid system (creates reservations)
                success = self.grid_system.assign_emergency_paths(
                    layer, cbs_response.solution, 1
                )
                if success:
                    metrics['success_count'] += len(cbs_response.solution)
                    if self.debug_mode:
                        print(f"Successfully assigned {len(cbs_response.solution)} emergency paths in {layer.name}")
                else:
                    if self.debug_mode:
                        print(f"Failed to assign emergency paths in {layer.name}")
            else:
                if self.debug_mode:
                    print(f"Emergency CBS failed in {layer.name}: {cbs_response.message}")
        
        metrics['computation_time'] = time.time() - start_time
        success = metrics['success_count'] == metrics['total_count']
        
        return success, metrics
    
    def process_regular_agents_with_cbs(self) -> Tuple[bool, Dict[str, Union[float, int, bool]]]:
        """
        Process regular agents using CBS (after emergency assignment).
        
        Returns:
            Tuple: (success, metrics)
        """
        start_time = time.time()
        metrics = {
            'computation_time': 0.0,
            'success_count': 0,
            'total_count': 0,
            'lower_success': 0,
            'upper_success': 0,
            'successful': False
        }
        
        # Process each layer
        total_success = True
        for layer in [LayerType.LOWER, LayerType.UPPER]:
            regular_agents = self.grid_system.get_regular_agents_for_processing(layer)
            layer_total = len(regular_agents)
            metrics['total_count'] += layer_total
            
            if not regular_agents:
                continue
            
            if self.debug_mode:
                print(f"Processing {len(regular_agents)} regular agents with CBS in {layer.name} layer")
            
            # Run CBS for regular agents (considers emergency reservations)
            cbs_response = self.cbs_solver.solve_regular_agents(layer, regular_agents, 3)
            
            layer_success_count = len(cbs_response.successful_agents)
            metrics['success_count'] += layer_success_count
            
            if layer == LayerType.LOWER:
                metrics['lower_success'] = layer_success_count
            else:
                metrics['upper_success'] = layer_success_count
            
            if cbs_response.result in [CBSResult.SUCCESS, CBSResult.PARTIAL_SUCCESS]:
                # Update grid with successful paths
                if cbs_response.solution:
                    self.grid_system.update_regular_agent_paths(layer, cbs_response.solution)
                    if self.debug_mode:
                        print(f"CBS assigned {len(cbs_response.solution)} paths in {layer.name}")
            else:
                total_success = False
                if self.debug_mode:
                    print(f"CBS failed in {layer.name}: {cbs_response.message}")
        
        metrics['computation_time'] = time.time() - start_time
        metrics['successful'] = total_success and (metrics['success_count'] > 0)
        
        return total_success, metrics
    
    def process_regular_agents_with_auction(self) -> Tuple[bool, Dict[str, Union[float, int, bool]]]:
        """
        Process regular agents using auction system (after emergency assignment).
        
        Returns:
            Tuple: (success, metrics)  
        """
        start_time = time.time()
        
        # Run multilayer auction
        auction_result = self.auction_system.run_multilayer_auction(start_time=3)
        
        # Extract metrics
        total_regular_agents = (len(auction_result.lower_layer_result.final_winners) + 
                               len(auction_result.lower_layer_result.unassigned_agents) +
                               len(auction_result.lower_layer_result.emergency_blocked_agents) +
                               len(auction_result.upper_layer_result.final_winners) +
                               len(auction_result.upper_layer_result.unassigned_agents) +
                               len(auction_result.upper_layer_result.emergency_blocked_agents))
        
        total_winners = (len(auction_result.lower_layer_result.final_winners) +
                        len(auction_result.upper_layer_result.final_winners))
        
        metrics = {
            'computation_time': auction_result.total_computation_time,
            'success_count': total_winners,
            'total_count': total_regular_agents,
            'lower_success': len(auction_result.lower_layer_result.final_winners),
            'upper_success': len(auction_result.upper_layer_result.final_winners),
            'successful': auction_result.overall_success_rate > 0,
            'total_revenue': auction_result.total_revenue,
            'lower_revenue': auction_result.lower_layer_result.total_revenue,
            'upper_revenue': auction_result.upper_layer_result.total_revenue
        }
        
        success = auction_result.overall_success_rate > 50.0  # Consider >50% success as overall success
        
        if self.debug_mode:
            print(f"Auction completed: {total_winners}/{total_regular_agents} agents successful")
            print(f"Revenue: ${auction_result.total_revenue:.2f}")
        
        return success, metrics
    
    def simulate_descent_operations(self) -> Dict[str, int]:
        """
        Simulate upper layer agents attempting descent.
        
        Returns:
            Dict: Descent operation metrics
        """
        descent_metrics = {
            'attempted_descents': 0,
            'successful_descents': 0,
            'blocked_descents': 0
        }
        
        # Check agents in upper layer
        for agent_id in self.grid_system.emergency_agents:
            if (self.grid_system.layer_assignments.get(agent_id) == LayerType.UPPER and
                self.grid_system.agent_states.get(agent_id) == AgentState.UPPER_LAYER):
                
                descent_metrics['attempted_descents'] += 1
                
                # Check descent condition at various time steps
                for t in range(5, self.max_time - 3):
                    if self.grid_system.check_descent_condition(agent_id, t):
                        descent_metrics['successful_descents'] += 1
                        # Simulate descent
                        self.grid_system.agent_states[agent_id] = AgentState.DESCENDING
                        break
                else:
                    descent_metrics['blocked_descents'] += 1
        
        # Repeat for regular agents
        for agent_id in self.grid_system.regular_agents:
            if (self.grid_system.layer_assignments.get(agent_id) == LayerType.UPPER and
                self.grid_system.agent_states.get(agent_id) == AgentState.UPPER_LAYER):
                
                descent_metrics['attempted_descents'] += 1
                
                for t in range(5, self.max_time - 3):
                    if self.grid_system.check_descent_condition(agent_id, t):
                        descent_metrics['successful_descents'] += 1
                        self.grid_system.agent_states[agent_id] = AgentState.DESCENDING
                        break
                else:
                    descent_metrics['blocked_descents'] += 1
        
        return descent_metrics
    
    def calculate_simulation_metrics(self, processing_mode: ProcessingMode, 
                                   emergency_metrics: Dict, regular_metrics: Dict,
                                   descent_metrics: Dict, total_time: float) -> SimulationMetrics:
        """Calculate comprehensive simulation metrics."""
        
        # Count agents
        total_emergency = len(self.grid_system.emergency_agents)
        total_regular = len(self.grid_system.regular_agents)
        total_agents = total_emergency + total_regular
        
        # Layer distribution
        lower_emergency = len([aid for aid in self.grid_system.emergency_agents 
                              if self.grid_system.layer_assignments.get(aid) == LayerType.LOWER])
        lower_regular = len([aid for aid in self.grid_system.regular_agents 
                            if self.grid_system.layer_assignments.get(aid) == LayerType.LOWER])
        
        # Success rates
        emergency_success_count = emergency_metrics.get('success_count', 0)
        regular_success_count = regular_metrics.get('success_count', 0)
        
        emergency_success_rate = (emergency_success_count / total_emergency * 100) if total_emergency > 0 else 100.0
        regular_success_rate = (regular_success_count / total_regular * 100) if total_regular > 0 else 0.0
        overall_success_rate = ((emergency_success_count + regular_success_count) / total_agents * 100) if total_agents > 0 else 0.0
        
        # Layer success rates
        lower_total = lower_emergency + lower_regular
        upper_total = (total_emergency - lower_emergency) + (total_regular - lower_regular)
        
        lower_success = emergency_success_count  # All emergency agents should succeed
        if 'lower_success' in regular_metrics:
            lower_success += regular_metrics['lower_success']
        
        upper_success = 0
        if 'upper_success' in regular_metrics:
            upper_success = regular_metrics['upper_success']
        
        lower_success_rate = (lower_success / lower_total * 100) if lower_total > 0 else 0.0
        upper_success_rate = (upper_success / upper_total * 100) if upper_total > 0 else 0.0
        
        # Strategy performance (for regular agents)
        strategy_stats = self.auction_system.get_performance_stats().get('strategy_performance', {})
        strategy_success_rates = {k: v.get('success_rate_percent', 0.0) for k, v in strategy_stats.items()}
        
        return SimulationMetrics(
            success_rate=overall_success_rate,
            total_computation_time=total_time,
            total_agents=total_agents,
            emergency_agents=total_emergency,
            regular_agents=total_regular,
            lower_layer_agents=lower_total,
            upper_layer_agents=upper_total,
            emergency_success_count=emergency_success_count,
            regular_success_count=regular_success_count,
            emergency_success_rate=emergency_success_rate,
            regular_success_rate=regular_success_rate,
            cbs_successful=regular_metrics.get('successful', False),
            cbs_computation_time=regular_metrics.get('computation_time', 0.0),
            auction_successful=regular_metrics.get('successful', False),
            auction_computation_time=regular_metrics.get('computation_time', 0.0),
            auction_total_revenue=regular_metrics.get('total_revenue', 0.0),
            strategy_success_rates=strategy_success_rates,
            emergency_conflicts_detected=0,  # Would need to track from processing
            regular_conflicts_resolved=0,    # Would need to track from processing
            successful_descents=descent_metrics.get('successful_descents', 0),
            blocked_descents=descent_metrics.get('blocked_descents', 0),
            lower_layer_success_rate=lower_success_rate,
            upper_layer_success_rate=upper_success_rate
        )
    
    def run_simulation(self, processing_mode: ProcessingMode = ProcessingMode.HYBRID) -> SimulationResult:
        """
        Run complete multilayer airspace simulation.
        
        Args:
            processing_mode: How to process regular agents
            
        Returns:
            SimulationResult: Complete simulation result
        """
        simulation_start_time = time.time()
        self.simulation_count += 1
        
        if self.debug_mode:
            print(f"\n{'='*50}")
            print(f"Starting Simulation #{self.simulation_count}")
            print(f"Processing Mode: {processing_mode.value}")
            print(f"{'='*50}")
        
        failure_reasons = []
        
        # Phase 1: Process emergency agents (always first)
        emergency_success, emergency_metrics = self.process_emergency_agents()
        if not emergency_success:
            failure_reasons.append("Emergency agent processing failed")
        
        # Phase 2: Process regular agents based on mode
        regular_success = True
        regular_metrics = {'computation_time': 0.0, 'success_count': 0, 'total_count': 0, 'successful': False}
        
        if processing_mode == ProcessingMode.HYBRID:
            # Try CBS first, fall back to auction if needed
            cbs_success, cbs_metrics = self.process_regular_agents_with_cbs()
            if cbs_success:
                regular_metrics = cbs_metrics
            else:
                if self.debug_mode:
                    print("CBS failed, falling back to auction")
                auction_success, auction_metrics = self.process_regular_agents_with_auction()
                regular_metrics = auction_metrics
                regular_success = auction_success
        
        elif processing_mode == ProcessingMode.EMERGENCY_CBS_ONLY:
            cbs_success, regular_metrics = self.process_regular_agents_with_cbs()
            regular_success = cbs_success
        
        elif processing_mode == ProcessingMode.EMERGENCY_AUCTION_ONLY:
            auction_success, regular_metrics = self.process_regular_agents_with_auction()
            regular_success = auction_success
        
        elif processing_mode == ProcessingMode.CBS_ONLY:
            # Skip emergency processing, just CBS
            cbs_success, regular_metrics = self.process_regular_agents_with_cbs()
            regular_success = cbs_success
        
        elif processing_mode == ProcessingMode.AUCTION_ONLY:
            # Skip emergency processing, just auction
            auction_success, regular_metrics = self.process_regular_agents_with_auction()
            regular_success = auction_success
        
        if not regular_success:
            failure_reasons.append("Regular agent processing failed")
        
        # Phase 3: Simulate descent operations
        descent_metrics = self.simulate_descent_operations()
        
        # Calculate comprehensive metrics
        total_time = time.time() - simulation_start_time
        metrics = self.calculate_simulation_metrics(
            processing_mode, emergency_metrics, regular_metrics, descent_metrics, total_time
        )
        
        # Extract agent paths
        agent_paths = {}
        for agent_id, agent_data in self.grid_system.emergency_agents.items():
            if agent_data['path']:
                agent_paths[agent_id] = agent_data['path']
        
        for agent_id, agent_data in self.grid_system.regular_agents.items():
            if agent_data['path']:
                agent_paths[agent_id] = agent_data['path']
        
        # Determine overall success
        overall_success = (emergency_success and regular_success and 
                          metrics.success_rate >= 80.0)  # 80% threshold for success
        
        message = f"Simulation completed: {metrics.success_rate:.1f}% success rate"
        if failure_reasons:
            message += f" (Issues: {', '.join(failure_reasons)})"
        
        if self.debug_mode:
            print(f"\nSimulation Results:")
            print(f"Overall Success Rate: {metrics.success_rate:.1f}%")
            print(f"Emergency Agents: {metrics.emergency_success_count}/{metrics.emergency_agents}")
            print(f"Regular Agents: {metrics.regular_success_count}/{metrics.regular_agents}")
            print(f"Total Computation Time: {metrics.total_computation_time:.3f}s")
            print(f"Successful Descents: {metrics.successful_descents}")
        
        return SimulationResult(
            success=overall_success,
            processing_mode=processing_mode,
            metrics=metrics,
            agent_paths=agent_paths,
            failure_reasons=failure_reasons,
            message=message
        )
    
    def run_batch_simulations(self, num_simulations: int, 
                            processing_mode: ProcessingMode = ProcessingMode.HYBRID) -> List[SimulationResult]:
        """
        Run multiple simulations for statistical analysis.
        
        Args:
            num_simulations: Number of simulations to run
            processing_mode: Processing mode to use
            
        Returns:
            List of simulation results
        """
        results = []
        
        if self.debug_mode:
            print(f"\nRunning {num_simulations} batch simulations with {processing_mode.value}")
        
        for i in range(num_simulations):
            # Create new random scenario for each simulation
            self.create_default_scenario()
            
            # Run simulation
            result = self.run_simulation(processing_mode)
            results.append(result)
            
            if self.debug_mode and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_simulations} simulations")
        
        return results
    
    def export_results_to_json(self, results: List[SimulationResult], filename: str) -> bool:
        """Export simulation results to JSON file."""
        try:
            export_data = {
                'simulation_parameters': {
                    'grid_width': self.width,
                    'grid_height': self.height,
                    'max_time': self.max_time,
                    'default_emergency_agents': self.default_emergency_agents,
                    'default_regular_agents': self.default_regular_agents
                },
                'results': []
            }
            
            for result in results:
                result_data = {
                    'success': result.success,
                    'processing_mode': result.processing_mode.value,
                    'message': result.message,
                    'failure_reasons': result.failure_reasons,
                    'metrics': {
                        'success_rate': result.metrics.success_rate,
                        'total_computation_time': result.metrics.total_computation_time,
                        'total_agents': result.metrics.total_agents,
                        'emergency_agents': result.metrics.emergency_agents,
                        'regular_agents': result.metrics.regular_agents,
                        'emergency_success_rate': result.metrics.emergency_success_rate,
                        'regular_success_rate': result.metrics.regular_success_rate,
                        'auction_total_revenue': result.metrics.auction_total_revenue,
                        'successful_descents': result.metrics.successful_descents,
                        'blocked_descents': result.metrics.blocked_descents,
                        'strategy_success_rates': result.metrics.strategy_success_rates
                    }
                }
                export_data['results'].append(result_data)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"Failed to export results: {e}")
            return False

def create_simulation_system(width: int = 20, height: int = 20, max_time: int = 100) -> MultilayerAirspaceSimulator:
    """Create a new simulation system with default parameters."""
    return MultilayerAirspaceSimulator(width, height, max_time)

def run_evaluation_study(num_simulations: int = 50) -> Dict[ProcessingMode, Dict[str, float]]:
    """
    Run comprehensive evaluation study comparing all processing modes.
    
    Args:
        num_simulations: Number of simulations per mode
        
    Returns:
        Dict mapping ProcessingMode to performance statistics
    """
    simulator = create_simulation_system()
    simulator.set_debug_mode(False)  # Disable debug for batch runs
    
    results = {}
    
    for mode in [ProcessingMode.HYBRID, ProcessingMode.EMERGENCY_CBS_ONLY, 
                ProcessingMode.EMERGENCY_AUCTION_ONLY]:
        
        print(f"Evaluating {mode.value}...")
        mode_results = simulator.run_batch_simulations(num_simulations, mode)
        
        # Calculate statistics
        success_rates = [r.metrics.success_rate for r in mode_results]
        computation_times = [r.metrics.total_computation_time for r in mode_results]
        revenues = [r.metrics.auction_total_revenue for r in mode_results]
        
        results[mode] = {
            'success_rate': sum(success_rates) / len(success_rates),
            'avg_computation_time': sum(computation_times) / len(computation_times),
            'avg_revenue': sum(revenues) / len(revenues),
            'successful_simulations': sum(1 for r in mode_results if r.success),
            'total_simulations': len(mode_results)
        }
        
        print(f"{mode.value}: {results[mode]['success_rate']:.1f}% avg success rate")
    
    return results

def analyze_scalability(max_agents: int = 50, step_size: int = 5) -> Dict[int, Dict[str, float]]:
    """
    Analyze system scalability with increasing agent counts.
    
    Args:
        max_agents: Maximum number of total agents to test
        step_size: Step size for agent count increases
        
    Returns:
        Dict mapping agent count to performance metrics
    """
    simulator = create_simulation_system(width=25, height=25, max_time=150)
    simulator.set_debug_mode(False)
    
    results = {}
    
    for total_agents in range(step_size, max_agents + 1, step_size):
        emergency_count = max(1, total_agents // 8)  # ~12.5% emergency agents
        regular_count = total_agents - emergency_count
        
        print(f"Testing scalability: {total_agents} agents ({emergency_count} emergency, {regular_count} regular)")
        
        # Create scenario with specified agent count
        try:
            positions = simulator.generate_unique_positions(total_agents * 2)
            emergency_agents = [(positions[i*2], positions[i*2+1]) for i in range(emergency_count)]
            regular_agents = [(positions[emergency_count*2 + i*2], positions[emergency_count*2 + i*2+1]) 
                             for i in range(regular_count)]
            
            simulator.create_scenario(emergency_agents, regular_agents)
            
            # Run simulation
            result = simulator.run_simulation(ProcessingMode.HYBRID)
            
            results[total_agents] = {
                'success_rate': result.metrics.success_rate,
                'computation_time': result.metrics.total_computation_time,
                'emergency_success_rate': result.metrics.emergency_success_rate,
                'regular_success_rate': result.metrics.regular_success_rate,
                'successful_descents': result.metrics.successful_descents
            }
            
        except ValueError as e:
            print(f"Skipping {total_agents} agents: {e}")
            continue
    
    return results

def demonstrate_system() -> None:
    """Demonstrate the complete multilayer airspace system."""
    print("="*60)
    print("MULTILAYER URBAN AIRSPACE SIMULATION DEMONSTRATION")
    print("="*60)
    
    # Create simulation system
    simulator = create_simulation_system()
    simulator.set_debug_mode(True)
    
    print("\n1. Creating Default Scenario...")
    success = simulator.create_default_scenario()
    if not success:
        print("Failed to create scenario")
        return
    
    print(f"Created scenario with {simulator.default_emergency_agents} emergency and {simulator.default_regular_agents} regular agents")
    
    # Show system state
    system_state = simulator.grid_system.get_system_state()
    print(f"System State: {system_state}")
    
    print("\n2. Running Hybrid Simulation...")
    result = simulator.run_simulation(ProcessingMode.HYBRID)
    
    print("\n3. Results Summary:")
    print(f"Overall Success: {result.success}")
    print(f"Success Rate: {result.metrics.success_rate:.1f}%")
    print(f"Emergency Success: {result.metrics.emergency_success_count}/{result.metrics.emergency_agents}")
    print(f"Regular Success: {result.metrics.regular_success_count}/{result.metrics.regular_agents}")
    print(f"Total Time: {result.metrics.total_computation_time:.3f}s")
    print(f"Revenue Generated: ${result.metrics.auction_total_revenue:.2f}")
    print(f"Successful Descents: {result.metrics.successful_descents}")
    
    print("\n4. Layer Distribution:")
    print(f"Lower Layer: {result.metrics.lower_layer_agents} agents ({result.metrics.lower_layer_success_rate:.1f}% success)")
    print(f"Upper Layer: {result.metrics.upper_layer_agents} agents ({result.metrics.upper_layer_success_rate:.1f}% success)")
    
    print("\n5. Strategy Performance (Regular Agents):")
    for strategy, success_rate in result.metrics.strategy_success_rates.items():
        print(f"{strategy.capitalize()}: {success_rate:.1f}% success rate")
    
    print("\n6. Running Comparative Analysis...")
    comparison_results = []
    
    for mode in [ProcessingMode.HYBRID, ProcessingMode.EMERGENCY_CBS_ONLY, ProcessingMode.EMERGENCY_AUCTION_ONLY]:
        print(f"Testing {mode.value}...")
        simulator.create_default_scenario()  # Reset scenario
        mode_result = simulator.run_simulation(mode)
        comparison_results.append((mode, mode_result))
    
    print("\n7. Comparative Results:")
    print(f"{'Mode':<25} {'Success Rate':<15} {'Time (s)':<10} {'Revenue':<10}")
    print("-" * 65)
    
    for mode, result in comparison_results:
        print(f"{mode.value:<25} {result.metrics.success_rate:<14.1f}% {result.metrics.total_computation_time:<9.3f} ${result.metrics.auction_total_revenue:<9.2f}")
    
    print("\n8. System Performance Statistics:")
    auction_stats = simulator.auction_system.get_performance_stats()
    cbs_stats = simulator.cbs_solver.get_performance_stats()
    pathfinder_stats = simulator.pathfinder.get_performance_stats()
    
    print(f"Auction System: {auction_stats.get('total_auctions', 0)} auctions, {auction_stats.get('success_rate_percent', 0):.1f}% success")
    print(f"CBS System: {cbs_stats.get('total_requests', 0)} requests, {cbs_stats.get('success_rate_percent', 0):.1f}% success")
    print(f"Pathfinder: {pathfinder_stats.get('total_requests', 0)} requests, {pathfinder_stats.get('cache_hit_rate_percent', 0):.1f}% cache hits")
    
    print("\n9. Export Results...")
    simulator.export_results_to_json([result], "demonstration_results.json")
    print("Results exported to demonstration_results.json")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    """Main execution for testing and demonstration."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "demo":
            demonstrate_system()
        
        elif command == "evaluate":
            num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            print(f"Running evaluation study with {num_sims} simulations per mode...")
            results = run_evaluation_study(num_sims)
            
            print("\nEvaluation Results:")
            print(f"{'Mode':<25} {'Avg Success':<12} {'Avg Time':<12} {'Simulations':<12}")
            print("-" * 65)
            for mode, stats in results.items():
                print(f"{mode.value:<25} {stats['success_rate']:<11.1f}% {stats['avg_computation_time']:<11.3f}s {stats['successful_simulations']}/{stats['total_simulations']}")
        
        elif command == "scalability":
            max_agents = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            print(f"Running scalability analysis up to {max_agents} agents...")
            results = analyze_scalability(max_agents)
            
            print("\nScalability Results:")
            print(f"{'Agents':<8} {'Success Rate':<12} {'Time (s)':<10} {'Descents':<10}")
            print("-" * 45)
            for agent_count, stats in results.items():
                print(f"{agent_count:<8} {stats['success_rate']:<11.1f}% {stats['computation_time']:<9.3f} {stats['successful_descents']:<10}")
        
        else:
            print("Usage: python main_orchestration.py [demo|evaluate|scalability] [params]")
    
    else:
        # Default: run demonstration
        demonstrate_system()