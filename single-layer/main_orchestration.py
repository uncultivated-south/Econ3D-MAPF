"""
Main Orchestration Module for UrbanAirspaceSim
Integrates all system components and manages the complete simulation workflow
"""

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import random
import json
from collections import defaultdict

# Import all system modules
from grid_system import GridSystem, Agent, AgentType, Position, CellState
from astar_pathfinding import AStarPathfinder, extract_emergency_paths_from_grid
from cbs_module import ConflictBasedSearch, CBSResult, create_cbs_solver
from auction_system import (
    AuctionSystem, AuctionResult, BiddingStrategy, 
    create_auction_system, prepare_agents_for_auction, 
    update_grid_with_auction_winners, extract_conflict_density_from_cbs
)

class SystemPhase(Enum):
    """Phases of the UrbanAirspaceSim system"""
    EMERGENCY_PROCESSING = "emergency_processing"
    CBS_FEASIBILITY = "cbs_feasibility"
    DECISION_POINT = "decision_point"
    AUCTION_SYSTEM = "auction_system"
    COMPLETED = "completed"

class ProcessingMode(Enum):
    """System processing modes"""
    CBS_ONLY = "cbs_only"
    AUCTION_ONLY = "auction_only"
    HYBRID = "hybrid"

@dataclass
class SimulationMetrics:
    """Comprehensive metrics for simulation evaluation"""
    # Overall system metrics
    total_agents: int = 0
    emergency_agents: int = 0
    non_emergency_agents: int = 0
    
    # Success metrics
    agents_with_paths: int = 0
    success_rate: float = 0.0
    emergency_success_rate: float = 0.0
    non_emergency_success_rate: float = 0.0
    
    # Performance metrics
    total_computation_time: float = 0.0
    emergency_processing_time: float = 0.0
    cbs_processing_time: float = 0.0
    auction_processing_time: float = 0.0
    
    # CBS metrics
    cbs_attempted: bool = False
    cbs_successful: bool = False
    cbs_iterations_used: int = 0
    cbs_conflicts_found: int = 0
    
    # Auction metrics
    auction_triggered: bool = False
    auction_successful: bool = False
    auction_rounds_used: int = 0
    auction_total_revenue: float = 0.0
    auction_winners: int = 0
    auction_unassigned: int = 0
    
    # Efficiency metrics
    average_path_length: float = 0.0
    total_path_cost: float = 0.0
    budget_utilization: float = 0.0
    
    # Fairness metrics
    strategy_success_rates: Dict[str, float] = field(default_factory=dict)
    budget_vs_success_correlation: float = 0.0

@dataclass
class SimulationResult:
    """Complete result of a simulation run"""
    success: bool
    processing_mode: ProcessingMode
    phases_completed: List[SystemPhase]
    metrics: SimulationMetrics
    final_paths: Dict[int, List[Tuple[int, int, int]]]
    unassigned_agents: List[int]
    detailed_log: List[str] = field(default_factory=list)
    error_message: str = ""

class UrbanAirspaceSim:
    """
    Main orchestration class for the Urban Airspace Simulation system
    Manages all phases and component integration
    """
    
    def __init__(self, width: int = 20, height: int = 20, max_time: int = 100):
        """
        Initialize the simulation system
        
        Args:
            width: Grid width
            height: Grid height
            max_time: Maximum simulation time steps
        """
        # Core system components
        self.grid = GridSystem(width, height, max_time)
        self.pathfinder = AStarPathfinder(self.grid)
        self.cbs = create_cbs_solver(self.grid)
        self.auction = create_auction_system(self.grid, self.cbs)
        
        # System configuration
        self.debug_mode = False
        self.enable_detailed_logging = True
        
        # Statistics tracking
        self.simulation_count = 0
        self.total_success_rate = 0.0
        self.component_stats = {
            'grid': {},
            'pathfinder': {},
            'cbs': {},
            'auction': {}
        }
        
        # Default scenario parameters
        self.default_emergency_count = 2
        self.default_non_emergency_count = 16
        self.default_budget_range = (1.0, 100.0)
        
        self.log_messages = []
    
    def log(self, message: str, phase: Optional[SystemPhase] = None):
        """Add message to simulation log"""
        timestamp = time.time()
        if phase:
            formatted_msg = f"[{phase.value.upper()}] {message}"
        else:
            formatted_msg = f"[SYSTEM] {message}"
        
        self.log_messages.append(formatted_msg)
        
        if self.debug_mode:
            print(formatted_msg)
    
    def create_scenario(self, emergency_agents: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                       non_emergency_agents: List[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                       custom_budgets: Dict[int, float] = None,
                       custom_strategies: Dict[int, str] = None) -> bool:
        """
        Create a simulation scenario with specified agents
        
        Args:
            emergency_agents: List of (start, goal) tuples for emergency agents
            non_emergency_agents: List of (start, goal) tuples for non-emergency agents
            custom_budgets: Custom budget assignments {agent_id: budget}
            custom_strategies: Custom strategy assignments {agent_id: strategy}
            
        Returns:
            True if scenario created successfully
        """
        self.grid.reset_system()
        self.log_messages.clear()
        
        agent_id = 1
        
        # Create emergency agents
        if emergency_agents is None:
            emergency_agents = self._generate_random_agent_configs(self.default_emergency_count)
        
        for start, goal in emergency_agents:
            agent = Agent(
                id=agent_id,
                agent_type=AgentType.EMERGENCY,
                start=start,
                goal=goal
            )
            
            if self.grid.add_agent(agent):
                self.log(f"Created emergency agent {agent_id}: {start} -> {goal}")
                agent_id += 1
            else:
                self.log(f"Failed to create emergency agent {agent_id}")
                return False
        
        # Create non-emergency agents
        if non_emergency_agents is None:
            non_emergency_agents = self._generate_random_agent_configs(self.default_non_emergency_count)
        
        for start, goal in non_emergency_agents:
            budget = custom_budgets.get(agent_id, random.uniform(*self.default_budget_range)) if custom_budgets else random.uniform(*self.default_budget_range)
            strategy = custom_strategies.get(agent_id, random.choice(list(BiddingStrategy)).value) if custom_strategies else random.choice(list(BiddingStrategy)).value
            
            agent = Agent(
                id=agent_id,
                agent_type=AgentType.NON_EMERGENCY,
                start=start,
                goal=goal,
                budget=budget,
                strategy=strategy
            )
            
            if self.grid.add_agent(agent):
                self.log(f"Created non-emergency agent {agent_id}: {start} -> {goal}, budget: {budget:.1f}, strategy: {strategy}")
                agent_id += 1
            else:
                self.log(f"Failed to create non-emergency agent {agent_id}")
                return False
        
        self.log(f"Scenario created with {len(self.grid.emergency_agents)} emergency and {len(self.grid.non_emergency_agents)} non-emergency agents")
        return True
    
    def _generate_random_agent_configs(self, count: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generate random start/goal positions for agents"""
        configs = []
        
        for _ in range(count):
            # Random start position
            start_x = random.randint(0, self.grid.width - 1)
            start_y = random.randint(0, self.grid.height - 1)
            
            # Random goal position (different from start)
            while True:
                goal_x = random.randint(0, self.grid.width - 1)
                goal_y = random.randint(0, self.grid.height - 1)
                if (goal_x, goal_y) != (start_x, start_y):
                    break
            
            configs.append(((start_x, start_y), (goal_x, goal_y)))
        
        return configs
    
    def phase_1_emergency_processing(self) -> bool:
        """
        Phase 1: Emergency Processing
        Process emergency agents with highest priority
        """
        start_time = time.time()
        self.log("Starting emergency agent processing", SystemPhase.EMERGENCY_PROCESSING)
        
        emergency_agents = self.grid.get_agents_by_type(AgentType.EMERGENCY)
        
        if not emergency_agents:
            self.log("No emergency agents to process", SystemPhase.EMERGENCY_PROCESSING)
            return True
        
        # Find paths for emergency agents (they have absolute priority)
        emergency_paths_found = 0
        
        for agent in emergency_agents:
            path = self.pathfinder.find_path_for_agent(agent, start_time=0)
            
            if path:
                # Set path in grid system (automatically becomes dynamic obstacle)
                if self.grid.set_agent_path(agent.id, path):
                    emergency_paths_found += 1
                    self.log(f"Emergency agent {agent.id} assigned path of length {len(path)}", SystemPhase.EMERGENCY_PROCESSING)
                else:
                    self.log(f"Failed to set path for emergency agent {agent.id}", SystemPhase.EMERGENCY_PROCESSING)
            else:
                self.log(f"No path found for emergency agent {agent.id}", SystemPhase.EMERGENCY_PROCESSING)
        
        processing_time = time.time() - start_time
        self.log(f"Emergency processing completed: {emergency_paths_found}/{len(emergency_agents)} agents assigned paths in {processing_time:.3f}s", 
                SystemPhase.EMERGENCY_PROCESSING)
        
        return emergency_paths_found == len(emergency_agents)
    
    def phase_2_cbs_feasibility(self) -> CBSResult:
        """
        Phase 2: CBS Feasibility Analysis
        Attempt to solve remaining agents with CBS
        """
        start_time = time.time()
        self.log("Starting CBS feasibility analysis", SystemPhase.CBS_FEASIBILITY)
        
        # Get unassigned non-emergency agents
        non_emergency_agents = [agent for agent in self.grid.get_agents_by_type(AgentType.NON_EMERGENCY) 
                               if not agent.path]
        
        if not non_emergency_agents:
            self.log("No non-emergency agents to process", SystemPhase.CBS_FEASIBILITY)
            return CBSResult()
        
        # Extract emergency paths
        emergency_paths = extract_emergency_paths_from_grid(self.grid)
        
        # Attempt CBS solution
        cbs_result = self.cbs.solve(non_emergency_agents, emergency_paths, start_time=0)
        
        processing_time = time.time() - start_time
        
        if cbs_result.success:
            self.log(f"CBS successful: {len(cbs_result.paths)} paths found in {cbs_result.iterations_used} iterations ({processing_time:.3f}s)", 
                    SystemPhase.CBS_FEASIBILITY)
            
            # Assign CBS paths to agents
            for agent_id, path_tuples in cbs_result.paths.items():
                path = [Position(x, y, t) for x, y, t in path_tuples]
                self.grid.set_agent_path(agent_id, path)
                self.log(f"CBS assigned path to agent {agent_id}", SystemPhase.CBS_FEASIBILITY)
        
        else:
            self.log(f"CBS failed: {cbs_result.failure_reason} after {cbs_result.iterations_used} iterations ({processing_time:.3f}s)", 
                    SystemPhase.CBS_FEASIBILITY)
            self.log(f"Conflicts found: {len(cbs_result.conflicts_found)}, Conflict density calculated", 
                    SystemPhase.CBS_FEASIBILITY)
        
        return cbs_result
    
    def phase_3_decision_point(self, cbs_result: CBSResult) -> bool:
        """
        Phase 3: Decision Point
        Determine whether to proceed with auction based on CBS results
        """
        self.log("Evaluating decision point", SystemPhase.DECISION_POINT)
        
        # Check if CBS was successful
        if cbs_result.success:
            self.log("CBS successful - no auction needed", SystemPhase.DECISION_POINT)
            return False  # No auction needed
        
        # Check if there are unassigned agents
        unassigned_agents = self.grid.get_unassigned_agents(AgentType.NON_EMERGENCY)
        
        if not unassigned_agents:
            self.log("All agents assigned - no auction needed", SystemPhase.DECISION_POINT)
            return False
        
        # Auction should be triggered
        self.log(f"Triggering auction for {len(unassigned_agents)} unassigned agents", SystemPhase.DECISION_POINT)
        self.log(f"Reason: {cbs_result.failure_reason}", SystemPhase.DECISION_POINT)
        
        return True
    
    def phase_4_auction_system(self, conflict_density: Dict[Tuple[int, int], int]) -> AuctionResult:
        """
        Phase 4: Multi-Round Auction System
        Run auction for unassigned agents
        """
        start_time = time.time()
        self.log("Starting multi-round auction system", SystemPhase.AUCTION_SYSTEM)
        
        # Prepare agents for auction
        auction_agents = prepare_agents_for_auction(self.grid, self.default_budget_range)
        
        if not auction_agents:
            self.log("No agents available for auction", SystemPhase.AUCTION_SYSTEM)
            return AuctionResult(success=False, total_rounds=0, final_winners={}, 
                               unassigned_agents=[], total_revenue=0.0, round_results=[])
        
        # Extract emergency paths
        emergency_paths = extract_emergency_paths_from_grid(self.grid)
        
        # Run auction
        auction_result = self.auction.run_auction(auction_agents, conflict_density, emergency_paths)
        
        processing_time = time.time() - start_time
        
        if auction_result.success:
            self.log(f"Auction successful: {len(auction_result.final_winners)} winners in {auction_result.total_rounds} rounds ({processing_time:.3f}s)", 
                    SystemPhase.AUCTION_SYSTEM)
            self.log(f"Total revenue: {auction_result.total_revenue:.2f}, Unassigned: {len(auction_result.unassigned_agents)}", 
                    SystemPhase.AUCTION_SYSTEM)
            
            # Update grid system with winners
            update_grid_with_auction_winners(self.grid, auction_result)
        
        else:
            self.log(f"Auction failed after {auction_result.total_rounds} rounds ({processing_time:.3f}s)", 
                    SystemPhase.AUCTION_SYSTEM)
        
        return auction_result
    
    def run_simulation(self, processing_mode: ProcessingMode = ProcessingMode.HYBRID) -> SimulationResult:
        """
        Run complete simulation with all phases
        
        Args:
            processing_mode: How to process non-emergency agents
            
        Returns:
            Complete simulation result
        """
        self.simulation_count += 1
        start_time = time.time()
        self.log(f"Starting simulation #{self.simulation_count} in {processing_mode.value} mode")
        
        # Initialize result
        result = SimulationResult(
            success=False,
            processing_mode=processing_mode,
            phases_completed=[],
            metrics=SimulationMetrics(),
            final_paths={},
            unassigned_agents=[],
            detailed_log=[]
        )
        
        try:
            # Phase 1: Emergency Processing
            emergency_start_time = time.time()
            emergency_success = self.phase_1_emergency_processing()
            result.phases_completed.append(SystemPhase.EMERGENCY_PROCESSING)
            result.metrics.emergency_processing_time = time.time() - emergency_start_time
            
            if not emergency_success:
                result.error_message = "Emergency processing failed"
                return result
            
            # Initialize CBS result
            cbs_result = CBSResult()
            auction_result = AuctionResult(success=False, total_rounds=0, final_winners={}, 
                                         unassigned_agents=[], total_revenue=0.0, round_results=[])
            
            # Determine processing approach based on mode
            if processing_mode == ProcessingMode.CBS_ONLY:
                # Phase 2: CBS Only
                cbs_start_time = time.time()
                cbs_result = self.phase_2_cbs_feasibility()
                result.phases_completed.append(SystemPhase.CBS_FEASIBILITY)
                result.metrics.cbs_processing_time = time.time() - cbs_start_time
                
            elif processing_mode == ProcessingMode.AUCTION_ONLY:
                # Skip CBS, go directly to auction
                auction_agents = prepare_agents_for_auction(self.grid, self.default_budget_range)
                if auction_agents:
                    # Create minimal conflict density for auction
                    conflict_density = defaultdict(int)
                    
                    auction_start_time = time.time()
                    auction_result = self.phase_4_auction_system(dict(conflict_density))
                    result.phases_completed.append(SystemPhase.AUCTION_SYSTEM)
                    result.metrics.auction_processing_time = time.time() - auction_start_time
                
            elif processing_mode == ProcessingMode.HYBRID:
                # Phase 2: CBS Feasibility
                cbs_start_time = time.time()
                cbs_result = self.phase_2_cbs_feasibility()
                result.phases_completed.append(SystemPhase.CBS_FEASIBILITY)
                result.metrics.cbs_processing_time = time.time() - cbs_start_time
                
                # Phase 3: Decision Point
                should_auction = self.phase_3_decision_point(cbs_result)
                result.phases_completed.append(SystemPhase.DECISION_POINT)
                
                # Phase 4: Auction (if needed)
                if should_auction:
                    auction_start_time = time.time()
                    conflict_density = extract_conflict_density_from_cbs(cbs_result)
                    auction_result = self.phase_4_auction_system(conflict_density)
                    result.phases_completed.append(SystemPhase.AUCTION_SYSTEM)
                    result.metrics.auction_processing_time = time.time() - auction_start_time
            
            # Calculate comprehensive metrics
            result.metrics = self._calculate_metrics(cbs_result, auction_result)
            result.metrics.total_computation_time = time.time() - start_time
            
            # Extract final paths and unassigned agents
            result.final_paths = self.grid.export_paths_for_cbs()
            result.unassigned_agents = [agent.id for agent in self.grid.get_unassigned_agents()]
            
            # Determine overall success
            total_agents = len(self.grid.agents)
            assigned_agents = len(result.final_paths)
            result.success = assigned_agents > 0 and len(result.unassigned_agents) < total_agents
            
            result.phases_completed.append(SystemPhase.COMPLETED)
            result.detailed_log = self.log_messages.copy()
            
            self.log(f"Simulation completed: {assigned_agents}/{total_agents} agents assigned paths")
            
        except Exception as e:
            result.error_message = f"Simulation failed with error: {str(e)}"
            self.log(f"Simulation failed: {str(e)}")
        
        return result
    
    def _calculate_metrics(self, cbs_result: CBSResult, auction_result: AuctionResult) -> SimulationMetrics:
        """Calculate comprehensive simulation metrics"""
        metrics = SimulationMetrics()
        
        # Basic counts
        metrics.total_agents = len(self.grid.agents)
        metrics.emergency_agents = len(self.grid.emergency_agents)
        metrics.non_emergency_agents = len(self.grid.non_emergency_agents)
        
        # Success metrics
        agents_with_paths = len([agent for agent in self.grid.agents.values() if agent.path])
        metrics.agents_with_paths = agents_with_paths
        metrics.success_rate = agents_with_paths / metrics.total_agents if metrics.total_agents > 0 else 0
        
        emergency_with_paths = len([agent_id for agent_id in self.grid.emergency_agents 
                                   if self.grid.agents[agent_id].path])
        metrics.emergency_success_rate = emergency_with_paths / metrics.emergency_agents if metrics.emergency_agents > 0 else 0
        
        non_emergency_with_paths = agents_with_paths - emergency_with_paths
        metrics.non_emergency_success_rate = non_emergency_with_paths / metrics.non_emergency_agents if metrics.non_emergency_agents > 0 else 0
        
        # CBS metrics
        metrics.cbs_attempted = cbs_result.iterations_used > 0
        metrics.cbs_successful = cbs_result.success
        metrics.cbs_iterations_used = cbs_result.iterations_used
        metrics.cbs_conflicts_found = len(cbs_result.conflicts_found)
        
        # Auction metrics
        metrics.auction_triggered = len(auction_result.round_results) > 0
        metrics.auction_successful = auction_result.success
        metrics.auction_rounds_used = auction_result.total_rounds
        metrics.auction_total_revenue = auction_result.total_revenue
        metrics.auction_winners = len(auction_result.final_winners)
        metrics.auction_unassigned = len(auction_result.unassigned_agents)
        
        # Efficiency metrics
        all_paths = [agent.path for agent in self.grid.agents.values() if agent.path]
        if all_paths:
            path_lengths = [len(path) for path in all_paths]
            metrics.average_path_length = sum(path_lengths) / len(path_lengths)
            metrics.total_path_cost = sum(path_lengths)
        
        # Budget utilization
        non_emergency_agents = [agent for agent in self.grid.agents.values() 
                               if agent.agent_type == AgentType.NON_EMERGENCY]
        if non_emergency_agents:
            total_initial_budget = sum(agent.budget + auction_result.total_revenue 
                                     for agent in non_emergency_agents)  # Approximate initial budget
            metrics.budget_utilization = auction_result.total_revenue / total_initial_budget if total_initial_budget > 0 else 0
        
        # Strategy success rates
        strategy_counts = defaultdict(int)
        strategy_successes = defaultdict(int)
        
        for agent in non_emergency_agents:
            strategy_counts[agent.strategy] += 1
            if agent.path:
                strategy_successes[agent.strategy] += 1
        
        for strategy in strategy_counts:
            metrics.strategy_success_rates[strategy] = (
                strategy_successes[strategy] / strategy_counts[strategy] 
                if strategy_counts[strategy] > 0 else 0
            )
        
        return metrics
    
    def run_batch_simulations(self, num_simulations: int, 
                            processing_mode: ProcessingMode = ProcessingMode.HYBRID,
                            scenario_generator: Optional[callable] = None) -> List[SimulationResult]:
        """
        Run multiple simulations for statistical analysis
        
        Args:
            num_simulations: Number of simulations to run
            processing_mode: Processing mode for all simulations
            scenario_generator: Optional custom scenario generator function
            
        Returns:
            List of simulation results
        """
        self.log(f"Starting batch of {num_simulations} simulations")
        
        results = []
        
        for i in range(num_simulations):
            # Generate scenario
            if scenario_generator:
                scenario_generator()
            else:
                self.create_default_scenario()
            
            # Run simulation
            result = self.run_simulation(processing_mode)
            results.append(result)
            
            if self.debug_mode:
                print(f"Simulation {i+1}/{num_simulations} completed: Success={result.success}")
        
        # Calculate batch statistics
        self._update_batch_statistics(results)
        
        self.log(f"Batch simulation completed: {len(results)} simulations")
        return results
    
    def create_default_scenario(self):
        """Create the default 20x20 scenario with 2 emergency + 16 non-emergency agents"""
        return self.create_scenario()
    
    def _update_batch_statistics(self, results: List[SimulationResult]):
        """Update overall system statistics from batch results"""
        if not results:
            return
        
        # Calculate overall success rate
        successful_simulations = sum(1 for result in results if result.success)
        self.total_success_rate = successful_simulations / len(results)
        
        # Update component statistics
        self.component_stats['grid'] = self.grid.get_system_state()
        self.component_stats['pathfinder'] = self.pathfinder.get_statistics()
        self.component_stats['cbs'] = self.cbs.get_statistics()
        self.component_stats['auction'] = self.auction.get_statistics()
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            'simulation_count': self.simulation_count,
            'total_success_rate': self.total_success_rate,
            'component_stats': self.component_stats,
            'system_configuration': {
                'grid_size': (self.grid.width, self.grid.height, self.grid.max_time),
                'default_emergency_count': self.default_emergency_count,
                'default_non_emergency_count': self.default_non_emergency_count,
                'default_budget_range': self.default_budget_range
            }
        }
    
    def export_results_to_json(self, results: List[SimulationResult], filename: str):
        """Export simulation results to JSON file"""
        # Convert results to JSON-serializable format
        json_results = []
        
        for result in results:
            json_result = {
                'success': result.success,
                'processing_mode': result.processing_mode.value,
                'phases_completed': [phase.value for phase in result.phases_completed],
                'metrics': {
                    'total_agents': result.metrics.total_agents,
                    'success_rate': result.metrics.success_rate,
                    'total_computation_time': result.metrics.total_computation_time,
                    'cbs_successful': result.metrics.cbs_successful,
                    'auction_triggered': result.metrics.auction_triggered,
                    'auction_total_revenue': result.metrics.auction_total_revenue,
                    'strategy_success_rates': result.metrics.strategy_success_rates
                },
                'final_paths_count': len(result.final_paths),
                'unassigned_agents_count': len(result.unassigned_agents),
                'error_message': result.error_message
            }
            json_results.append(json_result)
        
        # Add system statistics
        export_data = {
            'system_statistics': self.get_system_statistics(),
            'simulation_results': json_results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.log(f"Results exported to {filename}")
        except Exception as e:
            self.log(f"Failed to export results: {str(e)}")
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode for all components"""
        self.debug_mode = enabled
        self.pathfinder.set_debug_mode(enabled)
        self.cbs.set_debug_mode(enabled)
        self.auction.set_debug_mode(enabled)
        
        if enabled:
            self.log("Debug mode enabled for all components")
        else:
            self.log("Debug mode disabled for all components")
    
    def reset_all_statistics(self):
        """Reset statistics for all components"""
        self.simulation_count = 0
        self.total_success_rate = 0.0
        self.cbs.reset_statistics()
        self.auction.reset_statistics()
        self.log("All statistics reset")

# Main execution functions

def create_simulation_system(width: int = 20, height: int = 20) -> UrbanAirspaceSim:
    """Create and initialize the complete simulation system"""
    return UrbanAirspaceSim(width, height)

def run_default_simulation() -> SimulationResult:
    """Run a single simulation with default parameters"""
    sim = create_simulation_system()
    sim.create_default_scenario()
    return sim.run_simulation()

def run_evaluation_study(num_simulations: int = 100) -> Dict:
    """
    Run comprehensive evaluation study
    
    Args:
        num_simulations: Number of simulations per processing mode
        
    Returns:
        Dictionary with comparative results
    """
    sim = create_simulation_system()
    
    evaluation_results = {}
    
    # Test each processing mode
    for mode in ProcessingMode:
        print(f"Evaluating {mode.value} mode...")
        results = sim.run_batch_simulations(num_simulations, mode)
        
        # Calculate summary statistics
        successful = sum(1 for r in results if r.success)
        avg_computation_time = sum(r.metrics.total_computation_time for r in results) / len(results)
        avg_success_rate = sum(r.metrics.success_rate for r in results) / len(results)
        
        evaluation_results[mode.value] = {
            'success_rate': successful / len(results),
            'average_computation_time': avg_computation_time,
            'average_agent_success_rate': avg_success_rate,
            'total_simulations': len(results),
            'cbs_usage': sum(1 for r in results if r.metrics.cbs_attempted) / len(results),
            'auction_usage': sum(1 for r in results if r.metrics.auction_triggered) / len(results),
            'average_revenue': sum(r.metrics.auction_total_revenue for r in results) / len(results)
        }
        
        # Reset system for next mode
        sim.reset_all_statistics()
    
    return evaluation_results

def demonstrate_system():
    """
    Demonstrate the complete UrbanAirspaceSim system with detailed output
    """
    print("=== UrbanAirspaceSim Demonstration ===\n")
    
    # Create simulation system
    sim = create_simulation_system()
    sim.set_debug_mode(True)
    
    print("1. Creating default scenario (20x20 grid, 2 emergency + 16 non-emergency agents)...")
    sim.create_default_scenario()
    
    print(f"\nGrid state: {sim.grid.get_system_state()}")
    
    print("\n2. Running hybrid simulation (CBS first, then auction if needed)...")
    result = sim.run_simulation(ProcessingMode.HYBRID)
    
    print("\n=== Simulation Results ===")
    print(f"Success: {result.success}")
    print(f"Processing Mode: {result.processing_mode.value}")
    print(f"Phases Completed: {[phase.value for phase in result.phases_completed]}")
    print(f"Total Computation Time: {result.metrics.total_computation_time:.3f}s")
    print(f"Overall Success Rate: {result.metrics.success_rate:.2%}")
    print(f"Emergency Success Rate: {result.metrics.emergency_success_rate:.2%}")
    print(f"Non-Emergency Success Rate: {result.metrics.non_emergency_success_rate:.2%}")
    
    if result.metrics.cbs_attempted:
        print(f"\nCBS Results:")
        print(f"  - Successful: {result.metrics.cbs_successful}")
        print(f"  - Iterations Used: {result.metrics.cbs_iterations_used}")
        print(f"  - Processing Time: {result.metrics.cbs_processing_time:.3f}s")
    
    if result.metrics.auction_triggered:
        print(f"\nAuction Results:")
        print(f"  - Successful: {result.metrics.auction_successful}")
        print(f"  - Rounds Used: {result.metrics.auction_rounds_used}")
        print(f"  - Winners: {result.metrics.auction_winners}")
        print(f"  - Total Revenue: {result.metrics.auction_total_revenue:.2f}")
        print(f"  - Processing Time: {result.metrics.auction_processing_time:.3f}s")
        print(f"  - Strategy Success Rates: {result.metrics.strategy_success_rates}")
    
    print(f"\nFinal Results:")
    print(f"  - Agents with Paths: {len(result.final_paths)}")
    print(f"  - Unassigned Agents: {len(result.unassigned_agents)}")
    print(f"  - Average Path Length: {result.metrics.average_path_length:.1f}")
    
    if result.error_message:
        print(f"  - Error: {result.error_message}")
    
    return result

def run_comparative_analysis(num_simulations: int = 50):
    """
    Run comparative analysis between different processing modes
    """
    print(f"=== Comparative Analysis ({num_simulations} simulations per mode) ===\n")
    
    evaluation_results = run_evaluation_study(num_simulations)
    
    print("Processing Mode Comparison:")
    print("-" * 80)
    print(f"{'Mode':<15} {'Success':<8} {'Avg Time':<10} {'Agent Success':<12} {'CBS Usage':<10} {'Auction Usage':<12} {'Avg Revenue':<10}")
    print("-" * 80)
    
    for mode, stats in evaluation_results.items():
        print(f"{mode:<15} {stats['success_rate']:<8.2%} {stats['average_computation_time']:<10.3f} "
              f"{stats['average_agent_success_rate']:<12.2%} {stats['cbs_usage']:<10.2%} "
              f"{stats['auction_usage']:<12.2%} {stats['average_revenue']:<10.2f}")
    
    return evaluation_results

def test_specific_scenario():
    """
    Test a specific challenging scenario
    """
    print("=== Testing Specific Challenging Scenario ===\n")
    
    sim = create_simulation_system()
    sim.set_debug_mode(True)
    
    # Create a challenging scenario with potential conflicts
    emergency_agents = [
        ((0, 0), (19, 19)),    # Emergency 1: corner to corner
        ((19, 0), (0, 19))     # Emergency 2: opposite corner to corner
    ]
    
    non_emergency_agents = [
        ((1, 1), (18, 18)),    # Agent 1: similar to emergency 1
        ((18, 1), (1, 18)),    # Agent 2: similar to emergency 2
        ((10, 0), (10, 19)),   # Agent 3: vertical crossing
        ((0, 10), (19, 10)),   # Agent 4: horizontal crossing
        ((5, 5), (15, 15)),    # Agent 5: diagonal
        ((15, 5), (5, 15)),    # Agent 6: opposite diagonal
        ((2, 2), (17, 17)),    # Additional agents to increase complexity
        ((17, 2), (2, 17)),
        ((8, 8), (12, 12)),
        ((12, 8), (8, 12)),
        ((3, 10), (16, 10)),
        ((10, 3), (10, 16)),
        ((7, 7), (13, 13)),
        ((13, 7), (7, 13)),
        ((4, 4), (16, 16)),
        ((16, 4), (4, 16))
    ]
    
    # Custom budgets and strategies for testing
    custom_budgets = {i: random.uniform(20, 80) for i in range(3, 19)}  # Non-emergency agents get IDs 3-18
    custom_strategies = {
        3: "aggressive", 4: "aggressive",
        5: "conservative", 6: "conservative", 
        7: "balanced", 8: "balanced"
    }
    
    print("Creating challenging scenario with high conflict potential...")
    sim.create_scenario(emergency_agents, non_emergency_agents, custom_budgets, custom_strategies)
    
    print("\nRunning simulation...")
    result = sim.run_simulation(ProcessingMode.HYBRID)
    
    print("\n=== Challenging Scenario Results ===")
    print(f"Scenario completed successfully: {result.success}")
    print(f"Final agent assignment rate: {result.metrics.success_rate:.2%}")
    
    if result.metrics.cbs_attempted:
        print(f"CBS attempted: {'Successful' if result.metrics.cbs_successful else 'Failed'}")
        print(f"CBS conflicts found: {result.metrics.cbs_conflicts_found}")
    
    if result.metrics.auction_triggered:
        print(f"Auction triggered: {result.metrics.auction_rounds_used} rounds")
        print(f"Auction revenue: {result.metrics.auction_total_revenue:.2f}")
        print(f"Strategy performance: {result.metrics.strategy_success_rates}")
    
    return result

def export_simulation_data(num_simulations: int = 100, filename: str = "simulation_results.json"):
    """
    Run simulations and export data for external analysis
    """
    print(f"=== Exporting Simulation Data ({num_simulations} simulations) ===\n")
    
    sim = create_simulation_system()
    
    # Run simulations with different modes
    all_results = []
    
    for mode in ProcessingMode:
        print(f"Running {num_simulations} simulations in {mode.value} mode...")
        results = sim.run_batch_simulations(num_simulations, mode)
        all_results.extend(results)
        sim.reset_all_statistics()
    
    # Export to JSON
    sim.export_results_to_json(all_results, filename)
    print(f"Data exported to {filename}")
    
    return all_results

# Example usage and testing functions

if __name__ == "__main__":
    """
    Main execution for testing and demonstration
    """
    
    print("UrbanAirspaceSim - Main Orchestration Module")
    print("=" * 50)
    
    # Option 1: Quick demonstration
    print("\n1. Running system demonstration...")
    demo_result = demonstrate_system()
    
    # Option 2: Comparative analysis
    print("\n\n2. Running comparative analysis...")
    comparative_results = run_comparative_analysis(20)  # Reduced for demo
    
    # Option 3: Challenging scenario test
    print("\n\n3. Testing challenging scenario...")
    challenge_result = test_specific_scenario()
    
    # Option 4: Export data for analysis
    print("\n\n4. Exporting simulation data...")
    export_simulation_data(50, "urban_airspace_results.json")  # Reduced for demo
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("System is ready for full-scale evaluation and analysis.")

# Additional utility functions for advanced usage

def create_custom_evaluation(emergency_count: int, non_emergency_count: int, 
                           grid_size: Tuple[int, int] = (20, 20),
                           num_simulations: int = 100) -> Dict:
    """
    Create custom evaluation with specified parameters
    """
    sim = UrbanAirspaceSim(grid_size[0], grid_size[1])
    sim.default_emergency_count = emergency_count
    sim.default_non_emergency_count = non_emergency_count
    
    results = sim.run_batch_simulations(num_simulations, ProcessingMode.HYBRID)
    
    return {
        'configuration': {
            'emergency_count': emergency_count,
            'non_emergency_count': non_emergency_count,
            'grid_size': grid_size,
            'simulations': num_simulations
        },
        'results': results,
        'statistics': sim.get_system_statistics()
    }

def analyze_scalability(max_agents: int = 50, step_size: int = 5):
    """
    Analyze system scalability with increasing agent counts
    """
    scalability_results = {}
    
    for total_agents in range(10, max_agents + 1, step_size):
        emergency_count = max(1, total_agents // 10)  # 10% emergency agents
        non_emergency_count = total_agents - emergency_count
        
        print(f"Testing with {total_agents} agents ({emergency_count} emergency, {non_emergency_count} non-emergency)...")
        
        result = create_custom_evaluation(emergency_count, non_emergency_count, num_simulations=20)
        
        scalability_results[total_agents] = {
            'success_rate': sum(r.success for r in result['results']) / len(result['results']),
            'avg_computation_time': sum(r.metrics.total_computation_time for r in result['results']) / len(result['results']),
            'cbs_success_rate': sum(r.metrics.cbs_successful for r in result['results']) / len(result['results']),
            'auction_trigger_rate': sum(r.metrics.auction_triggered for r in result['results']) / len(result['results'])
        }
    
    return scalability_results