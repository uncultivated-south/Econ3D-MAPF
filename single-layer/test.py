"""
Integration Test Module for UrbanAirspaceSim
Comprehensive testing of CBS-only, Auction-only, and Hybrid approaches
Tests performance, fairness, and robustness across different agent counts
"""

import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
from decimal import Decimal
import statistics

# Import all refactored modules
from grid_system import GridSystem, Agent, AgentType, Position
from astar_pathfinding import AStarPathfinder, PathfindingConfig, create_conservative_config, create_fast_config
from cbs_module import ConflictBasedSearch, create_cbs_solver
from auction_system import AuctionSystem, create_auction_system, BiddingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestScenario:
    """Represents a single test scenario with fixed parameters"""
    scenario_id: str
    grid_width: int = 20
    grid_height: int = 20
    num_static_obstacles: int = 12
    num_emergency_agents: int = 2
    num_regular_agents: int = 16
    budget_min: float = 1.0
    budget_max: float = 150.0
    pareto_alpha: float = 1.16  # Shape parameter for Pareto distribution
    
    # Generated data
    static_obstacles: List[Tuple[int, int]] = field(default_factory=list)
    emergency_agents_data: List[Dict] = field(default_factory=list)  # start, goal for emergency agents
    regular_agents_data: List[Dict] = field(default_factory=list)    # start, goal, budget for regular agents
    
    def __post_init__(self):
        if not self.scenario_id:
            self.scenario_id = f"scenario_{self.num_regular_agents}_{time.time()}"

@dataclass
class AlgorithmResult:
    """Results from running a specific algorithm on a test scenario"""
    algorithm_name: str
    success: bool
    computation_time: float
    successful_agents: List[int]  # IDs of agents that got paths
    failed_agents: List[int]     # IDs of agents that didn't get paths
    
    # CBS-specific metrics
    cbs_iterations: int = 0
    cbs_conflicts_found: int = 0
    
    # Auction-specific metrics
    auction_revenue: Decimal = Decimal('0')
    auction_rounds: int = 0
    winning_bids: Dict[int, Decimal] = field(default_factory=dict)
    
    # Path quality metrics
    total_path_cost: float = 0.0
    average_path_length: float = 0.0
    
    # Additional metadata
    emergency_paths: Dict[int, List[Position]] = field(default_factory=dict)
    regular_agent_paths: Dict[int, List[Position]] = field(default_factory=dict)

@dataclass 
class FairnessMetrics:
    """Fairness analysis metrics"""
    success_rate_by_quartile: List[float] = field(default_factory=list)  # Q1, Q2, Q3, Q4 success rates
    gini_coefficient: float = 0.0
    budget_correlation_coefficient: float = 0.0
    
    # Additional fairness measures
    wealth_concentration_index: float = 0.0  # Share of success by top 20% of budgets
    budget_utilization_rate: float = 0.0    # For auction scenarios

@dataclass
class TestResult:
    """Comprehensive result for a single test scenario across all algorithms"""
    scenario: TestScenario
    results: Dict[str, AlgorithmResult] = field(default_factory=dict)  # algorithm_name -> result
    fairness_metrics: Dict[str, FairnessMetrics] = field(default_factory=dict)
    
    def get_comparison_summary(self) -> Dict:
        """Get summary statistics for algorithm comparison"""
        summary = {
            'scenario_id': self.scenario.scenario_id,
            'num_regular_agents': self.scenario.num_regular_agents,
            'algorithms': {}
        }
        
        for alg_name, result in self.results.items():
            summary['algorithms'][alg_name] = {
                'success_rate': len(result.successful_agents) / self.scenario.num_regular_agents,
                'computation_time': result.computation_time,
                'average_path_length': result.average_path_length,
                'revenue': float(result.auction_revenue) if result.auction_revenue else 0.0
            }
        
        return summary

class TestScenarioGenerator:
    """Generates consistent test scenarios with proper randomization"""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    def generate_pareto_budgets(self, n: int, min_budget: float, max_budget: float, 
                               alpha: float = 1.16) -> List[float]:
        """Generate budgets following Pareto distribution"""
        # Generate Pareto samples
        pareto_samples = self.np_rng.pareto(alpha, n)
        
        # Scale to desired range [min_budget, max_budget]
        min_sample, max_sample = np.min(pareto_samples), np.max(pareto_samples)
        
        if max_sample > min_sample:
            scaled_samples = min_budget + (max_budget - min_budget) * (
                (pareto_samples - min_sample) / (max_sample - min_sample)
            )
        else:
            scaled_samples = np.full(n, (min_budget + max_budget) / 2)
        
        return scaled_samples.tolist()
    
    def generate_static_obstacles(self, width: int, height: int, count: int) -> List[Tuple[int, int]]:
        """Generate random static obstacle positions"""
        obstacles = []
        max_attempts = count * 10  # Prevent infinite loops
        attempts = 0
        
        while len(obstacles) < count and attempts < max_attempts:
            x = self.rng.randint(0, width - 1)
            y = self.rng.randint(0, height - 1)
            
            if (x, y) not in obstacles:
                obstacles.append((x, y))
            
            attempts += 1
        
        return obstacles
    
    def generate_agent_positions(self, width: int, height: int, obstacles: List[Tuple[int, int]], 
                                count: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generate start and goal positions for agents, avoiding obstacles"""
        obstacle_set = set(obstacles)
        valid_positions = [
            (x, y) for x in range(width) for y in range(height)
            if (x, y) not in obstacle_set
        ]
        
        if len(valid_positions) < count * 2:
            raise ValueError(f"Not enough valid positions for {count} agents")
        
        agent_positions = []
        used_positions = set()
        
        for _ in range(count):
            # Select start position
            available_starts = [pos for pos in valid_positions if pos not in used_positions]
            start = self.rng.choice(available_starts)
            used_positions.add(start)
            
            # Select goal position (different from start)
            available_goals = [pos for pos in valid_positions if pos not in used_positions and pos != start]
            goal = self.rng.choice(available_goals)
            used_positions.add(goal)
            
            agent_positions.append((start, goal))
        
        return agent_positions
    
    def generate_scenario(self, num_regular_agents: int, seed_offset: int = 0) -> TestScenario:
        """Generate a complete test scenario"""
        # Set seed with offset for variation
        current_seed = self.rng.getstate()
        self.rng.seed(self.rng.getrandbits(32) + seed_offset)
        self.np_rng.seed(self.rng.getrandbits(32))
        
        try:
            scenario = TestScenario(
                scenario_id=f"agents_{num_regular_agents}_seed_{seed_offset}",
                num_regular_agents=num_regular_agents
            )
            
            # Generate static obstacles
            scenario.static_obstacles = self.generate_static_obstacles(
                scenario.grid_width, scenario.grid_height, scenario.num_static_obstacles
            )
            
            # Generate emergency agent positions
            emergency_positions = self.generate_agent_positions(
                scenario.grid_width, scenario.grid_height, scenario.static_obstacles,
                scenario.num_emergency_agents
            )
            
            scenario.emergency_agents_data = [
                {'start': start, 'goal': goal} 
                for start, goal in emergency_positions
            ]
            
            # Generate regular agent positions
            regular_positions = self.generate_agent_positions(
                scenario.grid_width, scenario.grid_height, scenario.static_obstacles,
                scenario.num_regular_agents
            )
            
            # Generate budgets
            budgets = self.generate_pareto_budgets(
                scenario.num_regular_agents, scenario.budget_min, scenario.budget_max, scenario.pareto_alpha
            )
            
            scenario.regular_agents_data = [
                {'start': start, 'goal': goal, 'budget': budget}
                for (start, goal), budget in zip(regular_positions, budgets)
            ]
            
            return scenario
            
        finally:
            # Restore original random state
            self.rng.setstate(current_seed)
    
    def generate_scenario_batch(self, num_regular_agents: int, batch_size: int) -> List[TestScenario]:
        """Generate multiple scenarios with the same agent count"""
        return [
            self.generate_scenario(num_regular_agents, seed_offset=i)
            for i in range(batch_size)
        ]

class TestRunner:
    """Runs algorithms on test scenarios and collects results"""
    
    def __init__(self, cbs_iteration_limit: int = 4000, max_auction_rounds: int = 5):
        self.cbs_iteration_limit = cbs_iteration_limit
        self.max_auction_rounds = max_auction_rounds
        
    def setup_grid_system(self, scenario: TestScenario) -> GridSystem:
        """Set up grid system with scenario parameters"""
        grid = GridSystem(scenario.grid_width, scenario.grid_height)
        
        # Add static obstacles
        for x, y in scenario.static_obstacles:
            grid.add_static_obstacle(x, y)
        
        return grid
    
    def create_agents(self, scenario: TestScenario) -> Tuple[List[Agent], List[Agent]]:
        """Create emergency and regular agents from scenario data"""
        emergency_agents = []
        regular_agents = []
        
        # Create emergency agents
        for i, data in enumerate(scenario.emergency_agents_data):
            agent = Agent(
                id=f"emergency_{i}",
                agent_type=AgentType.EMERGENCY,
                start=data['start'],
                goal=data['goal'],
                priority=100  # High priority
            )
            emergency_agents.append(agent)
        
        # Create regular agents
        for i, data in enumerate(scenario.regular_agents_data):
            # Assign random bidding strategies
            strategies = list(BiddingStrategy)
            strategy = random.choice(strategies)
            
            agent = Agent(
                id=i,  # Use integer IDs for regular agents
                agent_type=AgentType.NON_EMERGENCY,
                start=data['start'],
                goal=data['goal'],
                budget=data['budget'],
                strategy=strategy.value,
                priority=1  # Normal priority
            )
            regular_agents.append(agent)
        
        return emergency_agents, regular_agents
    
    def plan_emergency_agents(self, grid: GridSystem, emergency_agents: List[Agent]) -> Dict[int, List[Position]]:
        """Plan paths for emergency agents using CBS (consistent across all tests)"""
        if not emergency_agents:
            return {}
        
        # Add emergency agents to grid
        for agent in emergency_agents:
            grid.add_agent(agent)
        
        # Create pathfinder and CBS solver
        pathfinder = AStarPathfinder(grid, create_conservative_config())
        cbs = create_cbs_solver(grid, pathfinder)
        
        # Plan emergency paths
        result = cbs.solve(emergency_agents)
        
        emergency_paths = {}
        if result.success:
            emergency_paths = result.paths
            # Set paths in grid system
            for agent_id, path in emergency_paths.items():
                grid.set_agent_path(agent_id, path)
        else:
            logger.warning("Failed to plan emergency agent paths")
        
        return emergency_paths
    
    def run_cbs_only(self, grid: GridSystem, regular_agents: List[Agent]) -> AlgorithmResult:
        """Run CBS-only algorithm"""
        start_time = time.time()
        
        # Add regular agents to grid
        for agent in regular_agents:
            grid.add_agent(agent)
        
        # Create CBS solver with iteration limit
        pathfinder = AStarPathfinder(grid, create_conservative_config())
        cbs = create_cbs_solver(grid, pathfinder)
        cbs.max_iterations = self.cbs_iteration_limit
        
        # Run CBS
        result = cbs.solve(regular_agents)
        
        computation_time = time.time() - start_time
        
        # Extract results
        successful_agents = list(result.paths.keys()) if result.success else []
        failed_agents = [a.id for a in regular_agents if a.id not in successful_agents]
        
        # Calculate path metrics
        total_cost = sum(len(path) for path in result.paths.values())
        avg_length = total_cost / len(result.paths) if result.paths else 0.0
        
        return AlgorithmResult(
            algorithm_name="CBS-only",
            success=result.success,
            computation_time=computation_time,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            cbs_iterations=result.iterations_used,
            cbs_conflicts_found=len(result.conflicts_found),
            total_path_cost=total_cost,
            average_path_length=avg_length,
            regular_agent_paths=result.paths
        )
    
    def run_auction_only(self, grid: GridSystem, regular_agents: List[Agent], 
                        conflict_density: Dict[Tuple[int, int], int]) -> AlgorithmResult:
        """Run auction-only algorithm (CBS only for pricing initialization)"""
        start_time = time.time()
        
        # Add regular agents to grid
        for agent in regular_agents:
            grid.add_agent(agent)
        
        # Create auction system
        pathfinder = AStarPathfinder(grid, create_fast_config())
        cbs = create_cbs_solver(grid, pathfinder)
        auction = create_auction_system(grid, cbs, pathfinder)
        auction.max_rounds = self.max_auction_rounds
        
        # Run auction
        emergency_paths = grid.get_emergency_paths()
        auction_result = auction.run_auction(regular_agents, conflict_density, emergency_paths)
        
        computation_time = time.time() - start_time
        
        # Extract results
        successful_agents = list(auction_result.final_winners.keys())
        failed_agents = auction_result.unassigned_agents
        
        # Calculate path metrics
        total_cost = sum(len(path) for path in auction_result.final_winners.values())
        avg_length = total_cost / len(auction_result.final_winners) if auction_result.final_winners else 0.0
        
        # Extract winning bids
        winning_bids = {}
        for round_result in auction_result.round_results:
            for agent_id, bid in round_result.winning_bids.items():
                winning_bids[agent_id] = bid.total_bid_amount
        
        return AlgorithmResult(
            algorithm_name="Auction-only",
            success=auction_result.success,
            computation_time=computation_time,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            auction_revenue=auction_result.total_revenue,
            auction_rounds=auction_result.total_rounds,
            winning_bids=winning_bids,
            total_path_cost=total_cost,
            average_path_length=avg_length,
            regular_agent_paths=auction_result.final_winners
        )
    
    def run_hybrid(self, grid: GridSystem, regular_agents: List[Agent]) -> AlgorithmResult:
        """Run hybrid CBS→Auction algorithm"""
        start_time = time.time()
        
        # Add regular agents to grid
        for agent in regular_agents:
            grid.add_agent(agent)
        
        # First, try CBS
        pathfinder = AStarPathfinder(grid, create_conservative_config())
        cbs = create_cbs_solver(grid, pathfinder)
        cbs.max_iterations = self.cbs_iteration_limit
        
        cbs_result = cbs.solve(regular_agents)
        
        if cbs_result.success:
            # CBS succeeded - return CBS result
            computation_time = time.time() - start_time
            
            successful_agents = list(cbs_result.paths.keys())
            failed_agents = [a.id for a in regular_agents if a.id not in successful_agents]
            
            total_cost = sum(len(path) for path in cbs_result.paths.values())
            avg_length = total_cost / len(cbs_result.paths) if cbs_result.paths else 0.0
            
            return AlgorithmResult(
                algorithm_name="Hybrid (CBS)",
                success=True,
                computation_time=computation_time,
                successful_agents=successful_agents,
                failed_agents=failed_agents,
                cbs_iterations=cbs_result.iterations_used,
                cbs_conflicts_found=len(cbs_result.conflicts_found),
                total_path_cost=total_cost,
                average_path_length=avg_length,
                regular_agent_paths=cbs_result.paths
            )
        
        else:
            # CBS failed - clean slate and run auction
            # Reset grid system for regular agents (keep emergency paths)
            emergency_paths = grid.get_emergency_paths()
            
            # Clear all regular agent paths
            for agent in regular_agents:
                grid.clear_agent_path(agent.id)
            
            # Create auction system
            auction_pathfinder = AStarPathfinder(grid, create_fast_config())
            auction_cbs = create_cbs_solver(grid, auction_pathfinder)
            auction = create_auction_system(grid, auction_cbs, auction_pathfinder)
            auction.max_rounds = self.max_auction_rounds
            
            # Use CBS conflict density for auction pricing
            conflict_density = cbs_result.conflict_density
            auction_result = auction.run_auction(regular_agents, conflict_density, emergency_paths)
            
            computation_time = time.time() - start_time
            
            # Extract results
            successful_agents = list(auction_result.final_winners.keys())
            failed_agents = auction_result.unassigned_agents
            
            total_cost = sum(len(path) for path in auction_result.final_winners.values())
            avg_length = total_cost / len(auction_result.final_winners) if auction_result.final_winners else 0.0
            
            # Extract winning bids
            winning_bids = {}
            for round_result in auction_result.round_results:
                for agent_id, bid in round_result.winning_bids.items():
                    winning_bids[agent_id] = bid.total_bid_amount
            
            return AlgorithmResult(
                algorithm_name="Hybrid (Auction)",
                success=auction_result.success,
                computation_time=computation_time,
                successful_agents=successful_agents,
                failed_agents=failed_agents,
                cbs_iterations=cbs_result.iterations_used,
                cbs_conflicts_found=len(cbs_result.conflicts_found),
                auction_revenue=auction_result.total_revenue,
                auction_rounds=auction_result.total_rounds,
                winning_bids=winning_bids,
                total_path_cost=total_cost,
                average_path_length=avg_length,
                regular_agent_paths=auction_result.final_winners
            )
    
    def run_scenario(self, scenario: TestScenario) -> TestResult:
        """Run all algorithms on a single scenario"""
        logger.info(f"Running scenario {scenario.scenario_id}")
        
        test_result = TestResult(scenario=scenario)
        
        # Set up base grid system
        grid = self.setup_grid_system(scenario)
        emergency_agents, regular_agents = self.create_agents(scenario)
        
        # Plan emergency agent paths (consistent across all algorithms)
        emergency_paths = self.plan_emergency_agents(grid, emergency_agents)
        
        # Get initial conflict density for auction pricing
        initial_pathfinder = AStarPathfinder(grid, create_conservative_config())
        initial_cbs = create_cbs_solver(grid, initial_pathfinder)
        initial_result = initial_cbs.solve(regular_agents[:min(8, len(regular_agents))])  # Sample for density
        conflict_density = initial_result.conflict_density
        
        # Test each algorithm with fresh grid state
        algorithms = ["CBS-only", "Auction-only", "Hybrid"]
        
        for algorithm in algorithms:
            try:
                # Reset grid system for this test
                fresh_grid = self.setup_grid_system(scenario)
                fresh_emergency, fresh_regular = self.create_agents(scenario)
                
                # Re-establish emergency paths
                emergency_paths = self.plan_emergency_agents(fresh_grid, fresh_emergency)
                
                # Run algorithm
                if algorithm == "CBS-only":
                    result = self.run_cbs_only(fresh_grid, fresh_regular)
                elif algorithm == "Auction-only":
                    result = self.run_auction_only(fresh_grid, fresh_regular, conflict_density)
                elif algorithm == "Hybrid":
                    result = self.run_hybrid(fresh_grid, fresh_regular)
                
                result.emergency_paths = emergency_paths
                test_result.results[algorithm] = result
                
                logger.info(f"  {algorithm}: {len(result.successful_agents)}/{scenario.num_regular_agents} "
                           f"agents successful in {result.computation_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error running {algorithm} on {scenario.scenario_id}: {e}")
                # Create failure result
                test_result.results[algorithm] = AlgorithmResult(
                    algorithm_name=algorithm,
                    success=False,
                    computation_time=0.0,
                    successful_agents=[],
                    failed_agents=[a.id for a in regular_agents]
                )
        
        return test_result

class FairnessAnalyzer:
    """Analyzes fairness metrics for test results"""
    
    @staticmethod
    def calculate_gini_coefficient(values: List[float]) -> float:
        """Calculate Gini coefficient for a list of values"""
        if not values or len(values) == 1:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    @staticmethod
    def calculate_budget_correlation(budgets: List[float], successes: List[bool]) -> float:
        """Calculate correlation between budget and success"""
        if len(budgets) != len(successes) or len(budgets) < 2:
            return 0.0
        
        success_values = [1.0 if s else 0.0 for s in successes]
        
        try:
            correlation = np.corrcoef(budgets, success_values)[0, 1]
            return 0.0 if np.isnan(correlation) else correlation
        except:
            return 0.0
    
    @staticmethod
    def analyze_scenario_fairness(scenario: TestScenario, result: AlgorithmResult) -> FairnessMetrics:
        """Calculate fairness metrics for a single algorithm result"""
        # Extract budget and success data
        budgets = [agent_data['budget'] for agent_data in scenario.regular_agents_data]
        agent_ids = list(range(len(budgets)))
        successes = [agent_id in result.successful_agents for agent_id in agent_ids]
        
        # Create budget-success pairs and sort by budget
        budget_success_pairs = list(zip(budgets, successes, agent_ids))
        budget_success_pairs.sort(key=lambda x: x[0])  # Sort by budget
        
        # Calculate success rate by quartile
        n = len(budget_success_pairs)
        quartile_size = n // 4
        quartile_success_rates = []
        
        for q in range(4):
            start_idx = q * quartile_size
            end_idx = (q + 1) * quartile_size if q < 3 else n  # Last quartile gets remainder
            
            quartile_successes = [pair[1] for pair in budget_success_pairs[start_idx:end_idx]]
            success_rate = sum(quartile_successes) / len(quartile_successes) if quartile_successes else 0.0
            quartile_success_rates.append(success_rate)
        
        # Calculate Gini coefficient (using budgets of successful agents)
        successful_budgets = [budgets[i] for i in range(len(budgets)) if successes[i]]
        gini = FairnessAnalyzer.calculate_gini_coefficient(successful_budgets) if successful_budgets else 0.0
        
        # Calculate budget correlation
        correlation = FairnessAnalyzer.calculate_budget_correlation(budgets, successes)
        
        # Calculate wealth concentration (success share of top 20%)
        top_20_percent_count = max(1, n // 5)
        top_20_percent_successes = sum(pair[1] for pair in budget_success_pairs[-top_20_percent_count:])
        total_successes = sum(successes)
        wealth_concentration = (top_20_percent_successes / total_successes) if total_successes > 0 else 0.0
        
        # Calculate budget utilization rate (for auction algorithms)
        budget_utilization = 0.0
        if result.winning_bids:
            total_spent = sum(result.winning_bids.values())
            total_budget = sum(budgets)
            budget_utilization = float(total_spent) / total_budget if total_budget > 0 else 0.0
        
        return FairnessMetrics(
            success_rate_by_quartile=quartile_success_rates,
            gini_coefficient=gini,
            budget_correlation_coefficient=correlation,
            wealth_concentration_index=wealth_concentration,
            budget_utilization_rate=budget_utilization
        )

class ExperimentRunner:
    """Orchestrates the complete experimental study"""
    
    def __init__(self, base_seed: int = 42):
        self.scenario_generator = TestScenarioGenerator(base_seed)
        self.test_runner = TestRunner()
        self.fairness_analyzer = FairnessAnalyzer()
        
        # Experimental parameters
        self.agent_counts = [4, 8, 12, 16, 20, 24]  # Range of agent counts to test
        self.scenarios_per_count = 10  # Number of scenarios per agent count
        
    def run_agent_count_experiment(self) -> List[TestResult]:
        """Run the main experiment across different agent counts"""
        all_results = []
        
        logger.info("Starting agent count experiment")
        logger.info(f"Testing agent counts: {self.agent_counts}")
        logger.info(f"Scenarios per count: {self.scenarios_per_count}")
        
        for agent_count in self.agent_counts:
            logger.info(f"\n=== Testing {agent_count} regular agents ===")
            
            # Generate scenarios for this agent count
            scenarios = self.scenario_generator.generate_scenario_batch(agent_count, self.scenarios_per_count)
            
            count_results = []
            for scenario in scenarios:
                # Run test scenario
                test_result = self.test_runner.run_scenario(scenario)
                
                # Calculate fairness metrics for each algorithm
                for alg_name, alg_result in test_result.results.items():
                    fairness = self.fairness_analyzer.analyze_scenario_fairness(scenario, alg_result)
                    test_result.fairness_metrics[alg_name] = fairness
                
                count_results.append(test_result)
                all_results.append(test_result)
            
            # Log summary for this agent count
            self._log_agent_count_summary(agent_count, count_results)
        
        logger.info("\n=== Experiment Complete ===")
        return all_results
    
    def _log_agent_count_summary(self, agent_count: int, results: List[TestResult]):
        """Log summary statistics for a specific agent count"""
        algorithms = ["CBS-only", "Auction-only", "Hybrid"]
        
        logger.info(f"\nSummary for {agent_count} agents ({len(results)} scenarios):")
        
        for alg in algorithms:
            # Collect metrics across all scenarios
            success_rates = []
            computation_times = []
            revenues = []
            gini_coeffs = []
            budget_correlations = []
            
            for result in results:
                if alg in result.results:
                    alg_result = result.results[alg]
                    success_rate = len(alg_result.successful_agents) / agent_count
                    success_rates.append(success_rate)
                    computation_times.append(alg_result.computation_time)
                    revenues.append(float(alg_result.auction_revenue))
                    
                    if alg in result.fairness_metrics:
                        fairness = result.fairness_metrics[alg]
                        gini_coeffs.append(fairness.gini_coefficient)
                        budget_correlations.append(fairness.budget_correlation_coefficient)
            
            if success_rates:
                logger.info(f"  {alg}:")
                logger.info(f"    Success rate: {np.mean(success_rates):.3f} ± {np.std(success_rates):.3f}")
                logger.info(f"    Computation time: {np.mean(computation_times):.3f}s ± {np.std(computation_times):.3f}s")
                if revenues and any(r > 0 for r in revenues):
                    logger.info(f"    Revenue: {np.mean(revenues):.2f} ± {np.std(revenues):.2f}")
                if gini_coeffs:
                    logger.info(f"    Gini coefficient: {np.mean(gini_coeffs):.3f} ± {np.std(gini_coeffs):.3f}")
                if budget_correlations:
                    logger.info(f"    Budget correlation: {np.mean(budget_correlations):.3f} ± {np.std(budget_correlations):.3f}")
    
    def save_results(self, results: List[TestResult], filename: str = "experiment_results.json"):
        """Save experiment results to JSON file"""
        # Convert results to serializable format
        serializable_results = []
        
        for test_result in results:
            result_data = {
                'scenario': {
                    'scenario_id': test_result.scenario.scenario_id,
                    'num_regular_agents': test_result.scenario.num_regular_agents,
                    'num_emergency_agents': test_result.scenario.num_emergency_agents,
                    'grid_size': [test_result.scenario.grid_width, test_result.scenario.grid_height],
                    'static_obstacles': test_result.scenario.static_obstacles,
                    'regular_agents_data': test_result.scenario.regular_agents_data
                },
                'algorithm_results': {},
                'fairness_metrics': {}
            }
            
            # Convert algorithm results
            for alg_name, alg_result in test_result.results.items():
                result_data['algorithm_results'][alg_name] = {
                    'success': alg_result.success,
                    'computation_time': alg_result.computation_time,
                    'successful_agents': alg_result.successful_agents,
                    'failed_agents': alg_result.failed_agents,
                    'cbs_iterations': alg_result.cbs_iterations,
                    'cbs_conflicts_found': alg_result.cbs_conflicts_found,
                    'auction_revenue': float(alg_result.auction_revenue),
                    'auction_rounds': alg_result.auction_rounds,
                    'winning_bids': {str(k): float(v) for k, v in alg_result.winning_bids.items()},
                    'total_path_cost': alg_result.total_path_cost,
                    'average_path_length': alg_result.average_path_length
                }
            
            # Convert fairness metrics
            for alg_name, fairness in test_result.fairness_metrics.items():
                result_data['fairness_metrics'][alg_name] = {
                    'success_rate_by_quartile': fairness.success_rate_by_quartile,
                    'gini_coefficient': fairness.gini_coefficient,
                    'budget_correlation_coefficient': fairness.budget_correlation_coefficient,
                    'wealth_concentration_index': fairness.wealth_concentration_index,
                    'budget_utilization_rate': fairness.budget_utilization_rate
                }
            
            serializable_results.append(result_data)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def generate_summary_report(self, results: List[TestResult]) -> Dict:
        """Generate comprehensive summary report"""
        algorithms = ["CBS-only", "Auction-only", "Hybrid"]
        agent_counts = sorted(set(r.scenario.num_regular_agents for r in results))
        
        report = {
            'experiment_overview': {
                'total_scenarios': len(results),
                'agent_counts_tested': agent_counts,
                'algorithms_compared': algorithms
            },
            'performance_by_agent_count': {},
            'overall_algorithm_comparison': {},
            'fairness_analysis': {},
            'key_findings': []
        }
        
        # Performance by agent count
        for agent_count in agent_counts:
            count_results = [r for r in results if r.scenario.num_regular_agents == agent_count]
            
            agent_count_data = {
                'scenarios': len(count_results),
                'algorithms': {}
            }
            
            for alg in algorithms:
                metrics = self._collect_algorithm_metrics(count_results, alg)
                if metrics['valid_results'] > 0:
                    agent_count_data['algorithms'][alg] = metrics
            
            report['performance_by_agent_count'][agent_count] = agent_count_data
        
        # Overall algorithm comparison
        for alg in algorithms:
            metrics = self._collect_algorithm_metrics(results, alg)
            if metrics['valid_results'] > 0:
                report['overall_algorithm_comparison'][alg] = metrics
        
        # Fairness analysis
        report['fairness_analysis'] = self._analyze_fairness_trends(results)
        
        # Key findings
        report['key_findings'] = self._generate_key_findings(results, report)
        
        return report
    
    def _collect_algorithm_metrics(self, results: List[TestResult], algorithm: str) -> Dict:
        """Collect metrics for a specific algorithm across results"""
        success_rates = []
        computation_times = []
        revenues = []
        gini_coeffs = []
        budget_correlations = []
        quartile_success_rates = [[], [], [], []]
        
        for result in results:
            if algorithm in result.results:
                alg_result = result.results[algorithm]
                agent_count = result.scenario.num_regular_agents
                
                success_rate = len(alg_result.successful_agents) / agent_count
                success_rates.append(success_rate)
                computation_times.append(alg_result.computation_time)
                revenues.append(float(alg_result.auction_revenue))
                
                if algorithm in result.fairness_metrics:
                    fairness = result.fairness_metrics[algorithm]
                    gini_coeffs.append(fairness.gini_coefficient)
                    budget_correlations.append(fairness.budget_correlation_coefficient)
                    
                    for i, rate in enumerate(fairness.success_rate_by_quartile):
                        quartile_success_rates[i].append(rate)
        
        if not success_rates:
            return {'valid_results': 0}
        
        metrics = {
            'valid_results': len(success_rates),
            'success_rate': {
                'mean': np.mean(success_rates),
                'std': np.std(success_rates),
                'min': np.min(success_rates),
                'max': np.max(success_rates)
            },
            'computation_time': {
                'mean': np.mean(computation_times),
                'std': np.std(computation_times),
                'min': np.min(computation_times),
                'max': np.max(computation_times)
            }
        }
        
        if any(r > 0 for r in revenues):
            non_zero_revenues = [r for r in revenues if r > 0]
            metrics['revenue'] = {
                'mean': np.mean(non_zero_revenues),
                'std': np.std(non_zero_revenues),
                'scenarios_with_revenue': len(non_zero_revenues)
            }
        
        if gini_coeffs:
            metrics['fairness'] = {
                'gini_coefficient': {
                    'mean': np.mean(gini_coeffs),
                    'std': np.std(gini_coeffs)
                },
                'budget_correlation': {
                    'mean': np.mean(budget_correlations),
                    'std': np.std(budget_correlations)
                },
                'quartile_success_rates': {
                    f'Q{i+1}': {
                        'mean': np.mean(rates) if rates else 0.0,
                        'std': np.std(rates) if rates else 0.0
                    }
                    for i, rates in enumerate(quartile_success_rates)
                }
            }
        
        return metrics
    
    def _analyze_fairness_trends(self, results: List[TestResult]) -> Dict:
        """Analyze fairness trends across algorithms and agent counts"""
        algorithms = ["CBS-only", "Auction-only", "Hybrid"]
        agent_counts = sorted(set(r.scenario.num_regular_agents for r in results))
        
        fairness_trends = {
            'gini_by_agent_count': {alg: {} for alg in algorithms},
            'correlation_by_agent_count': {alg: {} for alg in algorithms},
            'quartile_analysis': {alg: {} for alg in algorithms}
        }
        
        for agent_count in agent_counts:
            count_results = [r for r in results if r.scenario.num_regular_agents == agent_count]
            
            for alg in algorithms:
                gini_coeffs = []
                correlations = []
                quartile_rates = [[] for _ in range(4)]
                
                for result in count_results:
                    if alg in result.fairness_metrics:
                        fairness = result.fairness_metrics[alg]
                        gini_coeffs.append(fairness.gini_coefficient)
                        correlations.append(fairness.budget_correlation_coefficient)
                        
                        for i, rate in enumerate(fairness.success_rate_by_quartile):
                            quartile_rates[i].append(rate)
                
                if gini_coeffs:
                    fairness_trends['gini_by_agent_count'][alg][agent_count] = np.mean(gini_coeffs)
                    fairness_trends['correlation_by_agent_count'][alg][agent_count] = np.mean(correlations)
                    
                    # Calculate quartile inequality (Q4 - Q1)
                    q1_mean = np.mean(quartile_rates[0]) if quartile_rates[0] else 0.0
                    q4_mean = np.mean(quartile_rates[3]) if quartile_rates[3] else 0.0
                    fairness_trends['quartile_analysis'][alg][agent_count] = q4_mean - q1_mean
        
        return fairness_trends
    
    def _generate_key_findings(self, results: List[TestResult], report: Dict) -> List[str]:
        """Generate key findings from the experimental results"""
        findings = []
        algorithms = ["CBS-only", "Auction-only", "Hybrid"]
        
        # Find best performing algorithm overall
        best_success_rates = {}
        for alg in algorithms:
            if alg in report['overall_algorithm_comparison']:
                best_success_rates[alg] = report['overall_algorithm_comparison'][alg]['success_rate']['mean']
        
        if best_success_rates:
            best_alg = max(best_success_rates, key=best_success_rates.get)
            findings.append(f"{best_alg} achieved the highest overall success rate: {best_success_rates[best_alg]:.3f}")
        
        # Analyze computation time trends
        fastest_alg = None
        fastest_time = float('inf')
        for alg in algorithms:
            if alg in report['overall_algorithm_comparison']:
                avg_time = report['overall_algorithm_comparison'][alg]['computation_time']['mean']
                if avg_time < fastest_time:
                    fastest_time = avg_time
                    fastest_alg = alg
        
        if fastest_alg:
            findings.append(f"{fastest_alg} was the fastest algorithm with average time: {fastest_time:.3f}s")
        
        # Analyze fairness trends
        if 'fairness' in report['overall_algorithm_comparison'].get('CBS-only', {}):
            cbs_gini = report['overall_algorithm_comparison']['CBS-only']['fairness']['gini_coefficient']['mean']
            findings.append(f"CBS-only achieved Gini coefficient of {cbs_gini:.3f}")
        
        # Check for scaling behavior
        agent_counts = sorted(report['performance_by_agent_count'].keys())
        if len(agent_counts) >= 2:
            findings.append(f"Tested scalability across {len(agent_counts)} different agent counts: {agent_counts}")
        
        # Revenue analysis for auction algorithms
        auction_algs = ['Auction-only', 'Hybrid']
        for alg in auction_algs:
            if alg in report['overall_algorithm_comparison'] and 'revenue' in report['overall_algorithm_comparison'][alg]:
                avg_revenue = report['overall_algorithm_comparison'][alg]['revenue']['mean']
                findings.append(f"{alg} generated average revenue of {avg_revenue:.2f} per scenario")
        
        return findings

def main():
    """Main function to run the complete experimental study"""
    
    # Configure experiment
    experiment = ExperimentRunner(base_seed=42)
    
    # Customize experimental parameters if needed
    experiment.agent_counts = [4, 8, 12, 16, 20, 24]  # Agent counts to test
    experiment.scenarios_per_count = 20  # Number of scenarios per agent count
    
    print("=== UrbanAirspaceSim Integration Test ===")
    print(f"Testing agent counts: {experiment.agent_counts}")
    print(f"Scenarios per count: {experiment.scenarios_per_count}")
    print(f"Total scenarios: {len(experiment.agent_counts) * experiment.scenarios_per_count}")
    print()
    
    try:
        # Run the experiment
        all_results = experiment.run_agent_count_experiment()
        
        # Save raw results
        experiment.save_results(all_results, "experiment_results.json")
        
        # Generate and save summary report
        summary_report = experiment.generate_summary_report(all_results)
        
        with open("experiment_summary.json", 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Print key findings
        print("\n=== KEY FINDINGS ===")
        for finding in summary_report['key_findings']:
            print(f"• {finding}")
        
        # Print final summary
        total_scenarios = len(all_results)
        algorithms_tested = len(summary_report['overall_algorithm_comparison'])
        
        print(f"\n=== EXPERIMENT COMPLETE ===")
        print(f"Total scenarios tested: {total_scenarios}")
        print(f"Algorithms compared: {algorithms_tested}")
        print(f"Results saved to: experiment_results.json")
        print(f"Summary saved to: experiment_summary.json")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()