"""
Improved CBS Data Collection Module
Generates training data with extended temporal sequences and multiple feature types
"""

import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import pickle
import json
import os
from collections import defaultdict

# Import from existing modules
from grid_system import GridSystem, Agent, AgentType, Position
from astar_pathfinding import AStarPathfinder, create_conservative_config
from cbs_module import ConflictBasedSearch, create_cbs_solver, CBSResult, CBSNode
from test import TestScenarioGenerator, TestRunner

logger = logging.getLogger(__name__)

@dataclass
class ImprovedConflictMatrixData:
    """Enhanced data structure with extended temporal and multiple features"""
    scenario_id: str
    
    # Extended temporal features (50 iterations instead of 10)
    conflict_matrices: np.ndarray  # Shape: (grid_width, grid_height, 50)
    constraint_counts: np.ndarray  # Shape: (50,) - number of constraints per iteration
    open_list_sizes: np.ndarray    # Shape: (50,) - CBS high-level nodes in open list
    conflicts_resolved: np.ndarray # Shape: (50,) - conflicts resolved per iteration
    
    # Derived features for better learning
    conflict_density_evolution: np.ndarray    # Shape: (50,) - total conflicts per iteration
    constraint_growth_rate: np.ndarray        # Shape: (50,) - rate of constraint addition
    search_efficiency: np.ndarray             # Shape: (50,) - conflicts resolved per node expanded
    
    # Early success indicators (computed from first 20 iterations)
    early_conflict_trend: float      # Slope of conflict reduction in first 20 iterations  
    early_resolution_rate: float     # Average conflicts resolved per iteration (first 20)
    early_search_stability: float    # Variance in open list size (first 20)
    
    # Multiple prediction targets (more learnable than binary success)
    cbs_success: bool                 # Original binary target
    iteration_count: int              # Regression target: total iterations needed
    convergence_speed: float          # Regression target: conflicts resolved per iteration
    
    # Metadata
    total_iterations: int
    computation_time: float
    final_conflicts: int
    num_regular_agents: int
    num_emergency_agents: int
    grid_size: Tuple[int, int]

@dataclass
class ConsolidatedImprovedDataset:
    """Container for improved training dataset"""
    # Core 3D features (extended temporal dimension)
    conflict_matrices: np.ndarray     # Shape: (num_samples, 20, 20, 50)
    constraint_evolution: np.ndarray  # Shape: (num_samples, 50)
    open_list_evolution: np.ndarray   # Shape: (num_samples, 50) 
    
    # Derived 1D features for each sample
    early_indicators: np.ndarray      # Shape: (num_samples, 3) - trend, rate, stability
    
    # Multiple target options
    binary_success_labels: np.ndarray    # Shape: (num_samples,) - original binary
    iteration_count_labels: np.ndarray   # Shape: (num_samples,) - regression target
    convergence_speed_labels: np.ndarray # Shape: (num_samples,) - regression target
    
    # Metadata
    scenario_ids: List[str]
    num_samples: int
    success_rate: float
    collection_config: Dict
    
    def save_to_file(self, filepath: str):
        """Save improved dataset"""
        data_to_save = {
            'conflict_matrices': self.conflict_matrices,
            'constraint_evolution': self.constraint_evolution,
            'open_list_evolution': self.open_list_evolution,
            'early_indicators': self.early_indicators,
            'binary_success_labels': self.binary_success_labels,
            'iteration_count_labels': self.iteration_count_labels,
            'convergence_speed_labels': self.convergence_speed_labels,
            'scenario_ids': self.scenario_ids,
            'num_samples': self.num_samples,
            'success_rate': self.success_rate,
            'collection_config': self.collection_config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        logger.info(f"Improved dataset saved to {filepath}")
        logger.info(f"Dataset shape: {self.conflict_matrices.shape}")
        logger.info(f"Temporal length: {self.conflict_matrices.shape[3]} iterations")
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load improved dataset"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            conflict_matrices=data['conflict_matrices'],
            constraint_evolution=data['constraint_evolution'],
            open_list_evolution=data['open_list_evolution'],
            early_indicators=data['early_indicators'],
            binary_success_labels=data['binary_success_labels'],
            iteration_count_labels=data['iteration_count_labels'],
            convergence_speed_labels=data['convergence_speed_labels'],
            scenario_ids=data['scenario_ids'],
            num_samples=data['num_samples'],
            success_rate=data['success_rate'],
            collection_config=data['collection_config']
        )

class ImprovedConflictMatrixCollector:
    """Enhanced collector for extended temporal sequences and multiple features"""
    
    def __init__(self, grid_width: int, grid_height: int, max_iterations: int = 50):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_iterations = max_iterations
        
        # Extended feature storage
        self.conflict_matrices = np.zeros((grid_width, grid_height, max_iterations), dtype=np.float32)
        self.constraint_counts = np.zeros(max_iterations, dtype=np.int32)
        self.open_list_sizes = np.zeros(max_iterations, dtype=np.int32)
        self.conflicts_resolved = np.zeros(max_iterations, dtype=np.float32)
        
        # Tracking for derived features
        self.previous_conflicts = set()
        self.current_iteration = 0
        self.collection_active = True
    
    def reset(self):
        """Reset collector for new CBS run"""
        self.conflict_matrices.fill(0)
        self.constraint_counts.fill(0)
        self.open_list_sizes.fill(0)
        self.conflicts_resolved.fill(0)
        self.previous_conflicts.clear()
        self.current_iteration = 0
        self.collection_active = True
    
    def record_cbs_iteration(self, conflicts: List, cbs_node, open_list_size: int, iteration: int):
        """Record comprehensive CBS state for this iteration"""
        if not self.collection_active or iteration >= self.max_iterations:
            return
        
        # Record spatial conflicts (existing functionality)
        conflict_matrix = np.zeros((self.grid_width, self.grid_height), dtype=np.float32)
        current_conflicts = set()
        
        for conflict in conflicts:
            x, y = conflict.position.x, conflict.position.y
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                conflict_matrix[x, y] += 1.0
                current_conflicts.add((x, y))
        
        self.conflict_matrices[:, :, iteration] = conflict_matrix
        
        # Record additional features
        self.constraint_counts[iteration] = len(cbs_node.constraints) if cbs_node else 0
        self.open_list_sizes[iteration] = open_list_size
        
        # Calculate conflicts resolved this iteration
        if iteration > 0:
            resolved_count = len(self.previous_conflicts - current_conflicts)
            self.conflicts_resolved[iteration] = resolved_count
        
        self.previous_conflicts = current_conflicts.copy()
        self.current_iteration = iteration + 1
        
        # Stop collecting after max_iterations
        if self.current_iteration >= self.max_iterations:
            self.collection_active = False
    
    def calculate_derived_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate derived features from raw data"""
        
        # Conflict density evolution (total conflicts per iteration)
        conflict_density = np.sum(self.conflict_matrices, axis=(0, 1))
        
        # Constraint growth rate (change in constraints per iteration)
        constraint_growth = np.diff(self.constraint_counts, prepend=0)
        
        # Search efficiency (conflicts resolved per constraint added)
        search_efficiency = np.zeros(self.max_iterations)
        for i in range(1, self.max_iterations):
            if constraint_growth[i] > 0:
                search_efficiency[i] = self.conflicts_resolved[i] / constraint_growth[i]
        
        return conflict_density, constraint_growth, search_efficiency
    
    def calculate_early_indicators(self) -> Tuple[float, float, float]:
        """Calculate early success indicators from first 20 iterations"""
        early_window = 20
        
        # Early conflict trend (slope of conflict reduction)
        early_conflicts = np.sum(self.conflict_matrices[:, :, :early_window], axis=(0, 1))
        if len(early_conflicts) >= 3:
            x = np.arange(len(early_conflicts))
            trend_slope = np.polyfit(x, early_conflicts, 1)[0]
            early_conflict_trend = -trend_slope  # Negative slope is good
        else:
            early_conflict_trend = 0.0
        
        # Early resolution rate
        early_resolution_rate = np.mean(self.conflicts_resolved[:early_window])
        
        # Early search stability (lower variance in open list size is better)
        early_open_list = self.open_list_sizes[:early_window]
        early_search_stability = 1.0 / (1.0 + np.var(early_open_list))  # Inverse variance
        
        return early_conflict_trend, early_resolution_rate, early_search_stability

class ImprovedModifiedCBS(ConflictBasedSearch):
    """Enhanced CBS that collects extended features during execution"""
    
    def __init__(self, grid_system: GridSystem, pathfinder: AStarPathfinder, 
                 matrix_collector: ImprovedConflictMatrixCollector):
        super().__init__(grid_system, pathfinder)
        self.matrix_collector = matrix_collector
    
    def solve_with_improved_collection(self, agents: List[Agent], start_time: int = 0) -> CBSResult:
        """Modified solve method with enhanced data collection"""
        
        # Reset collector for new run
        self.matrix_collector.reset()
        
        start_solve_time = time.time()
        self.stats['total_runs'] += 1
        
        result = CBSResult()
        result.pathfinding_calls = 0
        
        if not agents:
            result.success = True
            result.computation_time = time.time() - start_solve_time
            return result
        
        # Initialize root node
        root_node = self._create_root_node(agents, start_time, result)
        if not root_node:
            result.failure_reason = "Could not create initial solution"
            result.computation_time = time.time() - start_solve_time
            return result
        
        # Record initial state (iteration 0)
        self.matrix_collector.record_cbs_iteration(root_node.conflicts, root_node, 1, 0)
        
        # Check if already solved
        if root_node.is_solution():
            result.success = True
            result.paths = root_node.paths
            result.total_cost = root_node.solution_cost
            result.iterations_used = 1
            result.computation_time = time.time() - start_solve_time
            return result
        
        # CBS main search loop
        import heapq
        open_list = [root_node]
        heapq.heapify(open_list)
        
        iterations = 0
        best_node = root_node
        
        while (open_list and 
               iterations < self.max_iterations and 
               time.time() - start_solve_time < self.max_computation_time):
            
            iterations += 1
            current_node = heapq.heappop(open_list)
            
            # Record enhanced state information
            if iterations < self.matrix_collector.max_iterations:
                self.matrix_collector.record_cbs_iteration(
                    current_node.conflicts, current_node, len(open_list), iterations
                )
            
            # Track best node
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
                result.computation_time = time.time() - start_solve_time
                return result
            
            # Continue CBS search (existing logic)
            if not current_node.conflicts:
                continue
            
            conflict = self._select_conflict_to_resolve(current_node.conflicts)
            child_nodes = self._generate_child_nodes(current_node, conflict, agents, start_time, result)
            
            for child in child_nodes:
                if child and len(child.conflicts) <= self.max_conflicts_per_node:
                    heapq.heappush(open_list, child)
        
        # Search failed
        result.iterations_used = iterations
        result.computation_time = time.time() - start_solve_time
        result.final_conflicts = best_node.conflicts
        
        if iterations >= self.max_iterations:
            result.failure_reason = f"Maximum iterations ({self.max_iterations}) exceeded"
        else:
            result.failure_reason = "Time limit exceeded"
        
        return result

class ImprovedCBSDataCollector:
    """Main collector with enhanced data generation"""
    
    def __init__(self, base_seed: int = 42, cbs_iteration_limit: int = 600):
        self.scenario_generator = TestScenarioGenerator(base_seed)
        self.cbs_iteration_limit = cbs_iteration_limit
        
        # Configuration
        self.num_regular_agents = 12
        self.num_emergency_agents = 2
        self.grid_width = 20
        self.grid_height = 20
        
        self.collected_data: List[ImprovedConflictMatrixData] = []
    
    def setup_grid_system(self) -> GridSystem:
        """Set up grid system"""
        return GridSystem(self.grid_width, self.grid_height)
    
    def create_agents_from_scenario(self, scenario) -> Tuple[List[Agent], List[Agent]]:
        """Create agents from scenario data"""
        emergency_agents = []
        regular_agents = []
        
        # Create emergency agents
        for i, data in enumerate(scenario.emergency_agents_data):
            agent = Agent(
                id=f"emergency_{i}",
                agent_type=AgentType.EMERGENCY,
                start=data['start'],
                goal=data['goal'],
                priority=100
            )
            emergency_agents.append(agent)
        
        # Create regular agents
        for i, data in enumerate(scenario.regular_agents_data):
            agent = Agent(
                id=i,
                agent_type=AgentType.NON_EMERGENCY,
                start=data['start'],
                goal=data['goal'],
                budget=data['budget'],
                priority=1
            )
            regular_agents.append(agent)
        
        return emergency_agents, regular_agents
    
    def plan_emergency_agents(self, grid: GridSystem, emergency_agents: List[Agent]) -> Dict[int, List[Position]]:
        """Plan emergency agent paths"""
        if not emergency_agents:
            return {}
        
        for agent in emergency_agents:
            grid.add_agent(agent)
        
        pathfinder = AStarPathfinder(grid, create_conservative_config())
        cbs = create_cbs_solver(grid, pathfinder)
        
        result = cbs.solve(emergency_agents)
        
        emergency_paths = {}
        if result.success:
            emergency_paths = result.paths
            for agent_id, path in emergency_paths.items():
                grid.set_agent_path(agent_id, path)
        else:
            logger.warning("Failed to plan emergency agent paths")
        
        return emergency_paths
    
    def collect_single_improved_scenario(self, scenario_id: str, seed_offset: int = 0) -> Optional[ImprovedConflictMatrixData]:
        """Collect improved training data for a single scenario"""
        
        # Generate scenario
        scenario = self.scenario_generator.generate_scenario(self.num_regular_agents, seed_offset)
        scenario.scenario_id = scenario_id
        
        # Set up grid and agents
        grid = self.setup_grid_system()
        for x, y in scenario.static_obstacles:
            grid.add_static_obstacle(x, y)
        
        emergency_agents, regular_agents = self.create_agents_from_scenario(scenario)
        emergency_paths = self.plan_emergency_agents(grid, emergency_agents)
        
        for agent in regular_agents:
            grid.add_agent(agent)
        
        # Set up improved collector with extended temporal window
        matrix_collector = ImprovedConflictMatrixCollector(
            self.grid_width, self.grid_height, max_iterations=50  # Extended to 50 iterations
        )
        
        # Create improved CBS
        pathfinder = AStarPathfinder(grid, create_conservative_config())
        improved_cbs = ImprovedModifiedCBS(grid, pathfinder, matrix_collector)
        improved_cbs.max_iterations = self.cbs_iteration_limit
        
        # Run CBS with improved collection
        start_time = time.time()
        cbs_result = improved_cbs.solve_with_improved_collection(regular_agents)
        computation_time = time.time() - start_time
        
        # IMPROVED FILTERING: Require at least 50 iterations (up from 8)
        if cbs_result.iterations_used < 50:
            logger.info(f"Scenario {scenario_id}: CBS finished in {cbs_result.iterations_used} iterations, skipping (need ≥50)")
            return None
        
        # Calculate derived features and early indicators
        conflict_density, constraint_growth, search_efficiency = matrix_collector.calculate_derived_features()
        early_trend, early_rate, early_stability = matrix_collector.calculate_early_indicators()
        
        # Create improved training data
        improved_data = ImprovedConflictMatrixData(
            scenario_id=scenario_id,
            conflict_matrices=matrix_collector.conflict_matrices,
            constraint_counts=matrix_collector.constraint_counts,
            open_list_sizes=matrix_collector.open_list_sizes,
            conflicts_resolved=matrix_collector.conflicts_resolved,
            conflict_density_evolution=conflict_density,
            constraint_growth_rate=constraint_growth,
            search_efficiency=search_efficiency,
            early_conflict_trend=early_trend,
            early_resolution_rate=early_rate,
            early_search_stability=early_stability,
            cbs_success=cbs_result.success,
            iteration_count=cbs_result.iterations_used,
            convergence_speed=np.sum(matrix_collector.conflicts_resolved) / max(cbs_result.iterations_used, 1),
            total_iterations=cbs_result.iterations_used,
            computation_time=computation_time,
            final_conflicts=len(cbs_result.final_conflicts),
            num_regular_agents=self.num_regular_agents,
            num_emergency_agents=self.num_emergency_agents,
            grid_size=(self.grid_width, self.grid_height)
        )
        
        logger.info(f"Scenario {scenario_id}: CBS {'succeeded' if cbs_result.success else 'failed'} "
                   f"after {cbs_result.iterations_used} iterations")
        
        return improved_data
    
    def create_consolidated_improved_dataset(self, data_list: List[ImprovedConflictMatrixData]) -> ConsolidatedImprovedDataset:
        """Create consolidated improved training dataset"""
        
        num_samples = len(data_list)
        
        # Initialize arrays with improved dimensions
        conflict_matrices = np.zeros((num_samples, self.grid_width, self.grid_height, 50), dtype=np.float32)
        constraint_evolution = np.zeros((num_samples, 50), dtype=np.float32)
        open_list_evolution = np.zeros((num_samples, 50), dtype=np.float32)
        early_indicators = np.zeros((num_samples, 3), dtype=np.float32)
        
        # Multiple target labels
        binary_success_labels = np.zeros(num_samples, dtype=np.int32)
        iteration_count_labels = np.zeros(num_samples, dtype=np.float32)
        convergence_speed_labels = np.zeros(num_samples, dtype=np.float32)
        
        scenario_ids = []
        
        # Fill arrays
        for i, data in enumerate(data_list):
            conflict_matrices[i] = data.conflict_matrices
            constraint_evolution[i] = data.constraint_counts
            open_list_evolution[i] = data.open_list_sizes
            early_indicators[i] = [data.early_conflict_trend, data.early_resolution_rate, data.early_search_stability]
            
            binary_success_labels[i] = 1 if data.cbs_success else 0
            iteration_count_labels[i] = np.log(data.iteration_count + 1)  # Log transform for regression
            convergence_speed_labels[i] = data.convergence_speed
            
            scenario_ids.append(data.scenario_id)
        
        success_rate = np.mean(binary_success_labels)
        
        collection_config = {
            'num_regular_agents': self.num_regular_agents,
            'num_emergency_agents': self.num_emergency_agents,
            'grid_size': [self.grid_width, self.grid_height],
            'cbs_iteration_limit': self.cbs_iteration_limit,
            'min_iterations_required': 50,  # Increased from 8
            'max_iterations_collected': 50,  # Increased from 10
            'features': ['conflict_matrices', 'constraint_evolution', 'open_list_evolution', 'early_indicators']
        }
        
        return ConsolidatedImprovedDataset(
            conflict_matrices=conflict_matrices,
            constraint_evolution=constraint_evolution,
            open_list_evolution=open_list_evolution,
            early_indicators=early_indicators,
            binary_success_labels=binary_success_labels,
            iteration_count_labels=iteration_count_labels,
            convergence_speed_labels=convergence_speed_labels,
            scenario_ids=scenario_ids,
            num_samples=num_samples,
            success_rate=success_rate,
            collection_config=collection_config
        )
    
    def collect_improved_training_data(self, num_scenarios: int, output_dir: str = "improved_training_data") -> ConsolidatedImprovedDataset:
        """Generate improved training dataset"""
        
        logger.info(f"Starting IMPROVED data collection for {num_scenarios} scenarios")
        logger.info(f"KEY IMPROVEMENTS:")
        logger.info(f"- Extended temporal length: 50 iterations (was 10)")
        logger.info(f"- Multiple features: conflicts, constraints, open list sizes")
        logger.info(f"- Early indicators: trend, resolution rate, search stability")
        logger.info(f"- Multiple prediction targets: binary success, iteration count, convergence speed")
        logger.info(f"- Higher quality filter: ≥50 iterations required (was ≥8)")
        
        os.makedirs(output_dir, exist_ok=True)
        
        collected_data = []
        scenarios_collected = 0
        scenarios_attempted = 0
        
        while scenarios_collected < num_scenarios:
            scenario_id = f"improved_scenario_{scenarios_attempted}"
            scenarios_attempted += 1
            
            try:
                improved_data = self.collect_single_improved_scenario(scenario_id, scenarios_attempted)
                
                if improved_data is not None:
                    collected_data.append(improved_data)
                    scenarios_collected += 1
                    
                    if scenarios_collected % 10 == 0:
                        logger.info(f"Progress: {scenarios_collected}/{num_scenarios} scenarios collected")
                
            except Exception as e:
                logger.error(f"Error collecting scenario {scenario_id}: {e}")
                continue
            
            # Safety check
            if scenarios_attempted > num_scenarios * 5:
                logger.warning(f"Attempted {scenarios_attempted} scenarios, collected {scenarios_collected}")
                logger.warning("Many scenarios may be finishing too quickly (<50 iterations)")
                break
        
        if len(collected_data) == 0:
            raise ValueError("No valid scenarios collected! Consider reducing iteration requirement.")
        
        # Create consolidated dataset
        consolidated_dataset = self.create_consolidated_improved_dataset(collected_data)
        
        # Save improved dataset
        dataset_filepath = os.path.join(output_dir, "consolidated_improved_training_dataset.pkl")
        consolidated_dataset.save_to_file(dataset_filepath)
        
        # Save summary
        summary = {
            'total_scenarios': len(collected_data),
            'scenarios_attempted': scenarios_attempted,
            'success_count': int(np.sum(consolidated_dataset.binary_success_labels)),
            'success_rate': float(consolidated_dataset.success_rate),
            'temporal_length': 50,
            'features': ['conflict_matrices', 'constraint_evolution', 'open_list_evolution', 'early_indicators'],
            'prediction_targets': ['binary_success', 'log_iteration_count', 'convergence_speed'],
            'dataset_shape': list(consolidated_dataset.conflict_matrices.shape),
            'configuration': consolidated_dataset.collection_config
        }
        
        with open(os.path.join(output_dir, "improved_collection_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"IMPROVED data collection complete!")
        logger.info(f"Valid scenarios: {len(collected_data)} (attempted: {scenarios_attempted})")
        logger.info(f"Success rate: {summary['success_count']}/{len(collected_data)} ({summary['success_rate']*100:.1f}%)")
        logger.info(f"Dataset shape: {consolidated_dataset.conflict_matrices.shape}")
        logger.info(f"Temporal length: 50 iterations (5x improvement)")
        logger.info(f"Multiple prediction targets available")
        logger.info(f"Consolidated dataset: {dataset_filepath}")
        
        return consolidated_dataset

def main():
    """Generate improved training data"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create improved data collector
    collector = ImprovedCBSDataCollector(base_seed=42, cbs_iteration_limit=512)
    
    # Generate improved training data
    num_scenarios = 500  # May need more attempts due to 50-iteration filter
    improved_dataset = collector.collect_improved_training_data(
        num_scenarios, 
        output_dir="improved_cbs_training_data"
    )
    
    print(f"\n=== IMPROVED DATASET SUMMARY ===")
    print(f"Shape: {improved_dataset.conflict_matrices.shape}")
    print(f"Temporal length: 50 iterations (vs previous 10)")
    print(f"Features: Conflicts + Constraints + Open list + Early indicators")
    print(f"Targets: Binary success + Iteration count + Convergence speed")
    print(f"Success rate: {improved_dataset.success_rate:.3f}")
    print(f"Ready for training with multiple prediction targets!")

if __name__ == "__main__":
    main()