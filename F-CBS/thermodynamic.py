import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import random
import heapq
from fcbs_with_annealing import Grid, FCBS, FCBSNode, Agent, CBS

# Import your existing classes (assuming they're in the same file or imported)
# from your_cbs_module import *

class ThermodynamicTracker:
    """Track thermodynamic properties of CBS system during solving"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.iteration_data = []
        
    def calculate_rates(self, history_data, property_name, window_size=None):
        """Calculate rate of change for any property"""
        if window_size is None:
            window_size = min(self.window_size, len(history_data))
            
        if len(history_data) < 2:
            return 0.0
            
        recent_data = history_data[-window_size:] if len(history_data) >= window_size else history_data
        
        if len(recent_data) < 2:
            return 0.0
            
        values = [getattr(item, property_name) for item in recent_data]
        
        # Calculate rate using linear regression
        x = np.arange(len(values))
        if len(values) >= 2:
            try:
                slope, _ = np.polyfit(x, values, 1)
                return slope
            except:
                return 0.0
        return 0.0
    
    def calculate_variance(self, history_data, property_name, window_size=None):
        """Calculate variance of property over window"""
        if window_size is None:
            window_size = min(self.window_size, len(history_data))
            
        if len(history_data) < window_size:
            return float('inf')
            
        recent_data = history_data[-window_size:]
        values = [getattr(item, property_name) for item in recent_data]
        return np.var(values) if len(values) > 1 else 0.0
    
    def calculate_autocorrelation(self, history_data, property_name, lag=1, window_size=None):
        """Calculate autocorrelation of property"""
        if window_size is None:
            window_size = min(self.window_size, len(history_data))
            
        if len(history_data) < window_size or window_size <= lag:
            return 0.0
            
        recent_data = history_data[-window_size:]
        values = np.array([getattr(item, property_name) for item in recent_data])
        
        if len(values) <= lag:
            return 0.0
            
        # Calculate Pearson correlation between series and lagged series
        series1 = values[:-lag]
        series2 = values[lag:]
        
        if len(series1) < 2 or np.std(series1) == 0 or np.std(series2) == 0:
            return 0.0
            
        correlation = np.corrcoef(series1, series2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_exploration_rate(self, history_data, window_size=None):
        """Calculate how much new search space is being explored"""
        if window_size is None:
            window_size = min(self.window_size, len(history_data))
            
        if len(history_data) < window_size:
            return 1.0
            
        recent_nodes = history_data[-window_size:]
        unique_solutions = set()
        
        for node in recent_nodes:
            # Create hashable representation of solution
            try:
                solution_hash = tuple(sorted([
                    (agent_id, tuple(path)) 
                    for agent_id, path in node.solution.items()
                ]))
                unique_solutions.add(solution_hash)
            except:
                pass  # Skip if solution is malformed
                
        return len(unique_solutions) / len(recent_nodes) if recent_nodes else 0.0
    
    def calculate_conflict_persistence(self, conflict_history, window_size=None):
        """Measure how persistent conflicts are in same locations"""
        if window_size is None:
            window_size = min(self.window_size, len(conflict_history))
            
        if len(conflict_history) < window_size:
            return 0.0
            
        recent_conflicts = conflict_history[-window_size:]
        location_counts = defaultdict(int)
        total_conflicts = 0
        
        for conflicts in recent_conflicts:
            for conflict in conflicts:
                total_conflicts += 1
                if conflict['type'] == 'vertex':
                    location_counts[conflict['location']] += 1
                else:  # edge conflict
                    for loc in conflict['to_locations']:
                        location_counts[loc] += 0.5
        
        if total_conflicts == 0:
            return 0.0
            
        # Calculate entropy of conflict distribution
        conflict_entropy = 0.0
        for count in location_counts.values():
            if count > 0:
                p = count / total_conflicts
                conflict_entropy -= p * math.log2(p)
                
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(location_counts)) if location_counts else 1
        return 1.0 - (conflict_entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def record_iteration(self, iteration, node, conflict_history, node_history):
        """Record all metrics for current iteration"""
        
        # Calculate rates
        entropy_production_rate = self.calculate_rates(node_history, 'entropy')
        energy_dissipation_rate = -self.calculate_rates(node_history, 'free_energy')  # Negative because we want dissipation
        cost_change_rate = self.calculate_rates(node_history, 'cost')
        
        # Calculate variances
        entropy_variance = self.calculate_variance(node_history, 'entropy')
        energy_variance = self.calculate_variance(node_history, 'free_energy')
        cost_variance = self.calculate_variance(node_history, 'cost')
        
        # Calculate autocorrelations
        entropy_autocorr = self.calculate_autocorrelation(node_history, 'entropy', lag=1)
        energy_autocorr = self.calculate_autocorrelation(node_history, 'free_energy', lag=1)
        cost_autocorr = self.calculate_autocorrelation(node_history, 'cost', lag=1)
        
        # Calculate exploration and persistence
        exploration_rate = self.calculate_exploration_rate(node_history)
        conflict_persistence = self.calculate_conflict_persistence(conflict_history)
        
        # Calculate additional thermodynamic indicators
        num_conflicts = len(node.conflicts)
        conflict_density = num_conflicts / (12 * 12) if num_conflicts > 0 else 0.0  # Normalized by grid size
        
        # Phase space velocity (how fast the system state is changing)
        phase_space_velocity = 0.0
        if len(node_history) >= 2:
            prev_node = node_history[-2]
            # Measure change in system state
            cost_change = abs(node.cost - prev_node.cost)
            entropy_change = abs(node.entropy - prev_node.entropy)
            phase_space_velocity = math.sqrt(cost_change**2 + entropy_change**2)
        
        # Thermodynamic efficiency (how much progress per energy spent)
        thermodynamic_efficiency = 0.0
        if node.free_energy > 0:
            # Progress = reduction in conflicts
            if len(node_history) >= 2:
                prev_conflicts = len(node_history[-2].conflicts) if len(node_history) >= 2 else num_conflicts
                conflict_reduction = max(0, prev_conflicts - num_conflicts)
                thermodynamic_efficiency = conflict_reduction / node.free_energy
        
        data_point = {
            'iteration': iteration,
            'cost': node.cost,
            'entropy': node.entropy,
            'free_energy': node.free_energy,
            'num_conflicts': num_conflicts,
            'conflict_density': conflict_density,
            
            # Rates (production/dissipation)
            'entropy_production_rate': entropy_production_rate,
            'energy_dissipation_rate': energy_dissipation_rate,
            'cost_change_rate': cost_change_rate,
            
            # Variances (measure of fluctuations)
            'entropy_variance': entropy_variance,
            'energy_variance': energy_variance,
            'cost_variance': cost_variance,
            
            # Autocorrelations (measure of memory)
            'entropy_autocorr': entropy_autocorr,
            'energy_autocorr': energy_autocorr,
            'cost_autocorr': cost_autocorr,
            
            # System dynamics
            'exploration_rate': exploration_rate,
            'conflict_persistence': conflict_persistence,
            'phase_space_velocity': phase_space_velocity,
            'thermodynamic_efficiency': thermodynamic_efficiency,
            
            # Stability indicators
            'entropy_to_cost_ratio': node.entropy / node.cost if node.cost > 0 else 0,
            'energy_gradient': energy_dissipation_rate,
            'system_temperature': node.free_energy - node.cost if node.entropy > 0 else 0,  # Implicit temperature
        }
        
        self.iteration_data.append(data_point)
    
    def get_dataframe(self):
        """Return recorded data as pandas DataFrame"""
        return pd.DataFrame(self.iteration_data)


class ThermodynamicFCBS(FCBS):
    """Enhanced F-CBS that tracks thermodynamic properties"""
    
    def __init__(self, grid, agents, temperature=1.0, using_annealing=False, annealing_iterations=5):
        super().__init__(grid, agents, temperature, using_annealing, annealing_iterations)
        self.tracker = ThermodynamicTracker()
        self.node_history = []
        self.conflict_history = []
    
    def solve(self, max_iterations=1024):
        """Enhanced solve method with thermodynamic tracking"""
        start_time = time.time()
        root = FCBSNode()
        
        # Find initial solution
        for agent in self.agents:
            path = self.pathfinder.a_star(
                agent.start, agent.goal,
                root.vertex_constraints, root.edge_constraints
            )
            if path is None:
                return None, 0, time.time() - start_time, pd.DataFrame()
            root.solution[agent.id] = path
            root.cost += len(path) - 1
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        root.entropy = self.calculate_entropy(root.conflicts)
        root.free_energy = root.cost + self.current_temperature * root.entropy
        
        # Initialize tracking
        self.node_history.append(root)
        self.conflict_history.append(root.conflicts)
        self.tracker.record_iteration(0, root, self.conflict_history, self.node_history)
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            self.update_temperature(iterations)
            
            current = heapq.heappop(open_list)
            
            if not current.conflicts:
                # Solution found - record final state
                self.tracker.record_iteration(iterations, current, self.conflict_history, self.node_history)
                return current.solution, iterations, time.time() - start_time, self.tracker.get_dataframe()
            
            conflict = current.conflicts[0]
            
            for agent_id in conflict['agents']:
                new_node = FCBSNode()
                new_node.vertex_constraints = current.vertex_constraints.copy()
                new_node.edge_constraints = current.edge_constraints.copy()
                
                new_constraints = self.generate_constraints_for_conflict(conflict, agent_id)
                new_node.vertex_constraints.extend(new_constraints['vertex'])
                new_node.edge_constraints.extend(new_constraints['edge'])
                
                agent = next(a for a in self.agents if a.id == agent_id)
                new_path = self.pathfinder.a_star(
                    agent.start, agent.goal,
                    new_node.vertex_constraints, new_node.edge_constraints
                )
                
                if new_path is not None:
                    new_node.solution = current.solution.copy()
                    new_node.solution[agent_id] = new_path
                    new_node.cost = sum(len(path) - 1 for path in new_node.solution.values())
                    new_node.conflicts = self.conflict_detector.find_conflicts(new_node.solution)
                    new_node.entropy = self.calculate_entropy(new_node.conflicts)
                    new_node.free_energy = new_node.cost + self.current_temperature * new_node.entropy
                    
                    # Track this node
                    self.node_history.append(new_node)
                    self.conflict_history.append(new_node.conflicts)
                    self.tracker.record_iteration(iterations, new_node, self.conflict_history, self.node_history)
                    
                    heapq.heappush(open_list, new_node)
        
        # No solution found
        return None, iterations, time.time() - start_time, self.tracker.get_dataframe()


def run_thermodynamic_test(seed=42, max_iterations=2000):
    """Run thermodynamic analysis test on a specific scenario"""
    
    print(f"Running thermodynamic analysis test with seed {seed}")
    print("=" * 60)
    
    # Set up scenario
    random.seed(seed)
    np.random.seed(seed)
    
    width, height = 12, 12
    num_agents = 20
    num_obstacles = 12
    
    # Generate obstacles
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        obstacles.add((x, y))
    
    grid = Grid(width, height, obstacles)
    
    # Generate agents
    agents = []
    occupied = obstacles.copy()
    
    for i in range(num_agents):
        # Find valid start position
        while True:
            start = (random.randint(0, width-1), random.randint(0, height-1))
            if start not in occupied:
                occupied.add(start)
                break
        
        # Find valid goal position
        while True:
            goal = (random.randint(0, width-1), random.randint(0, height-1))
            if goal not in occupied and goal != start:
                break
        
        agents.append(Agent(i, start, goal))
    
    print(f"Generated scenario:")
    print(f"  Grid: {width}x{height}")
    print(f"  Agents: {num_agents}")
    print(f"  Obstacles: {num_obstacles}")
    print(f"  Obstacle positions: {sorted(obstacles)}")
    print(f"  Agent start->goal:")
    for agent in agents:
        print(f"    Agent {agent.id}: {agent.start} -> {agent.goal}")
    
    # Test different algorithms
    algorithms = [
        ("CBS", CBS(grid, agents)),
        ("F-CBS (T=1.0)", ThermodynamicFCBS(grid, agents, temperature=1.0)),
        ("F-CBS (T=5.0)", ThermodynamicFCBS(grid, agents, temperature=5.0)),
        ("F-CBS Anneal (T=10.0→0)", ThermodynamicFCBS(grid, agents, temperature=10.0, using_annealing=True, annealing_iterations=50)),
    ]
    
    results = []
    
    for alg_name, solver in algorithms:
        print(f"\nTesting {alg_name}...")
        start_time = time.time()
        
        if isinstance(solver, ThermodynamicFCBS):
            solution, iterations, solve_time, tracking_data = solver.solve(max_iterations)
        else:
            solution, iterations, solve_time = solver.solve(max_iterations)
            tracking_data = pd.DataFrame()  # Empty for regular CBS
        
        # Calculate solution cost
        cost = sum(len(path) - 1 for path in solution.values()) if solution else float('inf')
        
        result = {
            'algorithm': alg_name,
            'seed': seed,
            'solved': solution is not None,
            'iterations': iterations,
            'cost': cost,
            'solve_time': solve_time,
            'total_time': time.time() - start_time
        }
        
        results.append(result)
        
        print(f"  Result: {'SOLVED' if solution else 'FAILED'}")
        print(f"  Iterations: {iterations}")
        print(f"  Cost: {cost}")
        print(f"  Time: {solve_time:.3f}s")
        
        # Save detailed tracking data if available
        if not tracking_data.empty:
            filename = f"thermodynamic_trace_{alg_name.replace(' ', '_').replace('(', '').replace(')', '').replace('->', '_to_')}_seed_{seed}.csv"
            tracking_data['algorithm'] = alg_name
            tracking_data['seed'] = seed
            tracking_data.to_csv(filename, index=False)
            print(f"  Tracking data saved to: {filename}")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    summary_filename = f"thermodynamic_test_summary_seed_{seed}.csv"
    results_df.to_csv(summary_filename, index=False)
    print(f"\nSummary results saved to: {summary_filename}")
    
    return results_df


def analyze_thermodynamic_results(tracking_file):
    """Analyze thermodynamic properties from tracking data"""
    
    df = pd.read_csv(tracking_file)
    
    print(f"Analyzing thermodynamic data from {tracking_file}")
    print("=" * 60)
    
    # Basic statistics
    print("Final State:")
    print(f"  Final Cost: {df['cost'].iloc[-1]}")
    print(f"  Final Entropy: {df['entropy'].iloc[-1]:.3f}")
    print(f"  Final Free Energy: {df['free_energy'].iloc[-1]:.3f}")
    print(f"  Final Conflicts: {df['num_conflicts'].iloc[-1]}")
    
    print("\nThermodynamic Rates:")
    print(f"  Average Entropy Production Rate: {df['entropy_production_rate'].mean():.6f}")
    print(f"  Average Energy Dissipation Rate: {df['energy_dissipation_rate'].mean():.6f}")
    print(f"  Final Entropy Production Rate: {df['entropy_production_rate'].iloc[-50:].mean():.6f}")
    print(f"  Final Energy Dissipation Rate: {df['energy_dissipation_rate'].iloc[-50:].mean():.6f}")
    
    print("\nSystem Dynamics:")
    print(f"  Average Exploration Rate: {df['exploration_rate'].mean():.3f}")
    print(f"  Average Conflict Persistence: {df['conflict_persistence'].mean():.3f}")
    print(f"  Average Phase Space Velocity: {df['phase_space_velocity'].mean():.3f}")
    print(f"  Average Thermodynamic Efficiency: {df['thermodynamic_efficiency'].mean():.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot 1: Energy and Entropy over time
    axes[0, 0].plot(df['iteration'], df['cost'], label='Cost', alpha=0.7)
    axes[0, 0].plot(df['iteration'], df['free_energy'], label='Free Energy', alpha=0.7)
    axes[0, 0].set_title('Energy Evolution')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Entropy over time
    axes[0, 1].plot(df['iteration'], df['entropy'], color='red', alpha=0.7)
    axes[0, 1].set_title('Entropy Evolution')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Entropy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Production and Dissipation Rates
    axes[1, 0].plot(df['iteration'], df['entropy_production_rate'], label='Entropy Production Rate', alpha=0.7)
    axes[1, 0].plot(df['iteration'], df['energy_dissipation_rate'], label='Energy Dissipation Rate', alpha=0.7)
    axes[1, 0].set_title('Thermodynamic Rates')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: System Dynamics
    axes[1, 1].plot(df['iteration'], df['exploration_rate'], label='Exploration Rate', alpha=0.7)
    axes[1, 1].plot(df['iteration'], df['conflict_persistence'], label='Conflict Persistence', alpha=0.7)
    axes[1, 1].set_title('System Dynamics')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Phase Space
    axes[2, 0].scatter(df['cost'], df['entropy'], c=df['iteration'], cmap='viridis', alpha=0.6, s=10)
    axes[2, 0].set_title('Phase Space Trajectory')
    axes[2, 0].set_xlabel('Cost')
    axes[2, 0].set_ylabel('Entropy')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Conflicts over time
    axes[2, 1].plot(df['iteration'], df['num_conflicts'], alpha=0.7)
    axes[2, 1].set_title('Conflicts Over Time')
    axes[2, 1].set_xlabel('Iteration')
    axes[2, 1].set_ylabel('Number of Conflicts')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"thermodynamic_analysis_{tracking_file.replace('.csv', '.png')}")
    plt.show()
    
    return df


# Example usage
if __name__ == "__main__":
    # Run the test
    results = run_thermodynamic_test(seed=1, max_iterations=1000)
    
    # Analyze results if any F-CBS tracking files were generated
    import glob
    tracking_files = glob.glob("thermodynamic_trace_*.csv")
    
    for file in tracking_files:
        print(f"\n{'='*60}")
        analyze_thermodynamic_results(file)