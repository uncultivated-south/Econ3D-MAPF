import numpy as np
import heapq
import random
import time
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class Agent:
    id: int
    start: Tuple[int, int]
    goal: Tuple[int, int]
    path: List[Tuple[int, int]] = None

@dataclass
class VertexConstraint:
    agent: int
    location: Tuple[int, int]
    timestep: int
    
@dataclass
class EdgeConstraint:
    agent: int
    from_location: Tuple[int, int]
    to_location: Tuple[int, int]
    timestep: int

class CBSNode:
    def __init__(self):
        self.vertex_constraints = []
        self.edge_constraints = []
        self.solution = {}
        self.cost = 0
        self.conflicts = []
        self.entropy = 0.0
        self.free_energy = 0.0
        
    def __lt__(self, other):
        return self.cost < other.cost

class FCBSNode(CBSNode):
    def __lt__(self, other):
        return self.free_energy < other.free_energy

class Grid:
    def __init__(self, width=12, height=12, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
        
    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos not in self.obstacles
    
    def get_neighbors(self, pos):
        x, y = pos
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x, y)]  # Include staying in place
        return [n for n in neighbors if self.is_valid(n)]

class MovingAILoader:
    """Loader for Moving AI Lab benchmark files"""
    
    @staticmethod
    def load_map(map_file_path):
        """Load a .map file and return Grid object"""
        with open(map_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        type_line = lines[0].strip()
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
        map_line = lines[3].strip()  # "map" line
        
        # Parse grid
        obstacles = set()
        grid_lines = lines[4:4+height]
        
        for y, line in enumerate(grid_lines):
            line = line.strip()
            for x, char in enumerate(line):
                if char in ['@', 'O', 'T', 'W']:  # Various obstacle types
                    obstacles.add((x, y))
        
        return Grid(width, height, obstacles)
    
    @staticmethod
    def load_scenario(scen_file_path, num_agents=None):
        """Load a .scen file and return list of Agent objects"""
        agents = []
        
        with open(scen_file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header line if present
        start_idx = 1 if lines[0].startswith('version') else 0
        
        for i, line in enumerate(lines[start_idx:]):
            if num_agents and i >= num_agents:
                break
                
            parts = line.strip().split('\t')
            if len(parts) >= 9:  # Standard format
                bucket = int(parts[0])
                map_name = parts[1]
                map_width = int(parts[2])
                map_height = int(parts[3])
                start_x = int(parts[4])
                start_y = int(parts[5])
                goal_x = int(parts[6])
                goal_y = int(parts[7])
                optimal_length = float(parts[8])
                
                agent = Agent(i, (start_x, start_y), (goal_x, goal_y))
                agents.append(agent)
        
        return agents
    
    @staticmethod
    def get_benchmark_files(benchmark_dir):
        """Get all .map and .scen file pairs from benchmark directory"""
        map_files = []
        scen_files = []
        
        for root, dirs, files in os.walk(benchmark_dir):
            for file in files:
                if file.endswith('.map'):
                    map_files.append(os.path.join(root, file))
                elif file.endswith('.scen'):
                    scen_files.append(os.path.join(root, file))
        
        # Match map and scenario files
        benchmark_pairs = []
        for map_file in map_files:
            map_base = os.path.splitext(map_file)[0]
            for scen_file in scen_files:
                scen_base = os.path.splitext(scen_file)[0]
                # Check if they match (scenario files often have additional suffixes)
                if map_base in scen_base or os.path.basename(map_base) in os.path.basename(scen_base):
                    benchmark_pairs.append((map_file, scen_file))
        
        return benchmark_pairs

class PathFinder:
    def __init__(self, grid):
        self.grid = grid
    
    def a_star(self, start, goal, vertex_constraints=None, edge_constraints=None, max_iterations=1000000, timeout_seconds=30):
        """A* pathfinding with vertex and edge constraints"""
        vertex_constraints = vertex_constraints or []
        edge_constraints = edge_constraints or []
        
        # Build constraint tables
        vertex_constraint_table = defaultdict(set)
        edge_constraint_table = defaultdict(set)
        
        for c in vertex_constraints:
            vertex_constraint_table[c.timestep].add(c.location)
            
        for c in edge_constraints:
            edge_constraint_table[c.timestep].add((c.from_location, c.to_location))
        
        open_list = [(self.heuristic(start, goal), 0, start, [start])]
        closed_set = set()
        
        iteration = 0
        start_time = time.time()

        while open_list:
            # Check iteration limit
            iteration += 1
            if iteration > max_iterations:
                print(f"A* exceeded max iterations ({max_iterations}) for agent path {start} -> {goal}")
                return None
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                print(f"A* timeout ({timeout_seconds}s) for agent path {start} -> {goal}")
                return None
            
            _, cost, current, path = heapq.heappop(open_list)
            
            if current == goal:
                return path
            
            if (current, len(path)-1) in closed_set:
                continue
            closed_set.add((current, len(path)-1))
            
            for neighbor in self.grid.get_neighbors(current):
                timestep = len(path)
                
                # Check vertex constraint
                if neighbor in vertex_constraint_table[timestep]:
                    continue
                    
                # Check edge constraint
                if (current, neighbor) in edge_constraint_table[timestep]:
                    continue
                
                new_path = path + [neighbor]
                new_cost = cost + 1
                priority = new_cost + self.heuristic(neighbor, goal)
                heapq.heappush(open_list, (priority, new_cost, neighbor, new_path))
        
        return None  # No path found
    
    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class ConflictDetector:
    @staticmethod
    def find_conflicts(paths):
        """Find both vertex and edge conflicts between agents"""
        conflicts = []
        agents = list(paths.keys())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                max_len = max(len(path1), len(path2))
                
                for t in range(max_len):
                    pos1 = path1[t] if t < len(path1) else path1[-1]
                    pos2 = path2[t] if t < len(path2) else path2[-1]
                    
                    # Check vertex conflicts
                    if pos1 == pos2:
                        conflicts.append({
                            'type': 'vertex',
                            'agents': [agent1, agent2],
                            'location': pos1,
                            'timestep': t
                        })
                    
                    # Check edge conflicts (swapping positions)
                    if t > 0:
                        prev_pos1 = path1[t-1] if t-1 < len(path1) else path1[-1]
                        prev_pos2 = path2[t-1] if t-1 < len(path2) else path2[-1]
                        
                        # Edge conflict: agents swap positions
                        if pos1 == prev_pos2 and pos2 == prev_pos1 and pos1 != pos2:
                            conflicts.append({
                                'type': 'edge',
                                'agents': [agent1, agent2],
                                'from_locations': [prev_pos1, prev_pos2],
                                'to_locations': [pos1, pos2],
                                'timestep': t
                            })
        
        return conflicts

class CBS:
    def __init__(self, grid, agents):
        self.grid = grid
        self.agents = agents
        self.pathfinder = PathFinder(grid)
        self.conflict_detector = ConflictDetector()
    
    def generate_constraints_for_conflict(self, conflict, agent_id):
        """Generate appropriate constraints based on conflict type"""
        constraints = {'vertex': [], 'edge': []}
        
        if conflict['type'] == 'vertex':
            # Vertex conflict: constrain the agent from being at the location at the timestep
            constraints['vertex'].append(
                VertexConstraint(agent_id, conflict['location'], conflict['timestep'])
            )
        
        elif conflict['type'] == 'edge':
            # Edge conflict: constrain the specific edge movement
            agent_index = conflict['agents'].index(agent_id)
            from_loc = conflict['from_locations'][agent_index]
            to_loc = conflict['to_locations'][agent_index]
            
            constraints['edge'].append(
                EdgeConstraint(agent_id, from_loc, to_loc, conflict['timestep'])
            )
        
        return constraints
        
    def solve(self, max_iterations=1024):
        start_time = time.time()
        root = CBSNode()
        
        # Find initial solution
        for agent in self.agents:
            path = self.pathfinder.a_star(
                agent.start, agent.goal, 
                root.vertex_constraints, root.edge_constraints
            )
            if path is None:
                return None, 0, time.time() - start_time
            root.solution[agent.id] = path
            root.cost += len(path) - 1
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_list)
            
            if not current.conflicts:
                return current.solution, iterations, time.time() - start_time
            
            conflict = current.conflicts[0]
            
            for agent_id in conflict['agents']:
                new_node = CBSNode()
                new_node.vertex_constraints = current.vertex_constraints.copy()
                new_node.edge_constraints = current.edge_constraints.copy()
                
                # Generate appropriate constraints based on conflict type
                new_constraints = self.generate_constraints_for_conflict(conflict, agent_id)
                new_node.vertex_constraints.extend(new_constraints['vertex'])
                new_node.edge_constraints.extend(new_constraints['edge'])
                
                # Replan for constrained agent
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
                    
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time

class FCBS(CBS):
    def __init__(self, grid, agents, temperature=1.0, using_annealing=False, annealing_iterations=5):
        super().__init__(grid, agents)
        self.initial_temperature = temperature
        self.current_temperature = temperature
        self.use_annealing = using_annealing
        self.annealing_iterations = annealing_iterations
    
    def update_temperature(self, iteration):
        """Update temperature based on annealing schedule"""
        if self.use_annealing and iteration > self.annealing_iterations:
            self.current_temperature = 0.0
        else:
            self.current_temperature = self.initial_temperature

    def calculate_entropy(self, conflicts):
        """Calculate information entropy based on conflict distribution (vertex + edge conflicts)"""
        if not conflicts:
            return 0.0
        
        # Count conflicts per grid cell
        conflict_counts = defaultdict(float)  # Use float for fractional counts
        total_conflict_weight = 0.0
        
        for conflict in conflicts:
            if conflict['type'] == 'vertex':
                # Vertex conflict: full weight (1.0) to the location
                conflict_counts[conflict['location']] += 1.0
                total_conflict_weight += 1.0
                
            elif conflict['type'] == 'edge':
                # Edge conflict: 0.5 weight to each location involved
                for to_loc in conflict['to_locations']:
                    conflict_counts[to_loc] += 0.5
                total_conflict_weight += 1.0  # Total edge conflict weight is 1.0 (0.5 + 0.5)
        
        if total_conflict_weight == 0.0:
            return 0.0
        
        # Calculate entropy using the weighted conflict distribution
        entropy = 0.0
        for count in conflict_counts.values():
            if count > 0:
                p = count / total_conflict_weight
                entropy -= p * math.log2(p)
        
        return entropy
    
    def solve(self, max_iterations=1024):
        start_time = time.time()
        root = FCBSNode()
        
        # Find initial solution
        for agent in self.agents:
            path = self.pathfinder.a_star(
                agent.start, agent.goal,
                root.vertex_constraints, root.edge_constraints
            )
            if path is None:
                return None, 0, time.time() - start_time
            root.solution[agent.id] = path
            root.cost += len(path) - 1
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        root.entropy = self.calculate_entropy(root.conflicts)
        root.free_energy = root.cost + self.current_temperature * root.entropy
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1

            # Update temperature based on annealing schedule
            self.update_temperature(iterations)

            current = heapq.heappop(open_list)
            
            if not current.conflicts:
                return current.solution, iterations, time.time() - start_time
            
            conflict = current.conflicts[0]
            
            for agent_id in conflict['agents']:
                new_node = FCBSNode()
                new_node.vertex_constraints = current.vertex_constraints.copy()
                new_node.edge_constraints = current.edge_constraints.copy()
                
                # Generate appropriate constraints based on conflict type
                new_constraints = self.generate_constraints_for_conflict(conflict, agent_id)
                new_node.vertex_constraints.extend(new_constraints['vertex'])
                new_node.edge_constraints.extend(new_constraints['edge'])
                
                # Replan for constrained agent
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
                    # Use current temperature for free energy calculation
                    new_node.free_energy = new_node.cost + self.current_temperature * new_node.entropy
                    
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time

class BenchmarkExperimentRunner:
    def __init__(self, benchmark_dir, max_agents=10):
        self.benchmark_dir = benchmark_dir
        self.max_agents = max_agents
        self.temperatures = [1.0, 2.0, 5.0, 10.0, 12.0]
        self.loader = MovingAILoader()
    
    def run_benchmark_experiment(self, max_scenarios=None):
        """Run experiment on Moving AI Lab benchmarks"""
        results = []
        
        # Get all benchmark file pairs
        benchmark_pairs = self.loader.get_benchmark_files(self.benchmark_dir)
        
        if not benchmark_pairs:
            print(f"No benchmark files found in {self.benchmark_dir}")
            print("Please ensure you have .map and .scen files in the directory")
            return pd.DataFrame()
        
        print(f"Found {len(benchmark_pairs)} benchmark scenarios")
        
        # Limit scenarios if specified
        if max_scenarios:
            benchmark_pairs = benchmark_pairs[:max_scenarios]
        
        for scenario_id, (map_file, scen_file) in enumerate(benchmark_pairs):
            print(f"Running benchmark {scenario_id + 1}/{len(benchmark_pairs)}: {os.path.basename(map_file)}")
            
            try:
                # Load map and scenario
                grid = self.loader.load_map(map_file)
                agents = self.loader.load_scenario(scen_file, num_agents=self.max_agents)
                
                if not agents:
                    print(f"No agents found in {scen_file}, skipping...")
                    continue
                
                print(f"  Loaded {len(agents)} agents on {grid.width}x{grid.height} grid")
                
                # Test traditional CBS
                cbs = CBS(grid, agents)
                cbs_solution, cbs_iterations, cbs_time = cbs.solve()
                cbs_cost = sum(len(path) - 1 for path in cbs_solution.values()) if cbs_solution else float('inf')

                results.append({
                    'scenario': scenario_id,
                    'map_file': os.path.basename(map_file),
                    'scen_file': os.path.basename(scen_file),
                    'num_agents': len(agents),
                    'grid_size': f"{grid.width}x{grid.height}",
                    'algorithm': 'CBS',
                    'temperature': None,
                    'annealing': False,
                    'solved': cbs_solution is not None,
                    'iterations': cbs_iterations,
                    'cost': cbs_cost,
                    'time': cbs_time
                })
                
                # Test F-CBS at different temperatures (without annealing)
                for temp in self.temperatures:
                    fcbs = FCBS(grid, agents, temperature=temp, using_annealing=False)
                    fcbs_solution, fcbs_iterations, fcbs_time = fcbs.solve()
                    fcbs_cost = sum(len(path) - 1 for path in fcbs_solution.values()) if fcbs_solution else float('inf')
                    
                    results.append({
                        'scenario': scenario_id,
                        'map_file': os.path.basename(map_file),
                        'scen_file': os.path.basename(scen_file),
                        'num_agents': len(agents),
                        'grid_size': f"{grid.width}x{grid.height}",
                        'algorithm': f'F-CBS (T={temp})',
                        'temperature': temp,
                        'annealing': False,
                        'solved': fcbs_solution is not None,
                        'iterations': fcbs_iterations,
                        'cost': fcbs_cost,
                        'time': fcbs_time
                    })
                
                # Test F-CBS with annealing
                for temp in self.temperatures:
                    fcbs_anneal = FCBS(grid, agents, temperature=temp, using_annealing=True, annealing_iterations=5)
                    fcbs_anneal_solution, fcbs_anneal_iterations, fcbs_anneal_time = fcbs_anneal.solve()
                    fcbs_anneal_cost = sum(len(path) - 1 for path in fcbs_anneal_solution.values()) if fcbs_anneal_solution else float('inf')

                    results.append({
                        'scenario': scenario_id,
                        'map_file': os.path.basename(map_file),
                        'scen_file': os.path.basename(scen_file),
                        'num_agents': len(agents),
                        'grid_size': f"{grid.width}x{grid.height}",
                        'algorithm': f'F-CBS Anneal (T={temp}→0)',
                        'temperature': temp,
                        'annealing': True,
                        'solved': fcbs_anneal_solution is not None,
                        'iterations': fcbs_anneal_iterations,
                        'cost': fcbs_anneal_cost,
                        'time': fcbs_anneal_time
                    })
                    
            except Exception as e:
                print(f"Error processing {map_file}: {e}")
                continue

        return pd.DataFrame(results)
    
    def analyze_benchmark_results(self, results_df):
        """Analyze and visualize benchmark results"""
        if results_df.empty:
            print("No results to analyze")
            return
            
        print("=== MOVING AI LAB BENCHMARK RESULTS ===\n")
        
        # Overall statistics
        print(f"Total benchmark scenarios tested: {results_df['scenario'].nunique()}")
        print(f"Grid sizes tested: {sorted(results_df['grid_size'].unique())}")
        print(f"Agent counts tested: {sorted(results_df['num_agents'].unique())}")
        
        # Success rates
        print("\nSuccess Rates:")
        success_rates = results_df.groupby('algorithm')['solved'].mean().sort_values(ascending=False)
        for alg, rate in success_rates.items():
            print(f"{alg}: {rate:.1%}")
        
        # Analysis for successful cases only
        solved_results = results_df[results_df['solved'] == True]
        
        if len(solved_results) > 0:
            print("\nAverage Iterations (successful cases only):")
            avg_iterations = solved_results.groupby('algorithm')['iterations'].mean().sort_values()
            for alg, avg in avg_iterations.items():
                print(f"{alg}: {avg:.1f}")
            
            print("\nAverage Path Cost (successful cases only):")
            avg_cost = solved_results.groupby('algorithm')['cost'].mean().sort_values()
            for alg, cost in avg_cost.items():
                print(f"{alg}: {cost:.1f}")
            
            print("\nAverage Runtime (successful cases only):")
            avg_time = solved_results.groupby('algorithm')['time'].mean().sort_values()
            for alg, time_val in avg_time.items():
                print(f"{alg}: {time_val:.3f}s")
        
        # Performance by grid size
        print("\n=== PERFORMANCE BY GRID SIZE ===")
        for grid_size in sorted(results_df['grid_size'].unique()):
            grid_data = results_df[results_df['grid_size'] == grid_size]
            print(f"\nGrid {grid_size}:")
            grid_success = grid_data.groupby('algorithm')['solved'].mean()
            for alg, rate in grid_success.items():
                scenarios_count = len(grid_data[grid_data['algorithm'] == alg])
                print(f"  {alg}: {rate:.1%} ({scenarios_count} scenarios)")
        
        # Performance by agent count
        print("\n=== PERFORMANCE BY AGENT COUNT ===")
        for agent_count in sorted(results_df['num_agents'].unique()):
            agent_data = results_df[results_df['num_agents'] == agent_count]
            print(f"\n{agent_count} agents:")
            agent_success = agent_data.groupby('algorithm')['solved'].mean()
            for alg, rate in agent_success.items():
                scenarios_count = len(agent_data[agent_data['algorithm'] == alg])
                print(f"  {alg}: {rate:.1%} ({scenarios_count} scenarios)")
        
        # Create visualizations
        self._create_benchmark_visualizations(results_df, solved_results)
        
        return results_df
    
    def _create_benchmark_visualizations(self, results_df, solved_results):
        """Create comprehensive visualizations for benchmark results"""
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        
        # Success rate comparison
        success_rates = results_df.groupby('algorithm')['solved'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(success_rates)), success_rates.values)
        axes[0, 0].set_title('Success Rate by Algorithm (Moving AI Benchmarks)')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_xticks(range(len(success_rates)))
        axes[0, 0].set_xticklabels(success_rates.index, rotation=45, ha='right')
        
        # Iterations comparison (successful cases)
        if len(solved_results) > 0:
            algorithms = solved_results['algorithm'].unique()
            iteration_data = [solved_results[solved_results['algorithm'] == alg]['iterations'].values 
                             for alg in algorithms if len(solved_results[solved_results['algorithm'] == alg]) > 0]
            iteration_labels = [alg for alg in algorithms if len(solved_results[solved_results['algorithm'] == alg]) > 0]
            
            if iteration_data:
                axes[0, 1].boxplot(iteration_data)
                axes[0, 1].set_xticklabels(iteration_labels, rotation=45, ha='right')
            axes[0, 1].set_title('Iterations to Solution')
            axes[0, 1].set_ylabel('Iterations')
            
            # Cost comparison
            cost_data = [solved_results[solved_results['algorithm'] == alg]['cost'].values 
                        for alg in algorithms if len(solved_results[solved_results['algorithm'] == alg]) > 0]
            cost_labels = [alg for alg in algorithms if len(solved_results[solved_results['algorithm'] == alg]) > 0]
            
            if cost_data:
                axes[0, 2].boxplot(cost_data)
                axes[0, 2].set_xticklabels(cost_labels, rotation=45, ha='right')
            axes[0, 2].set_title('Solution Cost')
            axes[0, 2].set_ylabel('Total Path Cost')
        
        # Success rate by grid size
        grid_sizes = sorted(results_df['grid_size'].unique())
        alg_names = ['CBS', 'F-CBS (T=1.0)', 'F-CBS (T=5.0)', 'F-CBS Anneal (T=5.0→0)']
        x = np.arange(len(grid_sizes))
        width = 0.15
        
        for i, alg in enumerate(alg_names):
            if alg in results_df['algorithm'].values:
                success_by_grid = []
                for grid_size in grid_sizes:
                    grid_alg_data = results_df[(results_df['grid_size'] == grid_size) & (results_df['algorithm'] == alg)]
                    success_rate = grid_alg_data['solved'].mean() if len(grid_alg_data) > 0 else 0
                    success_by_grid.append(success_rate)
                
                axes[1, 0].bar(x + i*width, success_by_grid, width, label=alg)
        
        axes[1, 0].set_xlabel('Grid Size')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate by Grid Size')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels(grid_sizes)
        axes[1, 0].legend()
        
        # Success rate by agent count
        agent_counts = sorted(results_df['num_agents'].unique())
        x = np.arange(len(agent_counts))
        
        for i, alg in enumerate(alg_names):
            if alg in results_df['algorithm'].values:
                success_by_agents = []
                for agent_count in agent_counts:
                    agent_alg_data = results_df[(results_df['num_agents'] == agent_count) & (results_df['algorithm'] == alg)]
                    success_rate = agent_alg_data['solved'].mean() if len(agent_alg_data) > 0 else 0
                    success_by_agents.append(success_rate)
                
                axes[1, 1].bar(x + i*width, success_by_agents, width, label=alg)
        
        axes[1, 1].set_xlabel('Number of Agents')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_title('Success Rate by Agent Count')
        axes[1, 1].set_xticks(x + width * 1.5)
        axes[1, 1].set_xticklabels(agent_counts)
        axes[1, 1].legend()
        
        # Temperature vs Performance scatter plot (F-CBS variants only)
        temp_data = solved_results[solved_results['temperature'].notna()]
        if len(temp_data) > 0:
            scatter = axes[1, 2].scatter(temp_data['temperature'], temp_data['cost'], 
                                       c=temp_data['iterations'], cmap='viridis', alpha=0.6, s=30)
            axes[1, 2].set_xlabel('Temperature')
            axes[1, 2].set_ylabel('Solution Cost')
            axes[1, 2].set_title('Temperature vs Cost (color = iterations)')
            plt.colorbar(scatter, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.show()

# Legacy ExperimentRunner for random scenarios (kept for backward compatibility)
class ExperimentRunner:
    def __init__(self, grid_size=(12, 12), num_agents=10, num_obstacles=16):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.temperatures = [1.0, 2.0, 5.0, 10.0, 12.0]
    
    def generate_scenario(self, seed):
        """Generate a random scenario with given seed"""
        random.seed(seed)
        np.random.seed(seed)
        
        width, height = self.grid_size
        
        # Generate random obstacles
        obstacles = set()
        while len(obstacles) < self.num_obstacles:
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            obstacles.add((x, y))
        
        grid = Grid(width, height, obstacles)
        
        # Generate random agents
        agents = []
        occupied = obstacles.copy()
        
        for i in range(self.num_agents):
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
        
        return grid, agents
    
    def run_experiment(self, num_scenarios=100):
        """Run the complete experiment comparing CBS and F-CBS"""
        results = []
        
        for scenario_id in range(num_scenarios):
            print(f"Running random scenario {scenario_id + 1}/{num_scenarios}")
            
            grid, agents = self.generate_scenario(scenario_id)
            
            # Test traditional CBS
            cbs = CBS(grid, agents)
            cbs_solution, cbs_iterations, cbs_time = cbs.solve()
            cbs_cost = sum(len(path) - 1 for path in cbs_solution.values()) if cbs_solution else float('inf')

            results.append({
                'scenario': scenario_id,
                'algorithm': 'CBS',
                'temperature': None,
                'annealing': False,
                'solved': cbs_solution is not None,
                'iterations': cbs_iterations,
                'cost': cbs_cost,
                'time': cbs_time
            })
            
            # Test F-CBS at different temperatures (without annealing)
            for temp in self.temperatures:
                fcbs = FCBS(grid, agents, temperature=temp, using_annealing=False)
                fcbs_solution, fcbs_iterations, fcbs_time = fcbs.solve()
                fcbs_cost = sum(len(path) - 1 for path in fcbs_solution.values()) if fcbs_solution else float('inf')
                
                results.append({
                    'scenario': scenario_id,
                    'algorithm': f'F-CBS (T={temp})',
                    'temperature': temp,
                    'annealing': False,
                    'solved': fcbs_solution is not None,
                    'iterations': fcbs_iterations,
                    'cost': fcbs_cost,
                    'time': fcbs_time
                })
            
            # Test F-CBS with annealing
            for temp in self.temperatures:
                fcbs_anneal = FCBS(grid, agents, temperature=temp, using_annealing=True, annealing_iterations=5)
                fcbs_anneal_solution, fcbs_anneal_iterations, fcbs_anneal_time = fcbs_anneal.solve()
                fcbs_anneal_cost = sum(len(path) - 1 for path in fcbs_anneal_solution.values()) if fcbs_anneal_solution else float('inf')

                results.append({
                    'scenario': scenario_id,
                    'algorithm': f'F-CBS Anneal (T={temp}→0)',
                    'temperature': temp,
                    'annealing': True,
                    'solved': fcbs_anneal_solution is not None,
                    'iterations': fcbs_anneal_iterations,
                    'cost': fcbs_anneal_cost,
                    'time': fcbs_anneal_time
                })

        return pd.DataFrame(results)

# Example usage functions
def run_moving_ai_benchmarks(benchmark_directory, max_agents=8, max_scenarios=50):
    """
    Run F-CBS vs CBS comparison on Moving AI Lab benchmarks
    
    Args:
        benchmark_directory: Path to directory containing .map and .scen files
        max_agents: Maximum number of agents to test per scenario
        max_scenarios: Maximum number of benchmark scenarios to test
    """
    print("=== F-CBS vs CBS on Moving AI Lab Benchmarks ===")
    print(f"Benchmark directory: {benchmark_directory}")
    print(f"Max agents per scenario: {max_agents}")
    print(f"Max scenarios to test: {max_scenarios}")
    
    runner = BenchmarkExperimentRunner(benchmark_directory, max_agents=max_agents)
    
    # Run the experiment
    results = runner.run_benchmark_experiment(max_scenarios=max_scenarios)
    
    if not results.empty:
        # Analyze results
        runner.analyze_benchmark_results(results)
        
        # Save results
        output_file = 'fcbs_movingai_benchmark_results.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to '{output_file}'")
    else:
        print("No results generated. Please check your benchmark directory and files.")
    
    return results

def run_random_scenarios(num_scenarios=100, grid_size=(12, 12), num_agents=10):
    """
    Run F-CBS vs CBS comparison on randomly generated scenarios
    
    Args:
        num_scenarios: Number of random scenarios to generate and test
        grid_size: Tuple of (width, height) for the grid
        num_agents: Number of agents per scenario
    """
    print("=== F-CBS vs CBS on Random Scenarios ===")
    print(f"Grid size: {grid_size}")
    print(f"Agents per scenario: {num_agents}")
    print(f"Number of scenarios: {num_scenarios}")
    
    runner = ExperimentRunner(grid_size=grid_size, num_agents=num_agents)
    
    # Run the experiment
    results = runner.run_experiment(num_scenarios=num_scenarios)
    
    # Analyze results (reuse the benchmark analysis method)
    benchmark_runner = BenchmarkExperimentRunner(".", max_agents=num_agents)
    benchmark_runner.analyze_benchmark_results(results)
    
    # Save results
    output_file = 'fcbs_random_experiment_results.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    return results

# Test with a single benchmark scenario
def test_single_benchmark(map_file, scen_file, num_agents=5):
    """
    Test a single benchmark scenario for debugging/demonstration
    
    Args:
        map_file: Path to .map file
        scen_file: Path to .scen file  
        num_agents: Number of agents to test
    """
    print(f"=== Testing Single Benchmark ===")
    print(f"Map: {map_file}")
    print(f"Scenario: {scen_file}")
    print(f"Agents: {num_agents}")
    
    loader = MovingAILoader()
    
    try:
        # Load the benchmark
        grid = loader.load_map(map_file)
        agents = loader.load_scenario(scen_file, num_agents=num_agents)
        
        print(f"Loaded {len(agents)} agents on {grid.width}x{grid.height} grid")
        print(f"Obstacles: {len(grid.obstacles)}")
        
        # Test CBS
        print("\nTesting CBS...")
        cbs = CBS(grid, agents)
        cbs_solution, cbs_iterations, cbs_time = cbs.solve()
        
        if cbs_solution:
            cbs_cost = sum(len(path) - 1 for path in cbs_solution.values())
            print(f"CBS: SUCCESS - Cost: {cbs_cost}, Iterations: {cbs_iterations}, Time: {cbs_time:.3f}s")
        else:
            print(f"CBS: FAILED - Iterations: {cbs_iterations}, Time: {cbs_time:.3f}s")
        
        # Test F-CBS
        print("\nTesting F-CBS (T=5.0)...")
        fcbs = FCBS(grid, agents, temperature=5.0, using_annealing=False)
        fcbs_solution, fcbs_iterations, fcbs_time = fcbs.solve()
        
        if fcbs_solution:
            fcbs_cost = sum(len(path) - 1 for path in fcbs_solution.values())
            print(f"F-CBS: SUCCESS - Cost: {fcbs_cost}, Iterations: {fcbs_iterations}, Time: {fcbs_time:.3f}s")
        else:
            print(f"F-CBS: FAILED - Iterations: {fcbs_iterations}, Time: {fcbs_time:.3f}s")
            
        # Test F-CBS with annealing
        print("\nTesting F-CBS with annealing (T=5.0→0)...")
        fcbs_anneal = FCBS(grid, agents, temperature=5.0, using_annealing=True, annealing_iterations=5)
        fcbs_anneal_solution, fcbs_anneal_iterations, fcbs_anneal_time = fcbs_anneal.solve()
        
        if fcbs_anneal_solution:
            fcbs_anneal_cost = sum(len(path) - 1 for path in fcbs_anneal_solution.values())
            print(f"F-CBS Anneal: SUCCESS - Cost: {fcbs_anneal_cost}, Iterations: {fcbs_anneal_iterations}, Time: {fcbs_anneal_time:.3f}s")
        else:
            print(f"F-CBS Anneal: FAILED - Iterations: {fcbs_anneal_iterations}, Time: {fcbs_anneal_time:.3f}s")
        
    except Exception as e:
        print(f"Error: {e}")

# Main execution examples
if __name__ == "__main__":
    # Example 1: Test on Moving AI Lab benchmarks
    # You need to download benchmark files from https://movingai.com/benchmarks/mapf.html
    # and extract them to a directory, e.g., "benchmarks/"
    
    # Uncomment and modify the path to your benchmark directory:
    results = run_moving_ai_benchmarks("benchmarks/", max_agents=3, max_scenarios=20)
    
    # Example 2: Test on random scenarios (works without benchmark files)
    print("Running experiment on random scenarios...")
    # results = run_random_scenarios(num_scenarios=50, grid_size=(12, 12), num_agents=8)
    
    # Example 3: Test a single benchmark file (for debugging)
    # Uncomment and provide paths to specific .map and .scen files:
    # test_single_benchmark("benchmarks/empty-8-8.map", "benchmarks/empty-8-8-random-1.scen", num_agents=4)