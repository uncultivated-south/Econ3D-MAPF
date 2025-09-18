import numpy as np
import heapq
import random
import time
import math
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

class PathFinder:
    def __init__(self, grid):
        self.grid = grid
    
    def a_star(self, start, goal, vertex_constraints=None, edge_constraints=None, max_iterations=10000, timeout_seconds=30):
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
        """Run the complete experiment comparing CBS and F-CBS with vertex and edge conflicts"""
        results = []
        
        for scenario_id in range(num_scenarios):
            print(f"Running scenario {scenario_id + 1}/{num_scenarios}")
            
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
    
    def analyze_results(self, results_df):
        """Analyze and visualize results including vertex and edge conflict handling"""
        print("=== EXPERIMENT RESULTS (VERTEX + EDGE CONFLICTS) ===\n")
        
        # Success rates
        print("Success Rates:")
        success_rates = results_df.groupby('algorithm')['solved'].mean().sort_values(ascending=False)
        for alg, rate in success_rates.items():
            print(f"{alg}: {rate:.1%}")
        
        print("\nAverage Iterations (successful cases only):")
        solved_results = results_df[results_df['solved'] == True]
        avg_iterations = solved_results.groupby('algorithm')['iterations'].mean().sort_values()
        for alg, avg in avg_iterations.items():
            print(f"{alg}: {avg:.1f}")
        
        print("\nAverage Path Cost (successful cases only):")
        avg_cost = solved_results.groupby('algorithm')['cost'].mean().sort_values()
        for alg, cost in avg_cost.items():
            print(f"{alg}: {cost:.1f}")

        # Compare annealing vs non-annealing for each temperature
        print("\n=== ANNEALING COMPARISON ===")
        for temp in self.temperatures:
            no_anneal = solved_results[
                (solved_results['algorithm'] == f'F-CBS (T={temp})') & 
                (solved_results['annealing'] == False)
            ]
            with_anneal = solved_results[
                (solved_results['algorithm'] == f'F-CBS Anneal (T={temp}→0)') & 
                (solved_results['annealing'] == True)
            ]
            
            if len(no_anneal) > 0 and len(with_anneal) > 0:
                print(f"\nTemperature {temp}:")
                print(f"  No Annealing: {len(no_anneal)} solved, avg cost {no_anneal['cost'].mean():.1f}, avg iterations {no_anneal['iterations'].mean():.1f}")
                print(f"  With Annealing: {len(with_anneal)} solved, avg cost {with_anneal['cost'].mean():.1f}, avg iterations {with_anneal['iterations'].mean():.1f}")
                
                # Calculate improvement
                cost_improvement = (no_anneal['cost'].mean() - with_anneal['cost'].mean()) / no_anneal['cost'].mean() * 100
                iter_improvement = (no_anneal['iterations'].mean() - with_anneal['iterations'].mean()) / no_anneal['iterations'].mean() * 100
                print(f"  Cost improvement: {cost_improvement:.1f}%")
                print(f"  Iteration improvement: {iter_improvement:.1f}%")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Success rate comparison
        axes[0, 0].bar(range(len(success_rates)), success_rates.values)
        axes[0, 0].set_title('Success Rate by Algorithm (Vertex + Edge Conflicts)')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_xticks(range(len(success_rates)))
        axes[0, 0].set_xticklabels(success_rates.index, rotation=45, ha='right')
        
        # Iterations comparison
        if len(solved_results) > 0:
            algorithms = solved_results['algorithm'].unique()
            iteration_data = [solved_results[solved_results['algorithm'] == alg]['iterations'].values 
                             for alg in algorithms]
            non_empty_data = []
            non_empty_labels = []
            for i, data in enumerate(iteration_data):
                if len(data) > 0:
                    non_empty_data.append(data)
                    non_empty_labels.append(algorithms[i])
            
            if non_empty_data:
                axes[0, 1].boxplot(non_empty_data)
                axes[0, 1].set_xticklabels(non_empty_labels, rotation=45, ha='right')
            axes[0, 1].set_title('Iterations to Solution')
            axes[0, 1].set_ylabel('Iterations')
            
            # Cost comparison
            cost_data = [solved_results[solved_results['algorithm'] == alg]['cost'].values 
                        for alg in algorithms]
            non_empty_cost_data = []
            non_empty_cost_labels = []
            for i, data in enumerate(cost_data):
                if len(data) > 0:
                    non_empty_cost_data.append(data)
                    non_empty_cost_labels.append(algorithms[i])
            
            if non_empty_cost_data:
                axes[1, 0].boxplot(non_empty_cost_data)
                axes[1, 0].set_xticklabels(non_empty_cost_labels, rotation=45, ha='right')
            axes[1, 0].set_title('Solution Cost')
            axes[1, 0].set_ylabel('Total Path Cost')
        
        # Temperature vs Performance scatter plot
        temp_data = solved_results[solved_results['temperature'].notna()]
        if len(temp_data) > 0:
            scatter = axes[1, 1].scatter(temp_data['temperature'], temp_data['cost'], 
                                       c=temp_data['iterations'], cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Solution Cost')
            axes[1, 1].set_title('Temperature vs Cost (color = iterations)')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
        
        return success_rates, avg_iterations, avg_cost

# Example usage
if __name__ == "__main__":
    # Run experiment
    runner = ExperimentRunner()
    print("Starting F-CBS vs CBS comparison with VERTEX + EDGE conflicts...")
    print("Testing: CBS + 3 F-CBS variants + 3 F-CBS annealing variants = 7 total algorithms")
    print("This may take several minutes...")
    
    # Run experiment
    results = runner.run_experiment(num_scenarios=1000)  # Reduced for testing
    
    # Analyze results
    success_rates, avg_iterations, avg_cost = runner.analyze_results(results)
    
    # Save results
    results.to_csv('fcbs_vertex_edge_experiment_results.csv', index=False)
    print("\nResults saved to 'fcbs_vertex_edge_experiment_results.csv'")