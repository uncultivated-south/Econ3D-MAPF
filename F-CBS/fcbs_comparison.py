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
class Constraint:
    agent: int
    location: Tuple[int, int]
    timestep: int
    
class CBSNode:
    def __init__(self):
        self.constraints = []
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
    
    def a_star(self, start, goal, constraints=None):
        """A* pathfinding with constraints"""
        constraints = constraints or []
        constraint_table = defaultdict(set)
        
        # Build constraint table
        for c in constraints:
            constraint_table[c.timestep].add(c.location)
        
        open_list = [(self.heuristic(start, goal), 0, start, [start])]
        closed_set = set()
        
        while open_list:
            _, cost, current, path = heapq.heappop(open_list)
            
            if current == goal:
                return path
            
            if (current, len(path)-1) in closed_set:
                continue
            closed_set.add((current, len(path)-1))
            
            for neighbor in self.grid.get_neighbors(current):
                timestep = len(path)
                if neighbor not in constraint_table[timestep]:
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
        conflicts = []
        agents = list(paths.keys())
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                # Check vertex conflicts
                max_len = max(len(path1), len(path2))
                for t in range(max_len):
                    pos1 = path1[t] if t < len(path1) else path1[-1]
                    pos2 = path2[t] if t < len(path2) else path2[-1]
                    
                    if pos1 == pos2:
                        conflicts.append({
                            'type': 'vertex',
                            'agents': [agent1, agent2],
                            'location': pos1,
                            'timestep': t
                        })
                        break
        
        return conflicts

class CBS:
    def __init__(self, grid, agents):
        self.grid = grid
        self.agents = agents
        self.pathfinder = PathFinder(grid)
        self.conflict_detector = ConflictDetector()
        
    def solve(self, max_iterations=1024):
        start_time = time.time()
        root = CBSNode()
        
        # Find initial solution
        for agent in self.agents:
            path = self.pathfinder.a_star(agent.start, agent.goal, root.constraints)
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
                new_node.constraints = current.constraints.copy()
                new_node.constraints.append(
                    Constraint(agent_id, conflict['location'], conflict['timestep'])
                )
                
                # Replan for constrained agent
                agent = next(a for a in self.agents if a.id == agent_id)
                new_path = self.pathfinder.a_star(agent.start, agent.goal, new_node.constraints)
                
                if new_path is not None:
                    new_node.solution = current.solution.copy()
                    new_node.solution[agent_id] = new_path
                    new_node.cost = sum(len(path) - 1 for path in new_node.solution.values())
                    new_node.conflicts = self.conflict_detector.find_conflicts(new_node.solution)
                    
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time

class FCBS(CBS):
    def __init__(self, grid, agents, temperature=1.0):
        super().__init__(grid, agents)
        self.temperature = temperature
    
    def calculate_entropy(self, conflicts):
        """Calculate information entropy based on conflict distribution"""
        if not conflicts:
            return 0.0
        
        # Count conflicts per grid cell
        conflict_counts = defaultdict(int)
        total_conflicts = 0
        
        for conflict in conflicts:
            conflict_counts[conflict['location']] += 1
            total_conflicts += 1
        
        if total_conflicts == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in conflict_counts.values():
            if count > 0:
                p = count / total_conflicts
                entropy -= p * math.log2(p)
        
        return entropy
    
    def solve(self, max_iterations=1024):
        start_time = time.time()
        root = FCBSNode()
        
        # Find initial solution
        for agent in self.agents:
            path = self.pathfinder.a_star(agent.start, agent.goal, root.constraints)
            if path is None:
                return None, 0, time.time() - start_time
            root.solution[agent.id] = path
            root.cost += len(path) - 1
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        root.entropy = self.calculate_entropy(root.conflicts)
        root.free_energy = root.cost - self.temperature * root.entropy
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_list)
            
            if not current.conflicts:
                return current.solution, iterations, time.time() - start_time
            
            conflict = current.conflicts[0]
            
            for agent_id in conflict['agents']:
                new_node = FCBSNode()
                new_node.constraints = current.constraints.copy()
                new_node.constraints.append(
                    Constraint(agent_id, conflict['location'], conflict['timestep'])
                )
                
                # Replan for constrained agent
                agent = next(a for a in self.agents if a.id == agent_id)
                new_path = self.pathfinder.a_star(agent.start, agent.goal, new_node.constraints)
                
                if new_path is not None:
                    new_node.solution = current.solution.copy()
                    new_node.solution[agent_id] = new_path
                    new_node.cost = sum(len(path) - 1 for path in new_node.solution.values())
                    new_node.conflicts = self.conflict_detector.find_conflicts(new_node.solution)
                    new_node.entropy = self.calculate_entropy(new_node.conflicts)
                    new_node.free_energy = new_node.cost - self.temperature * new_node.entropy
                    
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time

class ExperimentRunner:
    def __init__(self, grid_size=(12, 12), num_agents=12, num_obstacles=16):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.temperatures = [0.1, 0.2, 0.5, 1.0]
    
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
            print(f"Running scenario {scenario_id + 1}/{num_scenarios}")
            
            grid, agents = self.generate_scenario(scenario_id)
            
            # Test traditional CBS
            cbs = CBS(grid, agents)
            cbs_solution, cbs_iterations, cbs_time = cbs.solve()
            cbs_cost = sum(len(path) - 1 for path in cbs_solution.values()) if cbs_solution else float('inf')
            
            # Test F-CBS at different temperatures
            for temp in self.temperatures:
                fcbs = FCBS(grid, agents, temperature=temp)
                fcbs_solution, fcbs_iterations, fcbs_time = fcbs.solve()
                fcbs_cost = sum(len(path) - 1 for path in fcbs_solution.values()) if fcbs_solution else float('inf')
                
                results.append({
                    'scenario': scenario_id,
                    'algorithm': f'F-CBS (T={temp})',
                    'temperature': temp,
                    'solved': fcbs_solution is not None,
                    'iterations': fcbs_iterations,
                    'cost': fcbs_cost,
                    'time': fcbs_time
                })
            
            results.append({
                'scenario': scenario_id,
                'algorithm': 'CBS',
                'temperature': None,
                'solved': cbs_solution is not None,
                'iterations': cbs_iterations,
                'cost': cbs_cost,
                'time': cbs_time
            })
        
        return pd.DataFrame(results)
    
    def analyze_results(self, results_df):
        """Analyze and visualize results"""
        print("=== EXPERIMENT RESULTS ===\n")
        
        # Success rates
        print("Success Rates:")
        success_rates = results_df.groupby('algorithm')['solved'].mean()
        for alg, rate in success_rates.items():
            print(f"{alg}: {rate:.1%}")
        
        print("\nAverage Iterations (successful cases only):")
        solved_results = results_df[results_df['solved'] == True]
        avg_iterations = solved_results.groupby('algorithm')['iterations'].mean()
        for alg, avg in avg_iterations.items():
            print(f"{alg}: {avg:.1f}")
        
        print("\nAverage Path Cost (successful cases only):")
        avg_cost = solved_results.groupby('algorithm')['cost'].mean()
        for alg, cost in avg_cost.items():
            print(f"{alg}: {cost:.1f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate comparison
        axes[0, 0].bar(range(len(success_rates)), success_rates.values)
        axes[0, 0].set_title('Success Rate by Algorithm')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_xticks(range(len(success_rates)))
        axes[0, 0].set_xticklabels(success_rates.index, rotation=45, ha='right')
        
        # Iterations comparison - manual boxplot
        if len(solved_results) > 0:
            algorithms = solved_results['algorithm'].unique()
            iteration_data = [solved_results[solved_results['algorithm'] == alg]['iterations'].values 
                             for alg in algorithms]
            # Filter out empty arrays
            non_empty_data = []
            non_empty_labels = []
            for i, data in enumerate(iteration_data):
                if len(data) > 0:
                    non_empty_data.append(data)
                    non_empty_labels.append(algorithms[i])
            
            if non_empty_data:
                bp1 = axes[0, 1].boxplot(non_empty_data)
                axes[0, 1].set_xticklabels(non_empty_labels, rotation=45, ha='right')
            axes[0, 1].set_title('Iterations to Solution')
            axes[0, 1].set_ylabel('Iterations')
            
            # Cost comparison - manual boxplot
            cost_data = [solved_results[solved_results['algorithm'] == alg]['cost'].values 
                        for alg in algorithms]
            # Filter out empty arrays
            non_empty_cost_data = []
            non_empty_cost_labels = []
            for i, data in enumerate(cost_data):
                if len(data) > 0:
                    non_empty_cost_data.append(data)
                    non_empty_cost_labels.append(algorithms[i])
            
            if non_empty_cost_data:
                bp2 = axes[1, 0].boxplot(non_empty_cost_data)
                axes[1, 0].set_xticklabels(non_empty_cost_labels, rotation=45, ha='right')
            axes[1, 0].set_title('Solution Cost')
            axes[1, 0].set_ylabel('Total Path Cost')
        else:
            axes[0, 1].text(0.5, 0.5, 'No solutions found', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Iterations to Solution')
            
            axes[1, 0].text(0.5, 0.5, 'No solutions found', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Solution Cost')
        
        # Temperature analysis for F-CBS
        fcbs_results = solved_results[solved_results['algorithm'].str.startswith('F-CBS')]
        if len(fcbs_results) > 0:
            temp_analysis = fcbs_results.groupby('temperature')['iterations'].mean()
            axes[1, 1].plot(temp_analysis.index, temp_analysis.values, 'o-')
            axes[1, 1].set_title('F-CBS Performance vs Temperature')
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Average Iterations')
        else:
            axes[1, 1].text(0.5, 0.5, 'No F-CBS solutions found', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('F-CBS Performance vs Temperature')
        
        plt.tight_layout()
        plt.show()
        
        return success_rates, avg_iterations, avg_cost

# Example usage
if __name__ == "__main__":
    # Run experiment
    runner = ExperimentRunner()
    print("Starting F-CBS vs CBS comparison experiment...")
    print("This may take several minutes...")
    
    # Run on smaller sample first for testing
    results = runner.run_experiment(num_scenarios=30)  # Change to 100 for full experiment
    
    # Analyze results
    success_rates, avg_iterations, avg_cost = runner.analyze_results(results)
    
    # Save results
    results.to_csv('fcbs_experiment_results.csv', index=False)
    print("\nResults saved to 'fcbs_experiment_results.csv'")