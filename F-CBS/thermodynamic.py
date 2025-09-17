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
class ThermodynamicTrace:
    """Store thermodynamic quantities during CBS search"""
    iteration: int
    entropy: float
    temperature: float
    free_energy: float
    cost: int
    entropy_production_rate: float
    conflicts_count: int
    
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

class FCBSNode:
    def __init__(self):
        self.vertex_constraints = []
        self.edge_constraints = []
        self.solution = {}
        self.cost = 0
        self.conflicts = []
        self.entropy = 0.0
        self.free_energy = 0.0
        
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
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x, y)]
        return [n for n in neighbors if self.is_valid(n)]

class PathFinder:
    def __init__(self, grid):
        self.grid = grid
    
    def a_star(self, start, goal, vertex_constraints=None, edge_constraints=None, max_iterations=10000, timeout_seconds=30):
        vertex_constraints = vertex_constraints or []
        edge_constraints = edge_constraints or []
        
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
            iteration += 1
            if iteration > max_iterations:
                return None
            
            if time.time() - start_time > timeout_seconds:
                return None
            
            _, cost, current, path = heapq.heappop(open_list)
            
            if current == goal:
                return path
            
            if (current, len(path)-1) in closed_set:
                continue
            closed_set.add((current, len(path)-1))
            
            for neighbor in self.grid.get_neighbors(current):
                timestep = len(path)
                
                if neighbor in vertex_constraint_table[timestep]:
                    continue
                    
                if (current, neighbor) in edge_constraint_table[timestep]:
                    continue
                
                new_path = path + [neighbor]
                new_cost = cost + 1
                priority = new_cost + self.heuristic(neighbor, goal)
                heapq.heappush(open_list, (priority, new_cost, neighbor, new_path))
        
        return None
    
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
                    
                    if t > 0:
                        prev_pos1 = path1[t-1] if t-1 < len(path1) else path1[-1]
                        prev_pos2 = path2[t-1] if t-1 < len(path2) else path2[-1]
                        
                        if pos1 == prev_pos2 and pos2 == prev_pos1 and pos1 != pos2:
                            conflicts.append({
                                'type': 'edge',
                                'agents': [agent1, agent2],
                                'from_locations': [prev_pos1, prev_pos2],
                                'to_locations': [pos1, pos2],
                                'timestep': t
                            })
        
        return conflicts

class ThermodynamicFCBS:
    def __init__(self, grid, agents, initial_temperature=-5.0, using_annealing=True, annealing_iterations=10):
        self.grid = grid
        self.agents = agents
        self.pathfinder = PathFinder(grid)
        self.conflict_detector = ConflictDetector()
        self.initial_temperature = initial_temperature
        self.current_temperature = initial_temperature
        self.use_annealing = using_annealing
        self.annealing_iterations = annealing_iterations
        
        # Thermodynamic tracking
        self.thermodynamic_trace = []
        self.previous_entropy = 0.0
    
    def update_temperature(self, iteration):
        if self.use_annealing and iteration > self.annealing_iterations:
            # Exponential annealing to zero
            decay_rate = 0.1
            remaining_iterations = iteration - self.annealing_iterations
            self.current_temperature = self.initial_temperature * math.exp(-decay_rate * remaining_iterations)
            
            # Clamp to zero when very small
            if abs(self.current_temperature) < 0.01:
                self.current_temperature = 0.0
        else:
            self.current_temperature = self.initial_temperature

    def calculate_entropy(self, conflicts):
        if not conflicts:
            return 0.0
        
        conflict_counts = defaultdict(float)
        total_conflict_weight = 0.0
        
        for conflict in conflicts:
            if conflict['type'] == 'vertex':
                conflict_counts[conflict['location']] += 1.0
                total_conflict_weight += 1.0
                
            elif conflict['type'] == 'edge':
                for to_loc in conflict['to_locations']:
                    conflict_counts[to_loc] += 0.5
                total_conflict_weight += 1.0
        
        if total_conflict_weight == 0.0:
            return 0.0
        
        entropy = 0.0
        for count in conflict_counts.values():
            if count > 0:
                p = count / total_conflict_weight
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_entropy_production_rate(self, current_entropy):
        """Calculate entropy production rate as change in entropy per iteration"""
        if len(self.thermodynamic_trace) == 0:
            return 0.0
        
        epr = current_entropy - self.previous_entropy
        return epr
    
    def generate_constraints_for_conflict(self, conflict, agent_id):
        constraints = {'vertex': [], 'edge': []}
        
        if conflict['type'] == 'vertex':
            constraints['vertex'].append(
                VertexConstraint(agent_id, conflict['location'], conflict['timestep'])
            )
        
        elif conflict['type'] == 'edge':
            agent_index = conflict['agents'].index(agent_id)
            from_loc = conflict['from_locations'][agent_index]
            to_loc = conflict['to_locations'][agent_index]
            
            constraints['edge'].append(
                EdgeConstraint(agent_id, from_loc, to_loc, conflict['timestep'])
            )
        
        return constraints
    
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
                return None, 0, time.time() - start_time, []
            root.solution[agent.id] = path
            root.cost += len(path) - 1
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        root.entropy = self.calculate_entropy(root.conflicts)
        root.free_energy = root.cost + self.current_temperature * root.entropy
        
        # Initialize thermodynamic tracking
        self.previous_entropy = root.entropy
        epr = self.calculate_entropy_production_rate(root.entropy)
        
        self.thermodynamic_trace.append(ThermodynamicTrace(
            iteration=0,
            entropy=root.entropy,
            temperature=self.current_temperature,
            free_energy=root.free_energy,
            cost=root.cost,
            entropy_production_rate=epr,
            conflicts_count=len(root.conflicts)
        ))
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            self.update_temperature(iterations)
            
            current = heapq.heappop(open_list)
            
            if not current.conflicts:
                return current.solution, iterations, time.time() - start_time, self.thermodynamic_trace
            
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
                    
                    # Track thermodynamics
                    epr = self.calculate_entropy_production_rate(new_node.entropy)
                    self.thermodynamic_trace.append(ThermodynamicTrace(
                        iteration=iterations,
                        entropy=new_node.entropy,
                        temperature=self.current_temperature,
                        free_energy=new_node.free_energy,
                        cost=new_node.cost,
                        entropy_production_rate=epr,
                        conflicts_count=len(new_node.conflicts)
                    ))
                    
                    self.previous_entropy = new_node.entropy
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time, self.thermodynamic_trace

class ThermodynamicAnalyzer:
    def __init__(self):
        pass
    
    def analyze_phase_transitions(self, trace_data):
        """Detect potential phase transitions in the thermodynamic trace"""
        if len(trace_data) < 5:
            return None
        
        iterations = [t.iteration for t in trace_data]
        epr_values = [t.entropy_production_rate for t in trace_data]
        entropy_values = [t.entropy for t in trace_data]
        
        # Look for sudden changes in EPR (potential phase transitions)
        epr_array = np.array(epr_values)
        epr_gradient = np.gradient(epr_array)
        
        # Find points where EPR changes significantly
        threshold = np.std(epr_gradient) * 2
        transition_points = []
        
        for i in range(1, len(epr_gradient)-1):
            if abs(epr_gradient[i]) > threshold:
                transition_points.append({
                    'iteration': iterations[i],
                    'epr': epr_values[i],
                    'entropy': entropy_values[i],
                    'epr_gradient': epr_gradient[i]
                })
        
        return transition_points
    
    def plot_thermodynamic_trajectory(self, successful_traces, failed_traces):
        """Plot thermodynamic trajectories for successful vs failed attempts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot successful traces
        for trace in successful_traces:
            iterations = [t.iteration for t in trace]
            entropies = [t.entropy for t in trace]
            eprs = [t.entropy_production_rate for t in trace]
            temperatures = [t.temperature for t in trace]
            free_energies = [t.free_energy for t in trace]
            costs = [t.cost for t in trace]
            
            axes[0, 0].plot(iterations, entropies, 'g-', alpha=0.6, linewidth=1)
            axes[0, 1].plot(iterations, eprs, 'g-', alpha=0.6, linewidth=1)
            axes[0, 2].plot(iterations, free_energies, 'g-', alpha=0.6, linewidth=1)
        
        # Plot failed traces
        for trace in failed_traces:
            iterations = [t.iteration for t in trace]
            entropies = [t.entropy for t in trace]
            eprs = [t.entropy_production_rate for t in trace]
            free_energies = [t.free_energy for t in trace]
            
            axes[1, 0].plot(iterations, entropies, 'r-', alpha=0.6, linewidth=1)
            axes[1, 1].plot(iterations, eprs, 'r-', alpha=0.6, linewidth=1)
            axes[1, 2].plot(iterations, free_energies, 'r-', alpha=0.6, linewidth=1)
        
        # Set titles and labels
        axes[0, 0].set_title('Entropy - Successful Cases')
        axes[0, 1].set_title('Entropy Production Rate - Successful Cases')
        axes[0, 2].set_title('Free Energy - Successful Cases')
        axes[1, 0].set_title('Entropy - Failed Cases')
        axes[1, 1].set_title('Entropy Production Rate - Failed Cases')
        axes[1, 2].set_title('Free Energy - Failed Cases')
        
        for ax in axes.flat:
            ax.set_xlabel('CBS Iteration')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def detect_phase_transition_signatures(self, traces):
        """Look for signatures of phase transitions across multiple runs"""
        all_transition_points = []
        
        for trace in traces:
            transitions = self.analyze_phase_transitions(trace)
            if transitions:
                all_transition_points.extend(transitions)
        
        if not all_transition_points:
            return None
        
        # Analyze transition statistics
        transition_iterations = [tp['iteration'] for tp in all_transition_points]
        transition_eprs = [tp['epr'] for tp in all_transition_points]
        
        return {
            'total_transitions': len(all_transition_points),
            'avg_transition_iteration': np.mean(transition_iterations),
            'std_transition_iteration': np.std(transition_iterations),
            'avg_transition_epr': np.mean(transition_eprs),
            'transition_points': all_transition_points
        }

# Example usage for thermodynamic analysis
def run_thermodynamic_experiment():
    """Run experiment focusing on thermodynamic analysis"""
    
    # Generate a test scenario
    random.seed(42)
    np.random.seed(42)
    
    grid = Grid(10, 10, obstacles={(3,3), (3,4), (4,3), (4,4), (6,6), (6,7)})
    agents = [
        Agent(0, (0, 0), (9, 9)),
        Agent(1, (0, 9), (9, 0)),
        Agent(2, (5, 0), (5, 9)),
        Agent(3, (0, 5), (9, 5))
    ]
    
    successful_traces = []
    failed_traces = []
    
    # Test multiple temperature values
    temperatures = [-10.0, -5.0, -2.0, -1.0]
    
    for temp in temperatures:
        print(f"Testing temperature {temp}")
        
        fcbs = ThermodynamicFCBS(grid, agents, initial_temperature=temp, 
                                using_annealing=True, annealing_iterations=5)
        solution, iterations, solve_time, trace = fcbs.solve(max_iterations=200)
        
        if solution:
            successful_traces.append(trace)
            print(f"  SUCCESS: {iterations} iterations, final entropy: {trace[-1].entropy:.3f}")
        else:
            failed_traces.append(trace)
            print(f"  FAILED: {iterations} iterations, final entropy: {trace[-1].entropy:.3f}")
    
    # Analyze results
    analyzer = ThermodynamicAnalyzer()
    
    print(f"\nAnalyzing {len(successful_traces)} successful and {len(failed_traces)} failed traces...")
    
    # Plot trajectories
    analyzer.plot_thermodynamic_trajectory(successful_traces, failed_traces)
    
    # Detect phase transitions
    successful_transitions = analyzer.detect_phase_transition_signatures(successful_traces)
    failed_transitions = analyzer.detect_phase_transition_signatures(failed_traces)
    
    print("\n=== PHASE TRANSITION ANALYSIS ===")
    if successful_transitions:
        print(f"Successful cases: {successful_transitions['total_transitions']} transitions detected")
        print(f"  Average transition at iteration: {successful_transitions['avg_transition_iteration']:.1f}")
        print(f"  Average EPR at transition: {successful_transitions['avg_transition_epr']:.3f}")
    
    if failed_transitions:
        print(f"Failed cases: {failed_transitions['total_transitions']} transitions detected")
        print(f"  Average transition at iteration: {failed_transitions['avg_transition_iteration']:.1f}")
        print(f"  Average EPR at transition: {failed_transitions['avg_transition_epr']:.3f}")

if __name__ == "__main__":
    run_thermodynamic_experiment()