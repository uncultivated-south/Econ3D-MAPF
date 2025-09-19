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
class AdaptiveConfig:
    """Configuration for adaptive neighbor selection"""
    base_range: int = 1  # Base movement range (1 = 4-connected, 2 = extended)
    max_range: int = 3   # Maximum movement range
    congestion_threshold: float = 0.5  # Threshold for congestion-based expansion
    conflict_expansion_factor: int = 1  # How much to expand when conflicts detected
    goal_proximity_threshold: int = 5   # Distance to goal for range expansion
    use_diagonal: bool = True           # Whether to allow diagonal movements
    use_large_steps: bool = True        # Whether to allow multi-step moves
    dynamic_waiting: bool = True        # Whether to adaptively allow waiting

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

@dataclass
class SafeInterval:
    start_time: int
    end_time: int  # exclusive

class SIPPNode:
    def __init__(self, location, interval_idx, g_score, h_score, parent=None, arrival_time=0):
        self.location = location
        self.interval_idx = interval_idx  # Index of the safe interval being used
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = g_score + h_score
        self.parent = parent
        self.arrival_time = arrival_time
    
    def __lt__(self, other):
        return self.f_score < other.f_score

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

class AdaptiveGrid:
    def __init__(self, width=12, height=12, obstacles=None, adaptive_config=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        
        # Congestion tracking
        self.congestion_map = defaultdict(float)
        self.conflict_history = defaultdict(int)
        self.agent_densities = defaultdict(float)
        
    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and pos not in self.obstacles
    
    def update_congestion(self, paths, current_time):
        """Update congestion information based on current paths"""
        self.congestion_map.clear()
        
        # Count agents at each position at current time
        position_counts = defaultdict(int)
        for agent_id, path in paths.items():
            if current_time < len(path):
                pos = path[current_time]
                position_counts[pos] += 1
        
        # Calculate congestion values
        for pos, count in position_counts.items():
            if count > 1:
                self.congestion_map[pos] = count / len(paths)
    
    def calculate_local_density(self, center_pos, radius=2):
        """Calculate agent density around a position"""
        x_center, y_center = center_pos
        agent_count = 0
        total_positions = 0
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                pos = (x_center + dx, y_center + dy)
                if self.is_valid(pos):
                    total_positions += 1
                    if pos in self.congestion_map:
                        agent_count += self.congestion_map[pos]
        
        return agent_count / max(total_positions, 1)
    
    def get_adaptive_neighbors(self, pos, goal, current_time=0, agent_id=0, context=None):
        """Get neighbors with adaptive range based on multiple factors"""
        context = context or {}
        
        # Determine adaptive range
        movement_range = self._calculate_adaptive_range(pos, goal, current_time, agent_id, context)
        
        neighbors = []
        x, y = pos
        
        # Generate neighbors based on movement range and configuration
        for dx in range(-movement_range, movement_range + 1):
            for dy in range(-movement_range, movement_range + 1):
                new_pos = (x + dx, y + dy)
                
                # Skip invalid positions
                if not self.is_valid(new_pos):
                    continue
                
                # Apply movement constraints
                if self._is_valid_move(pos, new_pos, movement_range):
                    neighbors.append(new_pos)
        
        # Add adaptive waiting behavior
        if self.adaptive_config.dynamic_waiting:
            if self._should_allow_waiting(pos, goal, current_time, context):
                neighbors.append(pos)  # Stay in place
        
        return neighbors
    
    def _calculate_adaptive_range(self, pos, goal, current_time, agent_id, context):
        """Calculate adaptive movement range based on multiple factors"""
        base_range = self.adaptive_config.base_range
        max_range = self.adaptive_config.max_range
        
        # Factor 1: Congestion-based expansion
        local_density = self.calculate_local_density(pos)
        congestion_expansion = 0
        if local_density > self.adaptive_config.congestion_threshold:
            congestion_expansion = min(1, int(local_density * 3))
        
        # Factor 2: Goal proximity expansion
        goal_distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        proximity_expansion = 0
        if goal_distance <= self.adaptive_config.goal_proximity_threshold:
            proximity_expansion = 1
        
        # Factor 3: Conflict history expansion
        conflict_expansion = 0
        if pos in self.conflict_history and self.conflict_history[pos] > 20:
            conflict_expansion = self.adaptive_config.conflict_expansion_factor
        
        # Factor 4: Time pressure (late in planning)
        time_pressure_expansion = 0
        if current_time > 50:  # Arbitrary threshold for "late" planning
            time_pressure_expansion = 1
        
        # Factor 5: Open space detection
        open_space_expansion = self._detect_open_space_expansion(pos)
        
        # Combine all factors
        total_expansion = (congestion_expansion + proximity_expansion + 
                          conflict_expansion + time_pressure_expansion + 
                          open_space_expansion)
        
        adaptive_range = min(1, max_range, base_range + total_expansion)
        
        return max(1, adaptive_range)  # Ensure at least base connectivity
    
    def _detect_open_space_expansion(self, pos):
        """Detect if agent is in open space and can benefit from larger moves"""
        x, y = pos
        open_radius = 3
        
        # Count open spaces in a radius
        open_count = 0
        total_count = 0
        
        for dx in range(-open_radius, open_radius + 1):
            for dy in range(-open_radius, open_radius + 1):
                test_pos = (x + dx, y + dy)
                if 0 <= test_pos[0] < self.width and 0 <= test_pos[1] < self.height:
                    total_count += 1
                    if test_pos not in self.obstacles:
                        open_count += 1
        
        open_ratio = open_count / max(total_count, 1)
        
        # If in very open area, allow larger steps
        if open_ratio > 0.8:
            return 2
        elif open_ratio > 0.6:
            return 1
        else:
            return 0
    
    def _is_valid_move(self, from_pos, to_pos, movement_range):
        """Check if a move is valid given the movement constraints"""
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        # Basic range check
        if max(dx, dy) > movement_range:
            return False
        
        # Diagonal movement check
        if not self.adaptive_config.use_diagonal and dx > 0 and dy > 0:
            return False
        
        # Large step validation
        if not self.adaptive_config.use_large_steps and (dx > 1 or dy > 1):
            return False
        
        # Knight's move or irregular patterns could be added here
        
        return True
    
    def _should_allow_waiting(self, pos, goal, current_time, context):
        """Determine if waiting should be allowed at current position"""
        if not self.adaptive_config.dynamic_waiting:
            return False
        
        # Always allow waiting if congested
        if self.calculate_local_density(pos) > 0.3:
            return True
        
        # Allow waiting if close to goal (for timing)
        goal_distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        if goal_distance <= 3:
            return True
        
        # Strategic waiting based on context
        if context.get('conflicts_nearby', False):
            return True
        
        return False
    
    def record_conflict(self, pos):
        """Record a conflict at a position for future adaptive decisions"""
        self.conflict_history[pos] += 1

class SafeIntervalTable:
    def __init__(self, grid, max_time=500):
        self.grid = grid
        self.max_time = max_time
        # Default: all locations are safe for all time
        self.safe_intervals = {}
        for x in range(grid.width):
            for y in range(grid.height):
                if grid.is_valid((x, y)):
                    self.safe_intervals[(x, y)] = [SafeInterval(0, max_time)]
    
    def add_vertex_constraint(self, constraint: VertexConstraint):
        """Block a specific location at a specific timestep"""
        location = constraint.location
        timestep = constraint.timestep
        
        if location not in self.safe_intervals:
            return
        
        new_intervals = []
        for interval in self.safe_intervals[location]:
            if timestep >= interval.end_time or timestep < interval.start_time:
                # Constraint doesn't affect this interval
                new_intervals.append(interval)
            elif timestep == interval.start_time:
                # Block the start of interval
                if timestep + 1 < interval.end_time:
                    new_intervals.append(SafeInterval(timestep + 1, interval.end_time))
            elif timestep == interval.end_time - 1:
                # Block the end of interval
                if interval.start_time < timestep:
                    new_intervals.append(SafeInterval(interval.start_time, timestep))
            else:
                # Block middle of interval - split it
                if interval.start_time < timestep:
                    new_intervals.append(SafeInterval(interval.start_time, timestep))
                if timestep + 1 < interval.end_time:
                    new_intervals.append(SafeInterval(timestep + 1, interval.end_time))
        
        self.safe_intervals[location] = new_intervals
    
    def add_edge_constraint(self, constraint: EdgeConstraint):
        """Block a specific edge movement at a specific timestep"""
        from_loc = constraint.from_location
        to_loc = constraint.to_location
        timestep = constraint.timestep
        
        # Block the destination at the timestep
        temp_vertex_constraint = VertexConstraint(constraint.agent, to_loc, timestep)
        self.add_vertex_constraint(temp_vertex_constraint)
    
    def get_safe_intervals(self, location):
        """Get safe intervals for a location"""
        return self.safe_intervals.get(location, [])
    
    def is_safe(self, location, time):
        """Check if location is safe at given time"""
        intervals = self.get_safe_intervals(location)
        for interval in intervals:
            if interval.start_time <= time < interval.end_time:
                return True
        return False

class AdaptiveSIPPPathFinder:
    def __init__(self, grid):
        self.grid = grid
    
    def find_path(self, start, goal, vertex_constraints=None, edge_constraints=None, max_time=500, agent_id=0, context=None):
        """SIPP pathfinding with adaptive neighbor selection"""
        vertex_constraints = vertex_constraints or []
        edge_constraints = edge_constraints or []
        context = context or {}
        
        # Build safe interval table
        safe_intervals = SafeIntervalTable(self.grid, max_time)
        
        # Apply constraints
        for constraint in vertex_constraints:
            safe_intervals.add_vertex_constraint(constraint)
        for constraint in edge_constraints:
            safe_intervals.add_edge_constraint(constraint)
        
        return self._adaptive_sipp_search(start, goal, safe_intervals, agent_id, context)
    
    def _adaptive_sipp_search(self, start, goal, safe_intervals, agent_id, context):
        """Core SIPP algorithm with adaptive neighbor selection"""
        if start == goal:
            return [start]
        
        # Priority queue: (f_score, node)
        open_list = []
        closed_set = set()  # (location, interval_idx)
        
        # Initialize with start location
        start_intervals = safe_intervals.get_safe_intervals(start)
        if not start_intervals:
            return None
        
        # Find the first safe interval at start
        earliest_interval_idx = None
        for i, interval in enumerate(start_intervals):
            if interval.start_time <= 0 < interval.end_time:
                earliest_interval_idx = i
                break
        
        if earliest_interval_idx is None:
            return None
        
        start_node = SIPPNode(
            location=start,
            interval_idx=earliest_interval_idx,
            g_score=0,
            h_score=self._adaptive_heuristic(start, goal, safe_intervals),
            arrival_time=0
        )
        
        heapq.heappush(open_list, (start_node.f_score, start_node))
        
        iteration = 0
        max_iterations = 100000
        
        while open_list and iteration < max_iterations:
            iteration += 1
            _, current = heapq.heappop(open_list)
            
            if current.location == goal:
                return self._reconstruct_adaptive_path(current)
            
            state_key = (current.location, current.interval_idx)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            # Expand neighbors with adaptive selection
            self._expand_adaptive_neighbors(current, goal, safe_intervals, open_list, 
                                          closed_set, agent_id, context)
        
        return None
    
    def _expand_adaptive_neighbors(self, current_node, goal, safe_intervals, 
                                 open_list, closed_set, agent_id, context):
        """Expand neighbors using adaptive neighbor selection"""
        # Get adaptive neighbors based on current situation
        adaptive_neighbors = self.grid.get_adaptive_neighbors(
            current_node.location, goal, current_node.arrival_time, agent_id, context
        )
        
        current_interval = safe_intervals.get_safe_intervals(current_node.location)[current_node.interval_idx]
        
        for neighbor_pos in adaptive_neighbors:
            neighbor_intervals = safe_intervals.get_safe_intervals(neighbor_pos)
            if not neighbor_intervals:
                continue
            
            # Calculate movement cost (adaptive based on distance)
            move_cost = self._calculate_move_cost(current_node.location, neighbor_pos)
            earliest_arrival = current_node.arrival_time + move_cost
            
            # Find compatible safe intervals at neighbor
            for interval_idx, interval in enumerate(neighbor_intervals):
                if earliest_arrival < interval.end_time and interval.start_time < current_interval.end_time:
                    arrival_time = max(earliest_arrival, interval.start_time)
                    
                    if arrival_time >= current_interval.end_time:
                        continue
                    
                    state_key = (neighbor_pos, interval_idx)
                    if state_key in closed_set:
                        continue
                    
                    g_score = arrival_time
                    h_score = self._adaptive_heuristic(neighbor_pos, goal, safe_intervals)
                    
                    neighbor_node = SIPPNode(
                        location=neighbor_pos,
                        interval_idx=interval_idx,
                        g_score=g_score,
                        h_score=h_score,
                        parent=current_node,
                        arrival_time=arrival_time
                    )
                    
                    heapq.heappush(open_list, (neighbor_node.f_score, neighbor_node))
    
    def _calculate_move_cost(self, from_pos, to_pos):
        """Calculate cost of movement between positions"""
        if from_pos == to_pos:
            return 1  # Waiting cost
        
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        # Manhattan distance for orthogonal moves
        if dx == 0 or dy == 0:
            return max(dx, dy)
        
        # Diagonal moves
        if dx == dy:
            return dx  # Diagonal distance
        
        # Complex moves (like knight's moves or large steps)
        return max(dx, dy)  # Chebyshev distance
    
    def _adaptive_heuristic(self, pos, goal, safe_intervals):
        """Adaptive heuristic that considers congestion and available moves"""
        # Base Manhattan distance
        base_heuristic = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # Congestion penalty
        local_density = self.grid.calculate_local_density(pos)
        congestion_penalty = local_density * 5
        
        # Goal area congestion
        goal_intervals = safe_intervals.get_safe_intervals(goal)
        if goal_intervals:
            total_safe_time = sum(interval.end_time - interval.start_time for interval in goal_intervals)
            interval_penalty = max(0, 10 - total_safe_time / 50)
        else:
            interval_penalty = 100
        
        # Adaptive range bonus (if we can take larger steps, reduce heuristic)
        adaptive_range = self.grid._calculate_adaptive_range(pos, goal, 0, 0, {})
        range_bonus = -min(2, adaptive_range - 1)
        
        return base_heuristic + congestion_penalty + interval_penalty + range_bonus
    
    def _reconstruct_adaptive_path(self, node):
        """Reconstruct path handling variable move costs"""
        path = []
        times = []
        current = node
        
        while current is not None:
            path.append(current.location)
            times.append(current.arrival_time)
            current = current.parent
        
        path.reverse()
        times.reverse()
        
        # Create time-accurate path
        if not path:
            return []
        
        time_accurate_path = []
        for i, (pos, time) in enumerate(zip(path, times)):
            if i == 0:
                # Fill from time 0 to first arrival
                for t in range(int(time) + 1):
                    time_accurate_path.append(pos)
            else:
                prev_time = int(times[i-1])
                curr_time = int(time)
                # Fill intermediate times
                for t in range(prev_time + 1, curr_time + 1):
                    time_accurate_path.append(pos)
        
        return time_accurate_path

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

class AdaptiveSIPP_CBS:
    def __init__(self, grid, agents, adaptive_config=None):
        # Convert regular grid to adaptive grid
        if not isinstance(grid, AdaptiveGrid):
            self.grid = AdaptiveGrid(grid.width, grid.height, grid.obstacles, adaptive_config)
        else:
            self.grid = grid
            
        self.agents = agents
        self.pathfinder = AdaptiveSIPPPathFinder(self.grid)
        self.conflict_detector = ConflictDetector()
    
    def solve(self, max_iterations=1000000):
        start_time = time.time()
        root = CBSNode()
        
        # Find initial solution using adaptive SIPP
        context = {'iteration': 0, 'conflicts_nearby': False}
        
        for agent in self.agents:
            path = self.pathfinder.find_path(
                agent.start, agent.goal,
                root.vertex_constraints, root.edge_constraints,
                agent_id=agent.id, context=context
            )
            if path is None:
                return None, 0, time.time() - start_time
            root.solution[agent.id] = path
            root.cost += len(path) - 1
        
        # Update grid congestion information
        self.grid.update_congestion(root.solution, 0)
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        
        open_list = [root]
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_list)
            
            if not current.conflicts:
                return current.solution, iterations, time.time() - start_time
            
            conflict = current.conflicts[0]
            
            # Record conflict for adaptive learning
            if conflict['type'] == 'vertex':
                self.grid.record_conflict(conflict['location'])
            
            for agent_id in conflict['agents']:
                new_node = CBSNode()
                new_node.vertex_constraints = current.vertex_constraints.copy()
                new_node.edge_constraints = current.edge_constraints.copy()
                
                # Generate constraints
                new_constraints = self._generate_constraints_for_conflict(conflict, agent_id)
                new_node.vertex_constraints.extend(new_constraints['vertex'])
                new_node.edge_constraints.extend(new_constraints['edge'])
                
                # Update context for adaptive pathfinding
                context = {
                    'iteration': iterations,
                    'conflicts_nearby': True,
                    'conflict_count': len(current.conflicts)
                }
                
                # Update congestion info
                self.grid.update_congestion(current.solution, iterations)
                
                # Replan with adaptive pathfinding
                agent = next(a for a in self.agents if a.id == agent_id)
                new_path = self.pathfinder.find_path(
                    agent.start, agent.goal,
                    new_node.vertex_constraints, new_node.edge_constraints,
                    agent_id=agent_id, context=context
                )
                
                if new_path is not None:
                    new_node.solution = current.solution.copy()
                    new_node.solution[agent_id] = new_path
                    new_node.cost = sum(len(path) - 1 for path in new_node.solution.values())
                    new_node.conflicts = self.conflict_detector.find_conflicts(new_node.solution)
                    
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time
    
    def _generate_constraints_for_conflict(self, conflict, agent_id):
        """Generate appropriate constraints based on conflict type"""
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
    
    def calculate_path_congestion_entropy(self, paths):
        """Calculate entropy based on space-time congestion"""
        if not paths:
            return 0.0
        
        spacetime_usage = defaultdict(int)
        total_usage = 0
        
        for agent_id, path in paths.items():
            for t, pos in enumerate(path):
                spacetime_usage[(pos, t)] += 1
                total_usage += 1
        
        if total_usage == 0:
            return 0.0
        
        # Calculate entropy from congestion distribution
        entropy = 0.0
        for usage_count in spacetime_usage.values():
            p = usage_count / total_usage
            entropy -= p * math.log2(p)
        
        return entropy

class SIPP_FCBS(AdaptiveSIPP_CBS):
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
    
    def solve(self, max_iterations=1000000):
        start_time = time.time()
        root = FCBSNode()
        
        # Find initial solution using SIPP
        for agent in self.agents:
            path = self.pathfinder.find_path(
                agent.start, agent.goal,
                root.vertex_constraints, root.edge_constraints
            )
            if path is None:
                return None, 0, time.time() - start_time
            root.solution[agent.id] = path
            root.cost += len(path) - 1  # Sum of individual path lengths
        
        root.conflicts = self.conflict_detector.find_conflicts(root.solution)
        root.entropy = self.calculate_path_congestion_entropy(root.solution)
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
                new_constraints = self._generate_constraints_for_conflict(conflict, agent_id)
                new_node.vertex_constraints.extend(new_constraints['vertex'])
                new_node.edge_constraints.extend(new_constraints['edge'])
                
                # Replan for constrained agent using SIPP
                agent = next(a for a in self.agents if a.id == agent_id)
                new_path = self.pathfinder.find_path(
                    agent.start, agent.goal,
                    new_node.vertex_constraints, new_node.edge_constraints
                )
                
                if new_path is not None:
                    new_node.solution = current.solution.copy()
                    new_node.solution[agent_id] = new_path
                    new_node.cost = sum(len(path) - 1 for path in new_node.solution.values())
                    new_node.conflicts = self.conflict_detector.find_conflicts(new_node.solution)
                    new_node.entropy = self.calculate_path_congestion_entropy(new_node.solution)
                    # Use current temperature for free energy calculation
                    new_node.free_energy = new_node.cost + self.current_temperature * new_node.entropy
                    
                    heapq.heappush(open_list, new_node)
        
        return None, iterations, time.time() - start_time

# Integration with existing experiment runners
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
        
        return AdaptiveGrid(width, height, obstacles)
    
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

# Example usage
def test_sipp_vs_astar(map_file, scen_file, num_agents=5):
    """Compare SIPP-CBS with original A*-CBS implementation"""
    print(f"=== SIPP vs A* Comparison ===")
    print(f"Map: {map_file}")
    print(f"Scenario: {scen_file}")
    print(f"Agents: {num_agents}")
    
    loader = MovingAILoader()
    
    try:
        # Load the benchmark
        grid = loader.load_map(map_file)
        agents = loader.load_scenario(scen_file, num_agents=num_agents)
        
        print(f"Loaded {len(agents)} agents on {grid.width}x{grid.height} grid")
        
        # Test SIPP-CBS
        print("\nTesting SIPP-CBS...")
        sipp_cbs = AdaptiveSIPP_CBS(grid, agents)
        sipp_solution, sipp_iterations, sipp_time = sipp_cbs.solve()
        
        if sipp_solution:
            sipp_cost = sum(len(path) - 1 for path in sipp_solution.values())
            print(f"SIPP-CBS: SUCCESS - Cost: {sipp_cost}, Iterations: {sipp_iterations}, Time: {sipp_time:.3f}s")
        else:
            print(f"SIPP-CBS: FAILED - Iterations: {sipp_iterations}, Time: {sipp_time:.3f}s")
        
        # Test SIPP-F-CBS
        print("\nTesting SIPP-F-CBS (T=5.0)...")
        sipp_fcbs = SIPP_FCBS(grid, agents, temperature=5.0, using_annealing=False)
        fcbs_solution, fcbs_iterations, fcbs_time = sipp_fcbs.solve()
        
        if fcbs_solution:
            fcbs_cost = sum(len(path) - 1 for path in fcbs_solution.values())
            entropy = sipp_fcbs.calculate_path_congestion_entropy(fcbs_solution)
            free_energy = fcbs_cost + 5.0 * entropy
            print(f"SIPP-F-CBS: SUCCESS - Cost: {fcbs_cost}, Entropy: {entropy:.3f}, Free Energy: {free_energy:.3f}")
            print(f"             Iterations: {fcbs_iterations}, Time: {fcbs_time:.3f}s")
        else:
            print(f"SIPP-F-CBS: FAILED - Iterations: {fcbs_iterations}, Time: {fcbs_time:.3f}s")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test with simple scenario first
    print("Testing SIPP implementation...")
    
    # Create a simple test scenario
    grid = AdaptiveGrid(8, 8, obstacles={(3, 3), (3, 4), (4, 3), (4, 4)})
    agents = [
        Agent(0, (0, 0), (7, 7)),
        Agent(1, (7, 0), (0, 7)),
        Agent(2, (0, 7), (7, 0))
    ]
    
    # Test SIPP-CBS
    sipp_cbs = AdaptiveSIPP_CBS(grid, agents)
    solution, iterations, solve_time = sipp_cbs.solve()
    
    if solution:
        total_cost = sum(len(path) - 1 for path in solution.values())
        entropy = sipp_cbs.calculate_path_congestion_entropy(solution)
        print(f"SIPP-CBS Solution found!")
        print(f"Total cost: {total_cost}")
        print(f"Congestion entropy: {entropy:.3f}")
        print(f"Iterations: {iterations}")
        print(f"Time: {solve_time:.3f}s")
        
        # Test F-CBS version
        sipp_fcbs = SIPP_FCBS(grid, agents, temperature=2.0)
        fcbs_solution, fcbs_iterations, fcbs_time = sipp_fcbs.solve()
        
        if fcbs_solution:
            fcbs_cost = sum(len(path) - 1 for path in fcbs_solution.values())
            fcbs_entropy = sipp_fcbs.calculate_path_congestion_entropy(fcbs_solution)
            free_energy = fcbs_cost + 2.0 * fcbs_entropy
            print(f"\nSIPP-F-CBS Solution found!")
            print(f"Total cost: {fcbs_cost}")
            print(f"Congestion entropy: {fcbs_entropy:.3f}")
            print(f"Free energy: {free_energy:.3f}")
            print(f"Iterations: {fcbs_iterations}")
            print(f"Time: {fcbs_time:.3f}s")
    else:
        print("No solution found")