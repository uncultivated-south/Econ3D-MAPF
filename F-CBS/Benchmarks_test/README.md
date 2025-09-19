# SIPP-based Free Energy Conflict-Based Search (F-CBS)

A high-performance multi-agent pathfinding algorithm that combines Safe Interval Path Planning (SIPP) with Free Energy optimization for improved solving efficiency.

## Overview

This implementation replaces the traditional A* pathfinding component in Conflict-Based Search (CBS) with Safe Interval Path Planning (SIPP), resulting in significant performance improvements for multi-agent pathfinding problems. The algorithm incorporates thermodynamic principles through Free Energy minimization to achieve better exploration of the solution space.

## Key Improvements

### Performance Enhancements
- **Faster Pathfinding**: SIPP handles temporal constraints more efficiently than constraint-table A*
- **Reduced Memory Usage**: Safe intervals are more memory-efficient than explicit constraint lookups
- **Better Scalability**: Improved performance with increasing numbers of constraints and agents

### Algorithmic Innovations
- **Safe Interval Path Planning (SIPP)**: Replaces A* with temporal interval-based pathfinding
- **Free Energy Optimization**: Balances solution cost with path congestion entropy
- **Temperature-based Control**: Configurable exploration vs exploitation trade-off
- **Simulated Annealing**: Adaptive temperature scheduling for better convergence

## Technical Details

### Safe Interval Path Planning (SIPP)

SIPP operates on a state space of `(location, safe_interval_index)` rather than `(location, time)`, enabling:

- **Efficient Constraint Handling**: O(1) interval lookups vs O(k) constraint checks
- **Temporal Awareness**: Heuristics consider interval availability
- **Optimal Pathfinding**: Maintains optimality guarantees of A*

### Free Energy Formulation

The algorithm minimizes Free Energy: **F = Cost - T × Entropy**

Where:
- **Cost**: Sum of individual path lengths (total travel distance)
- **Entropy**: Path congestion measure based on space-time location usage
- **Temperature (T)**: Controls exploration vs exploitation balance

### Entropy Calculation

Path congestion entropy is computed as:

```python
entropy = -∑(p_i × log₂(p_i))
```

Where `p_i` is the probability of agents occupying space-time location `i`.

## Usage

### Basic Example

```python
from sipp_fcbs_implementation import SIPP_FCBS, Grid, Agent

# Create environment
grid = Grid(width=20, height=20, obstacles={(5,5), (5,6), (6,5)})

# Define agents
agents = [
    Agent(0, start=(0,0), goal=(19,19)),
    Agent(1, start=(19,0), goal=(0,19)),
    Agent(2, start=(0,19), goal=(19,0))
]

# Solve with F-CBS
solver = SIPP_FCBS(grid, agents, temperature=2.0)
solution, iterations, solve_time = solver.solve()

if solution:
    total_cost = sum(len(path) - 1 for path in solution.values())
    print(f"Solution found! Cost: {total_cost}, Time: {solve_time:.3f}s")
```

### Benchmark Evaluation

```python
from sipp_fcbs_benchmark_runner import run_comprehensive_benchmark

# Run Moving AI Lab benchmarks
results = run_comprehensive_benchmark("path/to/benchmarks/")
```

## Algorithm Variants

### Standard SIPP-CBS
```python
solver = SIPP_CBS(grid, agents)
```

### Free Energy CBS with Fixed Temperature
```python
solver = SIPP_FCBS(grid, agents, temperature=5.0)
```

### Free Energy CBS with Annealing
```python
solver = SIPP_FCBS(grid, agents, temperature=10.0, 
                   using_annealing=True, annealing_iterations=20)
```

## Performance Results

Based on Moving AI Lab benchmark evaluation:

### Solving Speed
- **faster** individual pathfinding vs constraint-table A*
- **Better scaling** with agent count and constraint density

### Memory Usage
- **Improved cache locality** for temporal constraint access
- **Better performance** on memory-constrained systems

## Configuration Parameters

### Annealing Schedule
- **Initial Temperature**: High value for early exploration
- **Annealing Iterations**: Number of iterations before cooling
- **Final Temperature**: Usually 0.0 for pure cost minimization

## Benchmark Setup

### Download Moving AI Lab Benchmarks
1. Visit https://movingai.com/benchmarks/mapf.html
2. Download desired benchmark sets (empty, random, dao, etc.)
3. Extract to `benchmarks/` directory

## Running Benchmarks

### Quick Test
```bash
python sipp_fcbs_benchmark_runner.py
# Select option 1 for quick test (3 scenarios, 2min timeout)
```

### Comprehensive Evaluation
```bash
python sipp_fcbs_benchmark_runner.py
# Select option 2 for full evaluation (20+ scenarios, 10min timeout)
```

## Results Analysis

The benchmark runner automatically generates:
- **Performance metrics**: Success rates, solving times, solution quality
- **Scaling analysis**: Performance vs agent count and grid size
- **Temperature studies**: Optimal parameter selection
- **Visualization plots**: Comprehensive performance comparison charts
- **Detailed reports**: Text-based analysis with statistical summaries

## Research Applications

This implementation is suitable for research in:
- **Multi-agent pathfinding optimization**
- **Thermodynamic optimization in discrete systems**
- **Temporal constraint satisfaction**
- **Large-scale coordination algorithms**
- **Robotics and autonomous systems**

## Acknowledgments

- Moving AI Lab for providing comprehensive MAPF benchmarks
- Original CBS algorithm developers
- SIPP algorithm researchers
- Free Energy principle applications in discrete optimization