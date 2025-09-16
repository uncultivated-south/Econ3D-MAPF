# F-CBS: Free Energy Conflict-Based Search

A novel enhancement to the traditional Conflict-Based Search (CBS) algorithm for multi-agent pathfinding, incorporating thermodynamic principles to improve exploration and convergence behavior.

## Overview

F-CBS introduces the concept of **free energy** from thermodynamics into the CBS framework:

```
Free Energy = Internal Energy - Temperature × Entropy
F = U - T × S
```

Where:
- **Internal Energy (U)**: Traditional CBS cost (total path length)
- **Temperature (T)**: Adjustable parameter controlling exploration vs exploitation
- **Entropy (S)**: Information entropy based on conflict distribution across grid cells

## Key Innovation

Instead of purely minimizing path cost like traditional CBS, F-CBS uses free energy to guide search:

- **High Temperature**: Encourages exploration by prioritizing high-entropy nodes (distributed conflicts)
- **Low Temperature**: Focuses on exploitation by prioritizing low-cost solutions (concentrated conflicts)

This approach helps CBS escape local optima and find better solutions faster.

## Algorithm Details

### Entropy Calculation

The entropy is calculated based on conflict distribution:

1. Count total conflicts C across all grid cells
2. For each grid cell i, count conflicts c_i 
3. Calculate probability: p_i = c_i / C
4. Compute information entropy: H = -∑(p_i × log₂(p_i))

### Node Selection

- **Traditional CBS**: Selects node with minimum total cost
- **F-CBS**: Selects node with minimum free energy F = U - T×S

## Installation

```bash
pip install numpy matplotlib pandas heapq dataclasses
```

## Usage

### Basic Example

```python
from fcbs_comparison import ExperimentRunner, FCBS, CBS, Grid, Agent

# Create a scenario
grid = Grid(width=12, height=12, obstacles={(2,2), (5,5), (8,8)})
agents = [
    Agent(0, start=(0,0), goal=(10,10)),
    Agent(1, start=(11,11), goal=(1,1))
]

# Run F-CBS
fcbs = FCBS(grid, agents, temperature=0.5)
solution, iterations, time_taken = fcbs.solve()

# Run traditional CBS for comparison  
cbs = CBS(grid, agents)
cbs_solution, cbs_iterations, cbs_time = cbs.solve()
```

### Full Experiment

```python
# Run complete comparison experiment
runner = ExperimentRunner()
results = runner.run_experiment(num_scenarios=100)
success_rates, avg_iterations, avg_cost = runner.analyze_results(results)
```

## Experimental Setup

### Default Parameters
- **Grid Size**: 12×12
- **Agents**: 12 with random start/goal positions
- **Obstacles**: 16 randomly placed
- **Max Iterations**: 1024 (timeout threshold)
- **Test Temperatures**: [0.1, 0.2, 0.5, 1.0]
- **Test Scenarios**: 100 random configurations

### Metrics Tracked
- **Iterations to Solution**: Primary performance measure
- **Solution Quality**: Total path cost
- **Success Rate**: Percentage of problems solved within iteration limit
- **Computation Time**: Algorithm runtime

## Expected Results

### Temperature Effects
- **T = 0.1**: Conservative, similar to traditional CBS
- **T = 0.2-0.5**: Balanced exploration/exploitation (optimal range)
- **T = 1.0**: High exploration, potentially slower convergence

### Performance Gains
F-CBS typically shows:
- **Faster convergence** on complex multi-agent scenarios
- **Better escape** from local optima
- **Improved success rates** on difficult instances

## File Structure

```
├── fcbs_comparison.py      # Main implementation
├── README.md              # This documentation
└── results/
    ├── fcbs_experiment_results.csv    # Experiment output
    └── performance_plots.png          # Visualization
```

## Core Classes

### `FCBS(grid, agents, temperature)`
Enhanced CBS with free energy calculation
- Inherits from CBS class
- Adds entropy calculation and free energy node selection

### `CBS(grid, agents)`
Traditional Conflict-Based Search implementation
- Standard A* pathfinding with constraints
- Conflict detection and resolution

### `ExperimentRunner()`
Automated testing framework
- Generates random scenarios with consistent seeds
- Runs comparative analysis across algorithms
- Produces statistical summaries and visualizations

## Algorithm Comparison

| Feature | Traditional CBS | F-CBS |
|---------|----------------|-------|
| Node Selection | Min cost | Min free energy |
| Exploration | Limited | Temperature-controlled |
| Local Optima | Prone to getting stuck | Better escape mechanism |
| Parameters | None | Temperature T |
| Complexity | O(b^d) | O(b^d) + entropy calculation |

## Theoretical Foundation

F-CBS is inspired by:
- **Statistical Mechanics**: Free energy minimization principle
- **Simulated Annealing**: Temperature-based exploration control  
- **Information Theory**: Entropy as measure of uncertainty
- **Multi-Agent Pathfinding**: Conflict-based search framework

## Customization

### Custom Temperature Schedules
```python
# Implement adaptive temperature
class AdaptiveFCBS(FCBS):
    def update_temperature(self, iteration, max_iterations):
        # Linear cooling
        self.temperature = 1.0 * (1 - iteration/max_iterations)
        
    def solve(self, max_iterations=1024):
        # Override solve method to include temperature updates
        pass
```

### Custom Entropy Functions
```python  
def custom_entropy(conflicts):
    # Implement domain-specific entropy calculation
    # Example: weight conflicts by agent priority
    pass
```

## Performance Tips

1. **Temperature Tuning**: Start with T=0.5, adjust based on problem complexity
2. **Scenario Scaling**: Increase grid size and agent count gradually
3. **Constraint Handling**: Ensure proper constraint propagation for fair comparison
4. **Memory Management**: Monitor heap size for large problem instances

## Limitations

- **Parameter Sensitivity**: Requires temperature tuning for optimal performance
- **Computational Overhead**: Entropy calculation adds processing cost
- **Temperature Schedule**: Fixed temperature may not be optimal for all scenarios

## Future Enhancements

- **Adaptive Temperature**: Dynamic temperature adjustment during search
- **Multi-Objective**: Incorporate additional objectives beyond cost and entropy
- **Parallel Processing**: Distribute conflict resolution across multiple threads
- **Machine Learning**: Learn optimal temperature schedules from problem features