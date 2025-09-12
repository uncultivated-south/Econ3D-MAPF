# Econ-MAPF

A hybrid multi-agent pathfinding system that combines Conflict-Based Search (CBS) with auction-based mechanisms for urban airspace management.

## Overview

Econ-MAPF explores how algorithmic optimization can be seamlessly integrated with economic reasoning to solve complex multi-agent pathfinding problems. The system intelligently transitions between computational approaches (CBS) and market mechanisms (auctions) based on problem complexity and resource constraints.

### Key Features

- **Hybrid Approach**: Combines CBS efficiency with auction robustness
- **Priority-Based Processing**: Emergency agents receive absolute priority
- **Economic Mechanisms**: Market-driven conflict resolution with budget constraints
- **Scalable Architecture**: Handles varying problem sizes and complexities
- **Comprehensive Evaluation**: Built-in metrics for efficiency, fairness, and performance analysis

## System Architecture

### Four-Phase Processing Pipeline

1. **Emergency Processing**: Highest priority pathfinding for emergency agents
2. **CBS Feasibility Analysis**: Algorithmic solution attempt for regular agents
3. **Decision Point**: Intelligent transition logic between approaches
4. **Multi-Round Auction**: Economic allocation with conflict resolution

### Core Components

- **Grid System**: 3D airspace representation with dynamic obstacle management
- **A* Pathfinder**: Robust pathfinding with priority-based obstacle avoidance
- **CBS Module**: Conflict detection, resolution, and feasibility analysis
- **Auction System**: Multi-round bidding with strategy-based agents
- **Main Orchestrator**: Integrated workflow management and evaluation

## System Configuration

### Default Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| Grid Size | 20×20 | Spatial dimensions |
| Max Time | 100 | Maximum time steps |
| Emergency Agents | 2 | High-priority agents |
| Non-Emergency Agents | 16 | Regular agents |
| Budget Range | [1, 100] | Random budget allocation |
| Max CBS Iterations | 400 | (width × height) limit |
| Max Auction Rounds | 5 | Economic allocation rounds |

### Pricing Formula

```
StartingPrice = (AvgBudget × 0.1) × (ConflictDensity × 0.2) + 0.1
```

### Bidding Strategies

- **Conservative**: Minimum increments (10%), budget preservation
- **Aggressive**: Front-loaded bidding (150% early rounds), high risk/reward
- **Balanced**: Consistent increments (30%), equal round treatment

## Processing Modes

### Hybrid Mode (Recommended)
```python
result = sim.run_simulation(ProcessingMode.HYBRID)
```
- Attempts CBS first for computational efficiency
- Falls back to auction if CBS exceeds iteration limits
- Optimal balance of speed and robustness

### CBS Only Mode
```python
result = sim.run_simulation(ProcessingMode.CBS_ONLY)
```
- Pure algorithmic approach
- Fast when successful, fails on complex scenarios
- Useful for computational limit analysis

### Auction Only Mode
```python
result = sim.run_simulation(ProcessingMode.AUCTION_ONLY)
```
- Pure economic approach
- Robust but computationally intensive
- Useful for market mechanism analysis

## Key Metrics

### Performance Metrics
- **Success Rate**: Percentage of agents receiving paths
- **Computation Time**: Total processing time per simulation
- **CBS Efficiency**: Iterations used vs. maximum allowed
- **Auction Efficiency**: Revenue generated per round

### Fairness Metrics
- **Strategy Success Rates**: Performance by bidding strategy
- **Budget Utilization**: Economic efficiency measures
- **Emergency Priority**: Critical agent success rates

### Scalability Metrics
- **Agent Density**: Performance vs. agent count
- **Grid Utilization**: Spatial efficiency measures
- **Conflict Resolution**: Market vs. algorithmic effectiveness

## Technical Implementation

### Architecture Principles
- **Modular Design**: Independent, replaceable components
- **Clean Interfaces**: Standard data formats between modules
- **Extensibility**: Easy addition of new strategies and mechanisms
- **Performance Focus**: Efficient algorithms and data structures

### Data Flow
1. **Scenario Creation**: Agent configuration and grid initialization
2. **Emergency Processing**: Priority pathfinding with dynamic obstacles
3. **CBS Analysis**: Conflict detection and algorithmic resolution
4. **Economic Processing**: Market-based allocation and validation
5. **Result Integration**: Path assignment and performance measurement

### Key Classes and Functions

```python
# Core system components
GridSystem          # 3D airspace management
AStarPathfinder    # Individual agent pathfinding
ConflictBasedSearch # Multi-agent conflict resolution
AuctionSystem      # Economic allocation mechanism
UrbanAirspaceSim   # Main orchestration and evaluation

# Utility functions
create_simulation_system()    # System initialization
run_evaluation_study()       # Comparative analysis
demonstrate_system()         # Interactive demonstration
```

## Troubleshooting

### Common Issues

**CBS Timeout**: If CBS consistently exceeds iteration limits
```python
# Reduce problem complexity or increase limits
sim.cbs.max_iterations = 800  # Increase iteration limit
# Or use fewer agents in scenarios
```

**Auction Failures**: If no agents can afford paths
```python
# Adjust budget range or pricing parameters
sim.default_budget_range = (10, 200)  # Increase budget range
sim.auction.pricing_alpha = 0.05      # Reduce price sensitivity
```

**Memory Issues**: For large-scale simulations
```python
# Use smaller grids or fewer time steps
sim = UrbanAirspaceSim(15, 15, max_time=50)
```

## License

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

## Future Work

### Planned Features
- **Econ3D-MAPF**: Expanding to three-dimensional space
- **Machine Learning Integration**: Adaptive strategy learning
- **Distributed Processing**: Multi-core CBS and auction processing
- **Advanced Economic Models**: Dynamic pricing and complex auction formats

### Research Directions
- **Mechanism Design**: Optimal auction structures for airspace allocation
- **Game Theory**: Strategic behavior analysis in multi-agent systems
- **Robustness**: Performance under uncertainty and dynamic conditions
- **Scalability**: Techniques for very large-scale urban airspace management

---