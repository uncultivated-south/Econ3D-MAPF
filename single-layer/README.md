# Econ-MAPF

A comprehensive simulation system for multi-agent pathfinding in urban airspace environments, featuring algorithmic coordination (CBS), market-based coordination (auctions), and hybrid approaches.

## Overview

Econ-MAPF addresses the challenge of coordinating multiple agents (aircraft, drones) in shared airspace while considering both emergency priorities and economic factors. The system compares three coordination mechanisms:

- **CBS-only**: Pure algorithmic coordination using Conflict-Based Search
- **Auction-only**: Pure market-based coordination with competitive bidding
- **Hybrid**: CBS first, falling back to auctions when algorithmic approaches fail

## Architecture

### Core Modules

#### 1. Grid System (`grid_system.py`)
The foundational spatial-temporal representation of airspace:

- **3D Grid**: Represents airspace as (x, y, time) discrete space
- **Agent Management**: Handles emergency and non-emergency agents with priority-based coordination
- **Path Reservations**: Thread-safe atomic path operations with transaction semantics
- **Conflict Detection**: Real-time occupancy tracking and priority-based conflict resolution

Key features:
- Priority-based cell reservations (emergency agents override non-emergency)
- Atomic path updates with rollback capability
- Thread-safe operations for concurrent access
- Comprehensive validation and error handling

#### 2. A* Pathfinding (`astar_pathfinding.py`)
Enhanced pathfinding with constraint support:

- **Multiple Heuristics**: Manhattan, Euclidean, diagonal, and custom heuristic functions
- **Constraint Integration**: Vertex, edge, and temporal constraints from CBS
- **Performance Optimization**: Configurable search limits and caching
- **Priority-Aware Planning**: Respects agent priorities and existing reservations

Key features:
- Separate caches for different heuristic types (prevents cache collisions)
- Floating-point precision handling in node comparisons
- Comprehensive constraint validation and application
- Adaptive pathfinding configurations for different use cases

#### 3. Conflict-Based Search (`cbs_system.py`)
Multi-agent pathfinding with conflict resolution:

- **Complete Conflict Detection**: Vertex, edge, and following conflicts
- **Proper Constraint Implementation**: Actual constraint application to pathfinding
- **Performance Monitoring**: Detailed search statistics and bottleneck identification
- **Graceful Degradation**: Partial solutions when complete solutions aren't found

Key features:
- Fixed iteration limits (eliminates scaling bias)
- Comprehensive conflict taxonomy (vertex/edge/following)
- CBS-A* integration with proper constraint propagation
- Auction candidate identification for seamless hybrid operation

#### 4. Auction System (`auction_module.py`)
Market-based coordination with economic principles:

- **Precise Budget Management**: Decimal arithmetic prevents floating-point errors
- **Transaction Safety**: Reserve → compete → commit/release pattern
- **Dynamic Pricing**: Conflict density and market condition-based pricing
- **Comprehensive Fairness Analysis**: Multiple fairness metrics and wealth distribution tracking

Key features:
- Pareto distribution for realistic budget inequality
- Multi-round bidding with adaptive strategies
- CBS integration for conflict resolution among winners
- Complete audit trail for all financial transactions

### Integration Module (`test.py`)
Comprehensive experimental framework for system evaluation:

- **Controlled Experimentation**: Identical test scenarios across all algorithms
- **Statistical Analysis**: Multiple fairness metrics and performance comparisons  
- **Scalability Testing**: Systematic evaluation across different agent counts
- **Reproducible Results**: Seeded randomization for consistent comparisons

## Quick Start

### Basic Usage

```python
from grid_system import GridSystem, Agent, AgentType
from astar_pathfinding import AStarPathfinder, create_conservative_config
from cbs_system import create_cbs_solver
from auction_module import create_auction_system

# Create grid system
grid = GridSystem(20, 20, max_time=100)

# Add static obstacles
grid.add_static_obstacle(5, 5)
grid.add_static_obstacle(10, 10)

# Create agents
emergency_agent = Agent(
    id="emergency_1", 
    agent_type=AgentType.EMERGENCY,
    start=(0, 0), 
    goal=(19, 19),
    priority=100
)

regular_agent = Agent(
    id=1,
    agent_type=AgentType.NON_EMERGENCY,
    start=(1, 1),
    goal=(18, 18),
    budget=50.0,
    strategy="balanced"
)

# Add agents to grid
grid.add_agent(emergency_agent)
grid.add_agent(regular_agent)

# Run CBS
pathfinder = AStarPathfinder(grid, create_conservative_config())
cbs = create_cbs_solver(grid, pathfinder)
result = cbs.solve([emergency_agent, regular_agent])

if result.success:
    print(f"CBS found paths for {len(result.paths)} agents")
else:
    print("CBS failed - triggering auction")
    # Run auction for remaining agents
    auction = create_auction_system(grid, cbs, pathfinder)
    # ... auction logic
```

### Running Experiments

```python
from integration_test_module import ExperimentRunner

# Configure experiment
experiment = ExperimentRunner(base_seed=42)
experiment.agent_counts = [8, 16, 24, 32]
experiment.scenarios_per_count = 20

# Run complete experimental study
results = experiment.run_agent_count_experiment()

# Generate summary report
summary = experiment.generate_summary_report(results)
print("Key Findings:", summary['key_findings'])
```

## Experimental Design

### Test Parameters
- **Grid Size**: 20×20 spatial dimensions
- **Static Obstacles**: 12 obstacles per scenario
- **Regular Agents**: Variable count (4-32), subject to budget constraints
- **Budget Distribution**: Pareto distribution (α=1.16) in range [1, 150]
- **CBS Iteration Limit**: 4000 iterations before auction trigger

### Fairness Metrics
1. **Success Rate by Budget Quartile**: Measures wealth-based access inequality
2. **Gini Coefficient**: Measures inequality in success distribution
3. **Budget Correlation Coefficient**: Measures correlation between wealth and success

### Scenarios Tested
1. **CBS-only**: Pure algorithmic coordination
2. **Auction-only**: Pure market-based coordination (CBS only for pricing initialization)
3. **Hybrid**: CBS first with auction fallback (clean auction - no partial solution preservation)

## Algorithm Details

### CBS Implementation
- **Conflict Detection**: Comprehensive vertex/edge/following conflict identification
- **Constraint Generation**: Proper CBS constraint creation and application
- **Search Strategy**: Priority-based conflict resolution with early termination
- **Integration**: Seamless handoff to auction system when limits exceeded

### Auction Mechanism
- **Pricing Engine**: Dynamic pricing based on conflict density and market conditions
- **Bidding Strategies**: Conservative, aggressive, balanced, and adaptive strategies
- **Winner Selection**: Cell-by-cell highest bidder with complete path requirements
- **Conflict Resolution**: CBS validation of auction winners with priority ordering

### Hybrid Approach
- **Switching Criteria**: Fixed iteration limit (4000) for consistent comparison
- **Clean Transition**: Complete CBS solution discarded to ensure fair auction conditions
- **Pricing Initialization**: CBS conflict density used for auction starting prices

## Scalability
The system demonstrates different scaling characteristics:
- **CBS**: Excellent for low-conflict scenarios, degrades exponentially with conflict density
- **Auction**: Linear scaling with agent count, robust to high-conflict scenarios
- **Hybrid**: Best-case CBS performance with auction fallback protection

## Limitations and Future Work

### Current Limitations
1. **2D Spatial Representation**: No altitude dimension (can be extended)
2. **Discrete Time**: Continuous time not supported
3. **Static Obstacles**: Dynamic obstacles not implemented
4. **Weather Integration**: Environmental factors not considered

### Future Enhancements
1. **3D Airspace**: Full three-dimensional pathfinding
2. **Dynamic Constraints**: Weather, temporary flight restrictions
3. **Real-time Adaptation**: Online replanning capabilities
4. **Machine Learning**: Adaptive strategy selection based on historical performance

## Contributing

### Code Structure
The codebase follows a modular architecture with clear separation of concerns:
- Each module has comprehensive error handling and logging
- All financial operations use Decimal arithmetic for precision
- Thread-safe operations throughout with proper locking
- Extensive validation and testing infrastructure

### Testing
Run the integration tests to validate system behavior:
```python
python test.py
```

Tests cover:
- Individual module functionality
- Inter-module integration
- Performance characteristics
- Fairness metric calculations
- Edge cases and error conditions

## References

This implementation is based on established research in:
- Conflict-Based Search (Sharon et al., 2015)
- Multi-agent pathfinding algorithms
- Combinatorial auction theory
- Market-based multi-agent coordination