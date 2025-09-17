# F-CBS: Free Energy Conflict-Based Search

A thermodynamic approach to multi-agent pathfinding that treats Conflict-Based Search (CBS) as a non-equilibrium dynamical system.

## Overview

This project introduces **F-CBS (Free Energy Conflict-Based Search)**, a novel algorithm that enhances traditional CBS by incorporating principles from statistical mechanics and thermodynamics. By treating the CBS search process as an open dynamical system with controllable "temperature," F-CBS can find solutions to multi-agent pathfinding problems that traditional CBS cannot solve.

## Key Concepts

### Free Energy Formulation
```
F = Cost - T × Entropy
```

Where:
- **Cost**: Total path length (traditional CBS objective)
- **T**: System temperature (negative values promote ordering)
- **Entropy**: Information entropy of conflict distribution across the grid

### Negative Temperature Dynamics
Unlike traditional thermodynamic systems, F-CBS uses **negative temperatures** to promote ordering:
- **T < 0**: System seeks low-entropy (organized) configurations
- **Annealing**: T → 0 gradually transitions from exploration to exploitation
- **Temperature as control**: Energy injection/extraction controls search dynamics

### Thermodynamic Metrics

The system tracks several key thermodynamic quantities:

1. **Entropy Production Rate (EPR)**: `dS/dt` - measures conflict pattern changes
2. **Energy Dissipation Rate (EDR)**: `-dF/dt` - measures approach to equilibrium
3. **Phase Transitions**: Sudden changes in EPR/EDR indicating qualitative behavioral shifts

## Features

### Multi-Agent Pathfinding
- **Grid-based environments** with customizable obstacles
- **Vertex conflicts**: Multiple agents at same location/time
- **Edge conflicts**: Agents swapping positions between timesteps
- **Constraint propagation**: Systematic conflict resolution

### Thermodynamic Analysis
- **Real-time tracking** of entropy, free energy, EPR, and EDR
- **Phase transition detection** via gradient analysis
- **Success vs failure pattern analysis**
- **Temperature-dependent behavior characterization**

### Experimental Framework
- **Comprehensive benchmarking** against traditional CBS
- **Multiple temperature regimes** and annealing schedules
- **Statistical analysis** across randomized scenarios
- **Visualization tools** for thermodynamic trajectories

## Algorithm Details

### Traditional CBS vs F-CBS

| Aspect | CBS | F-CBS |
|--------|-----|-------|
| **Objective** | Minimize total cost | Minimize free energy |
| **Search Strategy** | Greedy cost minimization | Temperature-controlled exploration |
| **Conflict Resolution** | Immediate constraint addition | Thermodynamic constraint weighting |
| **Success Rate** | Lower on complex scenarios | Higher with proper temperature control |

### Entropy Calculation

F-CBS calculates conflict entropy using spatial distribution:

```python
def calculate_entropy(conflicts):
    # Weight conflicts by type and location
    for conflict in conflicts:
        if conflict['type'] == 'vertex':
            conflict_counts[location] += 1.0
        elif conflict['type'] == 'edge':
            for location in edge_endpoints:
                conflict_counts[location] += 0.5
    
    # Shannon entropy
    entropy = -Σ p_i * log2(p_i)
    return entropy
```