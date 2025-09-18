# CBS Thermodynamic Analysis: Physics-Inspired Multi-Agent Pathfinding

A novel approach to Multi-Agent Pathfinding (MAPF) that applies **thermodynamic principles** and **annealing methods** to improve Conflict-Based Search (CBS) performance and predict algorithmic behavior.

## 🌟 Overview

This project introduces two major innovations to CBS:

1. **🔥 Free Energy CBS (F-CBS) with Annealing**: A thermodynamic optimization approach that significantly improves CBS success rates
2. **🔮 Thermodynamic Solvability Prediction**: Early detection of whether CBS will find a solution based on non-equilibrium phase signatures

## 🔬 Core Innovation: Free Energy Optimization

### The Problem with Traditional CBS
Traditional CBS uses **pure cost minimization** (shortest total path length) to guide search through the constraint tree. However, this greedy approach can get trapped in high-conflict regions of the search space, leading to exponential explosion or failure to find solutions.

### Our Solution: Thermodynamic Free Energy
We replace cost-only optimization with **free energy minimization**, borrowing from statistical mechanics:

```
F = U - T × S
```

Where:
- **F**: Free Energy (our new optimization objective)
- **U**: Internal Energy (traditional path cost)
- **T**: Temperature (exploration parameter) 
- **S**: Entropy (information-theoretic measure of conflict distribution)

### Physics Concepts Borrowed

#### 1. Statistical Mechanics Framework
We model CBS as a **thermodynamic ensemble** where:
- **System States**: Different constraint-solution configurations
- **Energy Landscape**: Cost-entropy space that the algorithm explores
- **Equilibrium**: Conflict-free solution (minimum free energy state)

#### 2. Entropy as Conflict Organization
**Entropy Definition**: Information-theoretic measure of how conflicts are distributed across the grid:

```python
S = -Σ p_i × log₂(p_i)
```

Where `p_i` is the probability of conflicts occurring at grid location `i`.

**Physical Interpretation**:
- **High Entropy**: Conflicts spread uniformly (disordered, harder to resolve)
- **Low Entropy**: Conflicts concentrated in few locations (ordered, easier to resolve)

#### 3. Temperature-Controlled Exploration
**Temperature** controls the exploration-exploitation trade-off:
- **T > 0**: System explores broadly, accepts higher-cost paths to escape local minima
- **T < 0**: System exploits locally, focuses on cost minimization
- **T = 0**: Pure cost minimization (traditional CBS)

#### 4. Simulated Annealing Schedule
**Annealing Process**: Gradually reduce temperature during search:
```
T(t) = T₀    if t ≤ t_anneal
T(t) = 0     if t > t_anneal
```

**Physical Analogy**: Like cooling molten metal to form perfect crystals, we "cool" the search to crystallize optimal solutions.

**⚠️ Important Note**: Negative Temperature Interpretation
**Physics vs. Algorithm Discrepancy**: Our empirical results reveal an intriguing phenomenon that requires further investigation. While promising performance improvements are observed, our system effectively operates with negative temperatures in the thermodynamic sense, which conflicts with conventional physical interpretation where temperature must be positive in equilibrium systems.
**Our Current Hypothesis**: We propose that negative temperatures in our algorithm may represent external cooling mechanisms - active intervention that drives the system toward ordered states, analogous to refrigeration or heat pumps in physics. Conversely, positive temperatures may represent heating processes that add disorder to the system. This interpretation suggests our algorithm implements an active thermodynamic control rather than passive equilibrium dynamics.
**Ongoing Research**: This theoretical inconsistency represents a fascinating area for continued exploration. We are investigating whether: (1) our entropy formulation requires modification to align with physical principles, (2) the algorithm operates in a non-equilibrium regime where negative temperatures have meaning, or (3) a fundamentally different thermodynamic framework is needed for discrete combinatorial systems. Understanding this discrepancy may lead to deeper insights into the relationship between information theory, statistical mechanics, and algorithmic optimization.

## 🚀 F-CBS Algorithm: Thermodynamic CBS

### Enhanced Node Priority
Instead of pure cost-based priority:
```python
# Traditional CBS
priority = cost

# F-CBS (Our Approach)  
priority = cost + temperature * entropy
```

### Annealing Strategy
```python
def update_temperature(iteration):
    if iteration <= annealing_threshold:
        return initial_temperature  # Exploration phase
    else:
        return 0.0                  # Exploitation phase
```

### Performance Improvements
**Empirical Results** (1000 scenarios, 10 agents, 16 obstacles):

| Method | Success Rate | Avg Iterations |
|--------|--------------|----------------|
| CBS | 38.6% | 28.8 |
| F-CBS Anneal (T=12.0→0) | 43.5% | 14.9 |
| F-CBS Anneal (T=10.0→0) | 43.2% | 14.9 |
| F-CBS Anneal (T=5.0→0) | 43.0% | 13.8 |
| F-CBS Anneal (T=2.0→0) | 42.4% | 13.4 |
| F-CBS Anneal (T=1.0→0) | 41.4% | 15.0 |

## 🔮 Breakthrough Discovery: Thermodynamic Solvability Prediction

### The Empirical Observation
Through extensive experimentation, we discovered that CBS exhibits **phase transition signatures** that predict whether a solution will be found:

#### Solvable Scenarios ("Ordered Convergence Phase")
- **Entropy Production Rate** (`∂S/∂t`): Consistently **negative** (system becoming ordered)
- **Energy Dissipation Rate** (`-∂F/∂t`): Consistently **positive** (energy flowing out)
- **Conflict Evolution**: Monotonic decrease toward zero

#### Unsolvable Scenarios ("Disordered Oscillation Phase")  
- **Entropy Production Rate**: Fluctuating around zero (no progress toward order)
- **Energy Dissipation Rate**: Oscillating (trapped in metastable states)
- **Conflict Evolution**: Stuck oscillating within narrow range

### Physics Concepts: Non-Equilibrium Thermodynamics

#### 1. Entropy Production
**Physical Meaning**: Rate at which the system becomes more or less ordered.
```python
entropy_production_rate = ΔS / Δt
```
- **Negative rate**: System organizing (conflicts resolving systematically)
- **Zero rate**: System at steady state (likely trapped)
- **Positive rate**: System becoming more disordered

#### 2. Energy Dissipation  
**Physical Meaning**: Rate at which free energy leaves the system.
```python
energy_dissipation_rate = -ΔF / Δt  
```
- **Positive rate**: Energy flowing toward equilibrium (approaching solution)
- **Zero rate**: No energy flow (system stuck)
- **Negative rate**: Energy building up (moving away from solution)

#### 3. Phase Transitions
**Critical Phenomena**: Sharp transitions between qualitatively different behaviors.

**Solvable Phase Properties**:
- High ergodicity (efficient exploration)
- Steady energy dissipation
- Monotonic entropy decrease

**Unsolvable Phase Properties**:
- Ergodicity breaking (trapped in subspaces)
- Energy oscillation
- Entropy fluctuation without trend

#### 4. Non-Equilibrium Steady States
**Metastable Traps**: Local minima in free energy landscape where algorithm gets stuck.
- System continues to "work" (iterate) but makes no progress
- Energy and entropy oscillate without approaching equilibrium
- Characteristic signature of unsolvable scenarios

### Prediction Framework

#### Phase Classification Algorithm
```python
def detect_phase_transition_signatures(tracking_data):
    # Calculate thermodynamic rates
    entropy_rate = calculate_trend(entropy_time_series)
    energy_rate = calculate_trend(energy_time_series)
    
    # Solvable indicators
    ordered_convergence = (
        entropy_rate < -0.001 and      # Entropy decreasing
        energy_rate > 0.001 and        # Energy dissipating  
        trend_stability > 0.7          # Consistent trends
    )
    
    return "solvable" if ordered_convergence else "unsolvable"
```

#### Early Termination Benefits
- **Computational Savings**: reduction in wasted iterations
- **Resource Allocation**: Focus computation on promising scenarios

## 🧪 Theoretical Foundation

### Statistical Mechanics Perspective
We model CBS as an **open thermodynamic system**:

1. **System**: Current constraint-solution state
2. **Environment**: Space of all possible solutions  
3. **Energy Exchange**: Adding/removing constraints changes internal energy
4. **Entropy Exchange**: Conflict resolution affects system organization

### Information Theory Connection
**Conflicts as Information**: Each conflict represents "information" that must be processed:
- **High entropy conflicts**: Information spread randomly (hard to process)
- **Low entropy conflicts**: Information localized (easier to process systematically)

### Critical Phenomena Analogy
CBS failure resembles **phase transitions** in physics:
- **Order Parameter**: Number of remaining conflicts
- **Critical Point**: Threshold where solvability changes dramatically  
- **Finite-Size Effects**: Scaling behavior with problem size

## 📊 Key Innovations Summary

### 1. F-CBS Algorithm
- **Novel Objective**: Free energy instead of pure cost
- **Entropy Integration**: Conflict distribution awareness
- **Temperature Control**: Exploration-exploitation balance
- **Annealing Schedule**: Adaptive cooling strategy

### 2. Thermodynamic Tracking
- **Entropy Production Rate**: System organization measure
- **Energy Dissipation Rate**: Progress toward equilibrium
- **Phase Space Analysis**: State trajectory visualization
- **Stability Metrics**: Trend consistency quantification

### 3. Predictive Capabilities
- **Phase Classification**: Ordered vs disordered dynamics
- **Early Termination**: Stop before certain failure
- **Confidence Metrics**: Prediction reliability quantification
- **Computational Savings**: Significant efficiency gains

## 🎯 Quick Start

### Basic F-CBS with Annealing
```python
from cbs_thermodynamic import *

# Create scenario
grid = Grid(width=12, height=12, obstacles={(2,3), (5,7), (8,9)})
agents = [Agent(0, start=(0,0), goal=(11,11))]

# Run F-CBS with annealing (recommended)
solver = FCBS(grid, agents, 
              temperature=5.0,           # Initial exploration
              using_annealing=True,      # Enable cooling
              annealing_iterations=50)   # When to start cooling

solution, iterations, time = solver.solve()
print(f"Success: {solution is not None}, Iterations: {iterations}")
```

### Thermodynamic Prediction
```python
# Enhanced solver with tracking
solver = ThermodynamicFCBS(grid, agents, temperature=5.0, using_annealing=True)
solution, iterations, time, tracking_data = solver.solve()

# Make solvability prediction  
predictor = CBSPhasePredictor()
prediction = predictor.detect_phase_transition_signatures(tracking_data)
print(f"Prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.2f})")

# Visualize thermodynamic evolution
fig = predictor.visualize_phase_signatures(tracking_data)
plt.show()
```

### Comparative Analysis
```python
# Compare all methods
runner = ExperimentRunner()
results = runner.run_experiment(num_scenarios=100)
success_rates, avg_iterations, avg_cost = runner.analyze_results(results)
```

---