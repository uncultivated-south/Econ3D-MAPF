import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')
import random
from fcbs_with_annealing import Grid, Agent
from thermodynamic import ThermodynamicFCBS

class CBSPhasePredictor:
    """
    Predicts CBS solvability based on thermodynamic phase transition signatures
    
    Key insight: Solvable scenarios show:
    - Consistent negative entropy production rate (disorder decreasing)
    - Consistent positive energy dissipation rate (energy flowing out)
    - Steady conflict reduction
    
    Unsolvable scenarios show:
    - Fluctuating entropy production (no clear trend)
    - Fluctuating energy dissipation (no steady flow)
    - Oscillating conflict counts (trapped in metastable states)
    """
    
    def __init__(self, 
                 stabilization_window=50,     # Window to check for stability
                 prediction_threshold=100,     # Minimum iterations before prediction
                 confidence_window=30):        # Window for trend confidence
        
        self.stabilization_window = stabilization_window
        self.prediction_threshold = prediction_threshold
        self.confidence_window = confidence_window
        
        # Thresholds (can be tuned based on empirical data)
        self.entropy_trend_threshold = -0.001   # Negative trend indicates solving
        self.energy_trend_threshold = 0.001     # Positive trend indicates solving
        self.stability_ratio_threshold = 0.7    # Fraction of window with consistent trend
        self.conflict_oscillation_threshold = 0.3  # CV threshold for conflict stability
        
    def calculate_trend_stability(self, values, window_size=None):
        """
        Calculate how stable a trend is over a window
        Returns (trend_direction, stability_ratio, trend_strength)
        """
        if window_size is None:
            window_size = self.confidence_window
            
        if len(values) < window_size:
            return 0.0, 0.0, 0.0
            
        recent_values = values[-window_size:]
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_values))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        except:
            return 0.0, 0.0, 0.0
            
        # Calculate stability: how many consecutive points follow the trend
        predicted = slope * x + intercept
        residuals = recent_values - predicted
        
        # Count points that follow trend direction
        if slope != 0:
            correct_direction = np.sum(np.sign(np.diff(recent_values)) == np.sign(slope)) / max(1, len(recent_values) - 1)
        else:
            correct_direction = 0.0
            
        return slope, correct_direction, abs(r_value)
    
    def calculate_conflict_oscillation_index(self, conflict_counts, window_size=None):
        """
        Calculate coefficient of variation for conflict counts
        High CV indicates oscillation/instability
        """
        if window_size is None:
            window_size = self.confidence_window
            
        if len(conflict_counts) < window_size:
            return float('inf')
            
        recent_conflicts = conflict_counts[-window_size:]
        mean_conflicts = np.mean(recent_conflicts)
        
        if mean_conflicts == 0:
            return 0.0
            
        cv = np.std(recent_conflicts) / mean_conflicts
        return cv
    
    def detect_phase_transition_signatures(self, tracking_data):
        """
        Detect thermodynamic signatures indicating solvable vs unsolvable phases
        """
        if len(tracking_data) < self.prediction_threshold:
            return {
                'prediction': 'insufficient_data',
                'confidence': 0.0,
                'phase': 'unknown',
                'signatures': {}
            }
        
        # Extract time series
        entropy_rates = tracking_data['entropy_production_rate'].values
        energy_rates = tracking_data['energy_dissipation_rate'].values
        conflict_counts = tracking_data['num_conflicts'].values
        
        # Calculate trend stabilities
        entropy_slope, entropy_stability, entropy_r2 = self.calculate_trend_stability(entropy_rates)
        energy_slope, energy_stability, energy_r2 = self.calculate_trend_stability(energy_rates)
        
        # Calculate conflict oscillation
        conflict_oscillation = self.calculate_conflict_oscillation_index(conflict_counts)
        
        # Calculate recent averages
        recent_entropy_rate = np.mean(entropy_rates[-self.confidence_window:])
        recent_energy_rate = np.mean(energy_rates[-self.confidence_window:])
        recent_conflicts = np.mean(conflict_counts[-self.confidence_window:])
        
        # Phase transition signatures
        signatures = {
            'entropy_production_rate': recent_entropy_rate,
            'energy_dissipation_rate': recent_energy_rate,
            'entropy_trend_slope': entropy_slope,
            'energy_trend_slope': energy_slope,
            'entropy_stability': entropy_stability,
            'energy_stability': energy_stability,
            'entropy_correlation': entropy_r2,
            'energy_correlation': energy_r2,
            'conflict_oscillation_cv': conflict_oscillation,
            'recent_conflict_average': recent_conflicts,
            'total_iterations': len(tracking_data)
        }
        
        # Solvable phase indicators:
        solvable_indicators = [
            recent_entropy_rate < self.entropy_trend_threshold,  # Entropy decreasing
            recent_energy_rate > self.energy_trend_threshold,    # Energy dissipating
            entropy_stability > self.stability_ratio_threshold,  # Stable entropy trend
            energy_stability > self.stability_ratio_threshold,   # Stable energy trend
            conflict_oscillation < self.conflict_oscillation_threshold,  # Low conflict oscillation
            entropy_slope < 0,  # Negative entropy trend
            energy_slope > 0,   # Positive energy trend
        ]
        
        # Calculate confidence based on how many indicators agree
        solvable_score = sum(solvable_indicators) / len(solvable_indicators)
        
        # Make prediction
        if solvable_score > 0.6:
            prediction = 'solvable'
            phase = 'ordered_convergence'
            confidence = solvable_score
        elif solvable_score < 0.4:
            prediction = 'unsolvable'
            phase = 'disordered_oscillation'
            confidence = 1.0 - solvable_score
        else:
            prediction = 'uncertain'
            phase = 'critical_transition'
            confidence = 0.5
            
        signatures.update({
            'solvable_indicators': solvable_indicators,
            'solvable_score': solvable_score
        })
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'phase': phase,
            'signatures': signatures
        }
    
    def analyze_phase_evolution(self, tracking_data):
        """
        Analyze how the system evolves through different phases
        """
        if len(tracking_data) < 50:
            return None
            
        phases = []
        window_size = 50
        
        for i in range(window_size, len(tracking_data), 10):  # Every 10 iterations
            window_data = tracking_data.iloc[i-window_size:i]
            phase_info = self.detect_phase_transition_signatures(window_data)
            
            phases.append({
                'iteration': i,
                'phase': phase_info['phase'],
                'prediction': phase_info['prediction'],
                'confidence': phase_info['confidence'],
                'entropy_rate': phase_info['signatures']['entropy_production_rate'],
                'energy_rate': phase_info['signatures']['energy_dissipation_rate'],
                'conflict_oscillation': phase_info['signatures']['conflict_oscillation_cv']
            })
        
        return pd.DataFrame(phases)
    
    def visualize_phase_signatures(self, tracking_data, title="CBS Phase Analysis"):
        """
        Create comprehensive visualization of phase transition signatures
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16)
        
        iterations = tracking_data['iteration']
        
        # Plot 1: Thermodynamic rates with prediction zones
        axes[0, 0].plot(iterations, tracking_data['entropy_production_rate'], 
                       label='Entropy Production Rate', alpha=0.7, color='blue')
        axes[0, 0].axhline(y=self.entropy_trend_threshold, color='blue', linestyle='--', alpha=0.5,
                          label=f'Solvable Threshold ({self.entropy_trend_threshold})')
        axes[0, 0].plot(iterations, tracking_data['energy_dissipation_rate'], 
                       label='Energy Dissipation Rate', alpha=0.7, color='red')
        axes[0, 0].axhline(y=self.energy_trend_threshold, color='red', linestyle='--', alpha=0.5,
                          label=f'Solvable Threshold ({self.energy_trend_threshold})')
        axes[0, 0].set_title('Thermodynamic Rate Signatures')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Conflict evolution with oscillation analysis
        axes[0, 1].plot(iterations, tracking_data['num_conflicts'], alpha=0.8, color='orange')
        
        # Add moving average and oscillation bands
        window = 20
        if len(tracking_data) > window:
            moving_avg = tracking_data['num_conflicts'].rolling(window=window).mean()
            moving_std = tracking_data['num_conflicts'].rolling(window=window).std()
            axes[0, 1].plot(iterations, moving_avg, '--', color='black', alpha=0.7, label=f'MA({window})')
            axes[0, 1].fill_between(iterations, moving_avg - moving_std, moving_avg + moving_std, 
                                  alpha=0.2, color='gray', label='±1σ')
        
        axes[0, 1].set_title('Conflict Evolution & Oscillation')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Number of Conflicts')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Phase space with trajectory coloring
        scatter = axes[1, 0].scatter(tracking_data['cost'], tracking_data['entropy'], 
                                   c=iterations, cmap='plasma', alpha=0.6, s=10)
        axes[1, 0].set_title('Phase Space Trajectory')
        axes[1, 0].set_xlabel('Cost')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Iteration')
        
        # Plot 4: Rolling trend analysis
        window_size = 50
        if len(tracking_data) > window_size:
            rolling_entropy_trend = []
            rolling_energy_trend = []
            rolling_iterations = []
            
            for i in range(window_size, len(tracking_data)):
                window_data = tracking_data.iloc[i-window_size:i]
                
                # Calculate trends
                entropy_slope, _, _ = self.calculate_trend_stability(
                    window_data['entropy_production_rate'].values, window_size
                )
                energy_slope, _, _ = self.calculate_trend_stability(
                    window_data['energy_dissipation_rate'].values, window_size
                )
                
                rolling_entropy_trend.append(entropy_slope)
                rolling_energy_trend.append(energy_slope)
                rolling_iterations.append(i)
            
            axes[1, 1].plot(rolling_iterations, rolling_entropy_trend, 
                          label='Entropy Trend Slope', alpha=0.7)
            axes[1, 1].plot(rolling_iterations, rolling_energy_trend, 
                          label='Energy Trend Slope', alpha=0.7)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title(f'Rolling Trend Analysis (window={window_size})')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Slope')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: System dynamics
        axes[2, 0].plot(iterations, tracking_data['exploration_rate'], 
                       label='Exploration Rate', alpha=0.7)
        axes[2, 0].plot(iterations, tracking_data['conflict_persistence'], 
                       label='Conflict Persistence', alpha=0.7)
        axes[2, 0].set_title('System Dynamics')
        axes[2, 0].set_xlabel('Iteration')
        axes[2, 0].set_ylabel('Rate')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Prediction confidence over time
        phase_evolution = self.analyze_phase_evolution(tracking_data)
        if phase_evolution is not None:
            colors = {'solvable': 'green', 'unsolvable': 'red', 'uncertain': 'orange'}
            for phase_type in colors.keys():
                phase_data = phase_evolution[phase_evolution['prediction'] == phase_type]
                if len(phase_data) > 0:
                    axes[2, 1].scatter(phase_data['iteration'], phase_data['confidence'], 
                                     c=colors[phase_type], label=phase_type, alpha=0.7, s=20)
            
            axes[2, 1].set_title('Prediction Confidence Evolution')
            axes[2, 1].set_xlabel('Iteration')
            axes[2, 1].set_ylabel('Confidence')
            axes[2, 1].set_ylim(0, 1)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_prediction_report(self, tracking_data, actual_result=None):
        """
        Generate a comprehensive prediction report
        """
        prediction_info = self.detect_phase_transition_signatures(tracking_data)
        phase_evolution = self.analyze_phase_evolution(tracking_data)
        
        report = f"""
=== CBS PHASE TRANSITION PREDICTION REPORT ===

PREDICTION: {prediction_info['prediction'].upper()}
CONFIDENCE: {prediction_info['confidence']:.2f}
DETECTED PHASE: {prediction_info['phase']}

=== THERMODYNAMIC SIGNATURES ===
Entropy Production Rate (recent avg): {prediction_info['signatures']['entropy_production_rate']:.6f}
Energy Dissipation Rate (recent avg): {prediction_info['signatures']['energy_dissipation_rate']:.6f}
Entropy Trend Slope: {prediction_info['signatures']['entropy_trend_slope']:.6f}
Energy Trend Slope: {prediction_info['signatures']['energy_trend_slope']:.6f}

=== STABILITY ANALYSIS ===
Entropy Stability Ratio: {prediction_info['signatures']['entropy_stability']:.3f}
Energy Stability Ratio: {prediction_info['signatures']['energy_stability']:.3f}
Conflict Oscillation CV: {prediction_info['signatures']['conflict_oscillation_cv']:.3f}

=== SOLVABLE INDICATORS ===
"""
        
        indicator_names = [
            "Entropy decreasing consistently",
            "Energy dissipating consistently", 
            "Entropy trend stable",
            "Energy trend stable",
            "Low conflict oscillation",
            "Negative entropy slope",
            "Positive energy slope"
        ]
        
        for i, (name, indicator) in enumerate(zip(indicator_names, 
                                                 prediction_info['signatures']['solvable_indicators'])):
            status = "✓" if indicator else "✗"
            report += f"{status} {name}\n"
        
        report += f"\nSOLVABLE SCORE: {prediction_info['signatures']['solvable_score']:.3f}\n"
        
        if actual_result is not None:
            accuracy = "CORRECT" if (
                (prediction_info['prediction'] == 'solvable' and actual_result) or
                (prediction_info['prediction'] == 'unsolvable' and not actual_result)
            ) else "INCORRECT"
            report += f"\nACTUAL RESULT: {'SOLVED' if actual_result else 'FAILED'}"
            report += f"\nPREDICTION ACCURACY: {accuracy}\n"
        
        if phase_evolution is not None and len(phase_evolution) > 0:
            report += f"\n=== PHASE EVOLUTION ===\n"
            report += f"Initial Phase: {phase_evolution.iloc[0]['phase']}\n"
            report += f"Final Phase: {phase_evolution.iloc[-1]['phase']}\n"
            
            # Count phase transitions
            phase_changes = sum(phase_evolution['phase'].iloc[i] != phase_evolution['phase'].iloc[i-1] 
                              for i in range(1, len(phase_evolution)))
            report += f"Phase Transitions: {phase_changes}\n"
        
        return report


def test_phase_predictor_on_scenarios(num_scenarios=50, max_iterations=500):
    """
    Test the phase predictor on multiple scenarios to validate accuracy
    """
    
    predictor = CBSPhasePredictor()
    results = []
    
    print("Testing Phase Predictor on Multiple Scenarios")
    print("=" * 60)
    
    for seed in range(num_scenarios):
        print(f"\nScenario {seed + 1}/{num_scenarios}")
        
        # Generate scenario
        random.seed(seed)
        np.random.seed(seed)
        
        # Create grid and agents (12x12, 10 agents, 12 obstacles)
        width, height = 12, 12
        obstacles = set()
        while len(obstacles) < 12:
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            obstacles.add((x, y))
        
        grid = Grid(width, height, obstacles)
        
        agents = []
        occupied = obstacles.copy()
        
        for i in range(10):  # 10 agents for faster testing
            while True:
                start = (random.randint(0, width-1), random.randint(0, height-1))
                if start not in occupied:
                    occupied.add(start)
                    break
            
            while True:
                goal = (random.randint(0, width-1), random.randint(0, height-1))
                if goal not in occupied and goal != start:
                    break
            
            agents.append(Agent(i, start, goal))
        
        # Test with ThermodynamicFCBS
        solver = ThermodynamicFCBS(grid, agents, temperature=0)
        solution, iterations, solve_time, tracking_data = solver.solve(max_iterations)
        
        actual_solved = solution is not None
        
        # Make prediction at 20% of max iterations
        prediction_point = int(0.2 * max_iterations)
        if len(tracking_data) >= prediction_point:
            early_data = tracking_data.iloc[:prediction_point]
            prediction_info = predictor.detect_phase_transition_signatures(early_data)
            predicted_solvable = prediction_info['prediction'] == 'solvable'
            confidence = prediction_info['confidence']
        else:
            predicted_solvable = False
            confidence = 0.0
            prediction_info = {'prediction': 'insufficient_data'}
        
        # Record results
        correct = (predicted_solvable == actual_solved) if prediction_info['prediction'] != 'insufficient_data' else False
        
        result = {
            'seed': seed,
            'actual_solved': actual_solved,
            'predicted_solvable': predicted_solvable,
            'prediction': prediction_info['prediction'],
            'confidence': confidence,
            'correct': correct,
            'iterations': iterations,
            'prediction_point': prediction_point
        }
        
        results.append(result)
        
        status = "✓" if correct else "✗" if prediction_info['prediction'] != 'insufficient_data' else "?"
        print(f"  Actual: {'SOLVED' if actual_solved else 'FAILED'}")
        print(f"  Predicted: {prediction_info['prediction'].upper()} (conf: {confidence:.2f})")
        print(f"  Result: {status}")
    
    # Calculate accuracy
    results_df = pd.DataFrame(results)
    valid_predictions = results_df[results_df['prediction'] != 'insufficient_data']
    
    if len(valid_predictions) > 0:
        accuracy = valid_predictions['correct'].mean()
        print(f"\n=== PREDICTION ACCURACY ===")
        print(f"Valid Predictions: {len(valid_predictions)}/{len(results_df)}")
        print(f"Overall Accuracy: {accuracy:.2%}")
        
        # Breakdown by actual result
        for actual in [True, False]:
            subset = valid_predictions[valid_predictions['actual_solved'] == actual]
            if len(subset) > 0:
                acc = subset['correct'].mean()
                result_type = "Solvable" if actual else "Unsolvable"
                print(f"{result_type} Scenarios: {len(subset)} cases, {acc:.2%} accuracy")
    
    # Save results
    results_df.to_csv('phase_predictor_validation.csv', index=False)
    print(f"\nResults saved to 'phase_predictor_validation.csv'")
    
    return results_df


# Example usage
if __name__ == "__main__":
    # Test the predictor
    predictor = CBSPhasePredictor()
    
    # Load some tracking data (replace with actual file)
    # tracking_data = pd.read_csv('thermodynamic_trace_F-CBS_T=5.0_seed_42.csv')
    
    # Make prediction
    # prediction_info = predictor.detect_phase_transition_signatures(tracking_data)
    # print(predictor.generate_prediction_report(tracking_data))
    
    # Visualize
    # fig = predictor.visualize_phase_signatures(tracking_data)
    # plt.show()
    
    # Run validation test
    validation_results = test_phase_predictor_on_scenarios(num_scenarios=1000, max_iterations=1000)