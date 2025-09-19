import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import json

# Import the SIPP-based implementations from the previous artifact
from sipp_fcbs_implementation import (
    AdaptiveGrid, Agent, AdaptiveSIPP_CBS, SIPP_FCBS, MovingAILoader,
    VertexConstraint, EdgeConstraint
)

class MovingAIBenchmarkRunner:
    def __init__(self, benchmark_directory: str):
        """
        Initialize benchmark runner for Moving AI Lab datasets
        
        Args:
            benchmark_directory: Path to directory containing .map and .scen files
        """
        self.benchmark_dir = benchmark_directory
        self.loader = MovingAILoader()
        self.results = []
        
        # Temperature values to test for F-CBS
        self.temperature_values = [1.0, 2.0, 5.0, 10.0]
        
        # Algorithm configurations to test
        self.algorithm_configs = [
            {'name': 'SIPP-CBS', 'class': AdaptiveSIPP_CBS, 'params': {}},
            {'name': 'SIPP-F-CBS-Anneal-T5.0', 'class': SIPP_FCBS, 
             'params': {'temperature': 5.0, 'using_annealing': True, 'annealing_iterations': 5}},
            {'name': 'SIPP-F-CBS-Anneal-T10.0', 'class': SIPP_FCBS,
             'params': {'temperature': 10.0, 'using_annealing': True, 'annealing_iterations': 5}},
        ]
    
    def discover_benchmark_files(self) -> List[Tuple[str, str]]:
        """
        Discover all .map and .scen file pairs in the benchmark directory
        
        Returns:
            List of (map_file, scen_file) tuples
        """
        map_files = []
        scen_files = []
        
        # Recursively find all map and scenario files
        for root, dirs, files in os.walk(self.benchmark_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.map'):
                    map_files.append(file_path)
                elif file.endswith('.scen'):
                    scen_files.append(file_path)
        
        # Match map and scenario files
        benchmark_pairs = []
        for map_file in map_files:
            map_base = os.path.splitext(os.path.basename(map_file))[0]
            
            # Find matching scenario files
            for scen_file in scen_files:
                scen_base = os.path.basename(scen_file)
                if map_base in scen_base:
                    benchmark_pairs.append((map_file, scen_file))
        
        print(f"Discovered {len(benchmark_pairs)} benchmark pairs")
        return benchmark_pairs
    
    def run_single_benchmark(self, map_file: str, scen_file: str, 
                           agent_counts: List[int] = [15, 20, 25, 30, 35, 40],
                           timeout_seconds: int = 300) -> List[Dict]:
        """
        Run all algorithms on a single benchmark scenario
        
        Args:
            map_file: Path to .map file
            scen_file: Path to .scen file
            agent_counts: List of agent counts to test
            timeout_seconds: Timeout per algorithm run
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        try:
            # Load map and scenario
            grid = self.loader.load_map(map_file)
            all_agents = self.loader.load_scenario(scen_file)
            
            if not all_agents:
                print(f"No agents found in {scen_file}")
                return results
            
            map_name = os.path.basename(map_file)
            scen_name = os.path.basename(scen_file)
            
            print(f"Testing {map_name} with {scen_name}")
            print(f"Grid size: {grid.width}x{grid.height}, Obstacles: {len(grid.obstacles)}")
            
            # Test different agent counts
            for num_agents in agent_counts:
                if num_agents > len(all_agents):
                    continue
                    
                agents = all_agents[:num_agents]
                print(f"  Testing with {num_agents} agents...")
                
                # Test each algorithm configuration
                for config in self.algorithm_configs:
                    print(f"    Running {config['name']}...", end=" ")
                    
                    try:
                        # Create algorithm instance
                        algorithm = config['class'](grid, agents, **config['params'])
                        
                        # Run with timeout
                        start_time = time.time()
                        solution, iterations, solve_time = algorithm.solve(max_iterations=100000)
                        total_time = time.time() - start_time
                        
                        # Calculate metrics
                        if solution:
                            cost = sum(len(path) - 1 for path in solution.values())
                            makespan = max(len(path) - 1 for path in solution.values())
                            
                            # Calculate entropy if F-CBS
                            if hasattr(algorithm, 'calculate_path_congestion_entropy'):
                                entropy = algorithm.calculate_path_congestion_entropy(solution)
                                free_energy = cost + config['params'].get('temperature', 0) * entropy
                            else:
                                entropy = 0.0
                                free_energy = cost
                            
                            success = True
                            print("SUCCESS")
                        else:
                            cost = float('inf')
                            makespan = float('inf')
                            entropy = 0.0
                            free_energy = float('inf')
                            success = False
                            print("FAILED")
                        
                        # Check timeout
                        if total_time > timeout_seconds:
                            success = False
                            print(f" (TIMEOUT after {total_time:.1f}s)")
                        
                        # Record results
                        result = {
                            'map_file': map_name,
                            'scen_file': scen_name,
                            'grid_width': grid.width,
                            'grid_height': grid.height,
                            'num_obstacles': len(grid.obstacles),
                            'num_agents': num_agents,
                            'algorithm': config['name'],
                            'temperature': config['params'].get('temperature', None),
                            'annealing': config['params'].get('using_annealing', False),
                            'success': success,
                            'cost': cost,
                            'makespan': makespan,
                            'entropy': entropy,
                            'free_energy': free_energy,
                            'iterations': iterations,
                            'solve_time': solve_time,
                            'total_time': total_time,
                            'timeout': total_time > timeout_seconds
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"ERROR: {e}")
                        # Record failure
                        result = {
                            'map_file': map_name,
                            'scen_file': scen_name,
                            'grid_width': grid.width,
                            'grid_height': grid.height,
                            'num_obstacles': len(grid.obstacles),
                            'num_agents': num_agents,
                            'algorithm': config['name'],
                            'temperature': config['params'].get('temperature', None),
                            'annealing': config['params'].get('using_annealing', False),
                            'success': False,
                            'cost': float('inf'),
                            'makespan': float('inf'),
                            'entropy': 0.0,
                            'free_energy': float('inf'),
                            'iterations': 0,
                            'solve_time': 0,
                            'total_time': 0,
                            'timeout': False,
                            'error': str(e)
                        }
                        results.append(result)
        
        except Exception as e:
            print(f"Error loading benchmark {map_file}: {e}")
        
        return results
    
    def run_full_benchmark_suite(self, 
                                max_scenarios: Optional[int] = None,
                                agent_counts: List[int] = [5, 10, 15, 20, 25],
                                timeout_seconds: int = 300) -> pd.DataFrame:
        """
        Run the complete benchmark suite
        
        Args:
            max_scenarios: Maximum number of scenarios to test (None for all)
            agent_counts: List of agent counts to test
            timeout_seconds: Timeout per algorithm run
            
        Returns:
            DataFrame with all results
        """
        print("=== SIPP F-CBS Moving AI Lab Benchmark Suite ===")
        print(f"Benchmark directory: {self.benchmark_dir}")
        
        # Discover benchmark files
        benchmark_pairs = self.discover_benchmark_files()
        
        if not benchmark_pairs:
            print("No benchmark files found!")
            return pd.DataFrame()
        
        if max_scenarios:
            benchmark_pairs = benchmark_pairs[:max_scenarios]
            print(f"Limited to first {max_scenarios} scenarios")
        
        all_results = []
        
        # Run benchmarks
        for i, (map_file, scen_file) in enumerate(benchmark_pairs):
            print(f"\n--- Benchmark {i+1}/{len(benchmark_pairs)} ---")
            
            scenario_results = self.run_single_benchmark(
                map_file, scen_file, agent_counts, timeout_seconds
            )
            all_results.extend(scenario_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f'sipp_fcbs_benchmark_results_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df
    
    def analyze_results(self, results_df: pd.DataFrame) -> None:
        """
        Comprehensive analysis of benchmark results
        
        Args:
            results_df: DataFrame with benchmark results
        """
        if results_df.empty:
            print("No results to analyze")
            return
        
        print("\n" + "="*60)
        print("SIPP F-CBS BENCHMARK ANALYSIS")
        print("="*60)
        
        # Overall statistics
        print(f"\nDataset Overview:")
        print(f"Total scenarios tested: {results_df['scen_file'].nunique()}")
        print(f"Total map files: {results_df['map_file'].nunique()}")
        print(f"Agent counts tested: {sorted(results_df['num_agents'].unique())}")
        print(f"Grid sizes: {sorted(results_df[['grid_width', 'grid_height']].drop_duplicates().apply(lambda x: f'{x.grid_width}x{x.grid_height}', axis=1).unique())}")
        
        # Success rate analysis
        print(f"\n" + "-"*40)
        print("SUCCESS RATES")
        print("-"*40)
        
        success_rates = results_df.groupby('algorithm')['success'].agg(['count', 'sum', 'mean']).round(3)
        success_rates['success_rate'] = (success_rates['sum'] / success_rates['count'] * 100).round(1)
        success_rates = success_rates.sort_values('success_rate', ascending=False)
        
        print("\nOverall Success Rates:")
        for alg, row in success_rates.iterrows():
            print(f"{alg:25s}: {row.success_rate:5.1f}% ({row['sum']:.0f}/{row['count']:.0f})")
        
        # Performance analysis for successful runs
        successful_runs = results_df[results_df['success'] == True].copy()
        
        if len(successful_runs) > 0:
            print(f"\n" + "-"*40)
            print("PERFORMANCE METRICS (Successful runs only)")
            print("-"*40)
            
            # Cost analysis
            print("\nAverage Solution Cost:")
            cost_stats = successful_runs.groupby('algorithm')['cost'].agg(['mean', 'std', 'min', 'max']).round(2)
            cost_stats = cost_stats.sort_values('mean')
            
            for alg, row in cost_stats.iterrows():
                print(f"{alg:25s}: {row['mean']:7.1f} ± {row['std']:5.1f} (range: {row['min']:.0f}-{row['max']:.0f})")
            
            # Iteration analysis
            print("\nAverage Iterations to Solution:")
            iter_stats = successful_runs.groupby('algorithm')['iterations'].agg(['mean', 'std']).round(1)
            iter_stats = iter_stats.sort_values('mean')
            
            for alg, row in iter_stats.iterrows():
                print(f"{alg:25s}: {row['mean']:7.1f} ± {row['std']:5.1f}")
            
            # Runtime analysis
            print("\nAverage Runtime (seconds):")
            time_stats = successful_runs.groupby('algorithm')['solve_time'].agg(['mean', 'std']).round(3)
            time_stats = time_stats.sort_values('mean')
            
            for alg, row in time_stats.iterrows():
                print(f"{alg:25s}: {row['mean']:7.3f} ± {row['std']:6.3f}")
        
        # Agent count scaling analysis
        print(f"\n" + "-"*40)
        print("SCALING WITH AGENT COUNT")
        print("-"*40)
        
        for num_agents in sorted(results_df['num_agents'].unique()):
            agent_data = results_df[results_df['num_agents'] == num_agents]
            success_by_alg = agent_data.groupby('algorithm')['success'].mean() * 100
            
            print(f"\n{num_agents} agents:")
            for alg, success_rate in success_by_alg.sort_values(ascending=False).items():
                total_runs = len(agent_data[agent_data['algorithm'] == alg])
                print(f"  {alg:25s}: {success_rate:5.1f}% ({total_runs} runs)")
        
        # Temperature analysis for F-CBS variants
        fcbs_data = results_df[results_df['temperature'].notna()].copy()
        if len(fcbs_data) > 0:
            print(f"\n" + "-"*40)
            print("TEMPERATURE ANALYSIS (F-CBS variants)")
            print("-"*40)
            
            temp_analysis = fcbs_data.groupby('temperature').agg({
                'success': 'mean',
                'cost': lambda x: x[fcbs_data.loc[x.index, 'success']].mean() if any(fcbs_data.loc[x.index, 'success']) else np.nan,
                'iterations': lambda x: x[fcbs_data.loc[x.index, 'success']].mean() if any(fcbs_data.loc[x.index, 'success']) else np.nan,
                'entropy': lambda x: x[fcbs_data.loc[x.index, 'success']].mean() if any(fcbs_data.loc[x.index, 'success']) else np.nan
            }).round(3)
            
            print("\nTemperature vs Performance:")
            print("Temp   Success%   Avg Cost   Avg Iter   Avg Entropy")
            print("-" * 50)
            for temp, row in temp_analysis.iterrows():
                print(f"{temp:4.1f}   {row.success*100:6.1f}%   {row.cost:8.1f}   {row.iterations:8.1f}   {row.entropy:10.3f}")
        
        # Create visualizations
        self.create_visualizations(results_df)
    
    def create_visualizations(self, results_df: pd.DataFrame) -> None:
        """
        Create comprehensive visualizations of benchmark results
        
        Args:
            results_df: DataFrame with benchmark results
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Success Rate Comparison
        ax1 = plt.subplot(3, 3, 1)
        success_rates = results_df.groupby('algorithm')['success'].mean().sort_values(ascending=True)
        success_rates.plot(kind='barh', ax=ax1)
        ax1.set_title('Success Rate by Algorithm')
        ax1.set_xlabel('Success Rate')
        ax1.set_xlim(0, 1)
        
        # 2. Success Rate by Agent Count
        ax2 = plt.subplot(3, 3, 2)
        agent_success = results_df.groupby(['num_agents', 'algorithm'])['success'].mean().unstack(fill_value=0)
        
        # Select key algorithms for clarity
        key_algorithms = ['SIPP-CBS', 'SIPP-F-CBS-Anneal-T5.0', 'SIPP-F-CBS-Anneal-T10.0']
        available_algs = [alg for alg in key_algorithms if alg in agent_success.columns]
        
        if available_algs:
            agent_success[available_algs].plot(kind='line', marker='o', ax=ax2)
            ax2.set_title('Success Rate vs Agent Count')
            ax2.set_xlabel('Number of Agents')
            ax2.set_ylabel('Success Rate')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Runtime Distribution (successful runs)
        ax3 = plt.subplot(3, 3, 3)
        successful_runs = results_df[results_df['success'] == True]
        if len(successful_runs) > 0:
            algorithms_to_plot = successful_runs['algorithm'].value_counts().head(6).index
            runtime_data = [successful_runs[successful_runs['algorithm'] == alg]['solve_time'].values 
                           for alg in algorithms_to_plot]
            ax3.boxplot(runtime_data, labels=[alg.replace('SIPP-', '') for alg in algorithms_to_plot])
            ax3.set_title('Runtime Distribution')
            ax3.set_ylabel('Solve Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Cost vs Iterations Scatter
        ax4 = plt.subplot(3, 3, 4)
        if len(successful_runs) > 0:
            for alg in available_algs:
                alg_data = successful_runs[successful_runs['algorithm'] == alg]
                if len(alg_data) > 0:
                    ax4.scatter(alg_data['iterations'], alg_data['cost'], 
                              label=alg.replace('SIPP-', ''), alpha=0.6, s=20)
            ax4.set_xlabel('Iterations')
            ax4.set_ylabel('Solution Cost')
            ax4.set_title('Cost vs Iterations')
            ax4.legend()
        
        # 5. Temperature vs Performance (F-CBS only)
        ax5 = plt.subplot(3, 3, 5)
        fcbs_data = results_df[results_df['temperature'].notna()]
        if len(fcbs_data) > 0:
            fcbs_successful = fcbs_data[fcbs_data['success'] == True]
            if len(fcbs_successful) > 0:
                temp_performance = fcbs_successful.groupby('temperature').agg({
                    'cost': 'mean',
                    'iterations': 'mean',
                    'success': 'count'
                }).reset_index()
                
                ax5_twin = ax5.twinx()
                line1 = ax5.plot(temp_performance['temperature'], temp_performance['cost'], 
                               'bo-', label='Average Cost')
                line2 = ax5_twin.plot(temp_performance['temperature'], temp_performance['iterations'], 
                                    'ro-', label='Average Iterations')
                
                ax5.set_xlabel('Temperature')
                ax5.set_ylabel('Average Cost', color='b')
                ax5_twin.set_ylabel('Average Iterations', color='r')
                ax5.set_title('Temperature Impact on F-CBS')
                
                # Combine legends
                lines1, labels1 = ax5.get_legend_handles_labels()
                lines2, labels2 = ax5_twin.get_legend_handles_labels()
                ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 6. Grid Size Impact
        ax6 = plt.subplot(3, 3, 6)
        results_df['grid_size_cat'] = results_df.apply(
            lambda x: f"{x.grid_width}x{x.grid_height}", axis=1)
        
        size_success = results_df.groupby(['grid_size_cat', 'algorithm'])['success'].mean().unstack(fill_value=0)
        if len(size_success) > 0 and len(available_algs) > 0:
            size_success_subset = size_success[[col for col in available_algs if col in size_success.columns]]
            if len(size_success_subset.columns) > 0:
                size_success_subset.plot(kind='bar', ax=ax6, width=0.8)
                ax6.set_title('Success Rate by Grid Size')
                ax6.set_xlabel('Grid Size')
                ax6.set_ylabel('Success Rate')
                ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax6.tick_params(axis='x', rotation=45)
        
        # 7. Entropy Distribution (F-CBS variants)
        ax7 = plt.subplot(3, 3, 7)
        entropy_data = successful_runs[successful_runs['entropy'] > 0]
        if len(entropy_data) > 0:
            entropy_by_alg = entropy_data.groupby('algorithm')['entropy'].apply(list).to_dict()
            algorithms_with_entropy = [alg for alg in entropy_by_alg.keys() if 'F-CBS' in alg][:4]
            
            if algorithms_with_entropy:
                entropy_values = [entropy_by_alg[alg] for alg in algorithms_with_entropy]
                ax7.boxplot(entropy_values, labels=[alg.replace('SIPP-F-CBS-', '') for alg in algorithms_with_entropy])
                ax7.set_title('Entropy Distribution (F-CBS)')
                ax7.set_ylabel('Congestion Entropy')
                ax7.tick_params(axis='x', rotation=45)
        
        # 8. Free Energy vs Cost
        ax8 = plt.subplot(3, 3, 8)
        free_energy_data = successful_runs[successful_runs['free_energy'] != successful_runs['cost']]
        if len(free_energy_data) > 0:
            ax8.scatter(free_energy_data['cost'], free_energy_data['free_energy'], 
                       c=free_energy_data['temperature'], cmap='viridis', alpha=0.6)
            ax8.plot([free_energy_data['cost'].min(), free_energy_data['cost'].max()],
                    [free_energy_data['cost'].min(), free_energy_data['cost'].max()], 
                    'k--', alpha=0.5, label='Cost = Free Energy')
            
            cbar = plt.colorbar(ax8.collections[0], ax=ax8)
            cbar.set_label('Temperature')
            
            ax8.set_xlabel('Solution Cost')
            ax8.set_ylabel('Free Energy')
            ax8.set_title('Free Energy vs Cost')
            ax8.legend()
        
        # 9. Algorithm Comparison Heatmap
        ax9 = plt.subplot(3, 3, 9)
        
        # Create a performance matrix
        performance_metrics = ['success', 'cost', 'iterations', 'solve_time']
        alg_performance = successful_runs.groupby('algorithm')[performance_metrics].mean()
        
        if len(alg_performance) > 0:
            # Normalize each metric to 0-1 scale for comparison
            normalized_performance = alg_performance.copy()
            for metric in performance_metrics:
                if metric == 'success':  # Higher is better
                    continue
                else:  # Lower is better for cost, iterations, time
                    min_val = alg_performance[metric].min()
                    max_val = alg_performance[metric].max()
                    if max_val > min_val:
                        normalized_performance[metric] = 1 - (alg_performance[metric] - min_val) / (max_val - min_val)
            
            # Select top algorithms for heatmap
            top_algorithms = alg_performance.head(8).index
            heatmap_data = normalized_performance.loc[top_algorithms]
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                       ax=ax9, cbar_kws={'label': 'Normalized Performance'})
            ax9.set_title('Algorithm Performance Heatmap')
            ax9.set_ylabel('Algorithm')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'sipp_fcbs_benchmark_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results_df: pd.DataFrame, output_file: str = None) -> None:
        """
        Generate a detailed text report of benchmark results
        
        Args:
            results_df: DataFrame with benchmark results
            output_file: Optional output file path for the report
        """
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f'sipp_fcbs_benchmark_report_{timestamp}.txt'
        
        report_lines = []
        report_lines.append("SIPP F-CBS Moving AI Lab Benchmark Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Add all the analysis content
        # This would include the same analysis as in analyze_results but formatted for text output
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Detailed report saved to {output_file}")

# Example usage and testing functions
def run_quick_benchmark_test(benchmark_dir: str) -> pd.DataFrame:
    """
    Run a quick benchmark test with a few scenarios
    
    Args:
        benchmark_dir: Path to Moving AI Lab benchmark directory
        
    Returns:
        DataFrame with results
    """
    runner = MovingAIBenchmarkRunner(benchmark_dir)
    
    print("Running quick benchmark test...")
    results_df = runner.run_full_benchmark_suite(
        max_scenarios=100,  # Test only first 3 scenarios
        agent_counts=[5, 10, 15],  # Test fewer agent counts
        timeout_seconds=120  # Shorter timeout
    )
    
    if not results_df.empty:
        runner.analyze_results(results_df)
    
    return results_df

def run_comprehensive_benchmark(benchmark_dir: str) -> pd.DataFrame:
    """
    Run comprehensive benchmark suite
    
    Args:
        benchmark_dir: Path to Moving AI Lab benchmark directory
        
    Returns:
        DataFrame with results
    """
    runner = MovingAIBenchmarkRunner(benchmark_dir)
    
    print("Running comprehensive benchmark suite...")
    results_df = runner.run_full_benchmark_suite(
        max_scenarios=20,  # Test more scenarios
        agent_counts=[15, 17, 19, 21, 23, 25, 27, 29, 31],  # Full range
        timeout_seconds=600  # Longer timeout
    )
    
    if not results_df.empty:
        runner.analyze_results(results_df)
        runner.generate_report(results_df)
    
    return results_df

if __name__ == "__main__":
    import sys
    
    # Example usage
    print("SIPP F-CBS Benchmark Runner")
    print("=" * 30)
    
    # Check if benchmark directory exists
    possible_paths = [
        "./benchmarks",
        "../benchmarks", 
        "./movingai_benchmarks",
        "../movingai_benchmarks"
    ]
    
    benchmark_directory = None
    
    # Try to find benchmark directory automatically
    for path in possible_paths:
        if os.path.exists(path):
            map_files = [f for f in os.listdir(path) if f.endswith('.map')]
            if map_files:
                benchmark_directory = path
                print(f"Found benchmark directory: {benchmark_directory}")
                break
    
    if benchmark_directory is None:
        print("Benchmark directory not found!")
        print("\nTo set up benchmarks:")
        print("1. Download from https://movingai.com/benchmarks/mapf.html")
        print("2. Extract to a 'benchmarks' folder in this directory")
        print("3. Or update the benchmark_directory variable below")
        print("\nExpected structure:")
        print("benchmarks/")
        print("  ├── empty/")
        print("  │   ├── *.map")
        print("  │   └── *.scen")
        print("  └── random/")
        print("      ├── *.map")
        print("      └── *.scen")
        sys.exit(1)
    
    # Run benchmark
    print("\nChoose benchmark type:")
    print("1. Quick test (3 scenarios, 5-15 agents, 2min timeout)")
    print("2. Comprehensive (20 scenarios, 5-35 agents, 10min timeout)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nRunning quick benchmark test...")
        results = run_quick_benchmark_test(benchmark_directory)
    elif choice == "2":
        print("\nRunning comprehensive benchmark...")
        results = run_comprehensive_benchmark(benchmark_directory)
    else:
        print("Invalid choice. Running quick test by default...")
        results = run_quick_benchmark_test(benchmark_directory)