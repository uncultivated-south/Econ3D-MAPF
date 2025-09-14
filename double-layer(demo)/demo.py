"""
Demonstration of Double-Layer Airspace System
Shows how to use the system with sample aircraft scenarios
"""

from double_layer_airspace import (
    create_double_layer_system, DoubleLayerAirspaceSystem, 
    FlightLevel, analyze_layer_separation_benefit, validate_double_layer_paths
)
from grid_system import Agent, AgentType
from astar_pathfinding import PathfindingConfig, HeuristicType
import logging

# Setup logging for demo
logging.basicConfig(level=logging.INFO)

def create_sample_agents():
    """Create sample agents with different flight patterns"""
    agents = []
    
    # Eastward flights (should go to lower layer)
    agents.extend([
        Agent(id=1, agent_type=AgentType.NON_EMERGENCY, 
              start=(1, 5), goal=(8, 5), budget=50.0, strategy="balanced"),
        Agent(id=2, agent_type=AgentType.NON_EMERGENCY,
              start=(2, 3), goal=(7, 4), budget=60.0, strategy="aggressive"),
    ])
    
    # Westward flights (should go to higher layer)
    agents.extend([
        Agent(id=3, agent_type=AgentType.NON_EMERGENCY,
              start=(8, 2), goal=(2, 3), budget=45.0, strategy="conservative"),
        Agent(id=4, agent_type=AgentType.NON_EMERGENCY,
              start=(9, 6), goal=(1, 6), budget=70.0, strategy="balanced"),
    ])
    
    # Northbound flights (should go to higher layer)
    agents.extend([
        Agent(id=5, agent_type=AgentType.NON_EMERGENCY,
              start=(4, 1), goal=(4, 8), budget=55.0, strategy="adaptive"),
    ])
    
    # Southbound flights (should go to lower layer)
    agents.extend([
        Agent(id=6, agent_type=AgentType.NON_EMERGENCY,
              start=(6, 8), goal=(6, 2), budget=40.0, strategy="conservative"),
        Agent(id=7, agent_type=AgentType.NON_EMERGENCY,
              start=(4,2), goal=(8, 7), budget=62.0, strategy="aggressive"),
        Agent(id=8, agent_type=AgentType.NON_EMERGENCY,
              start=(1, 8), goal=(3, 2), budget=55.0, strategy="conservative"),
        Agent(id=9, agent_type=AgentType.NON_EMERGENCY,
              start=(6, 8), goal=(6, 2), budget=80.0, strategy="balanced"),
        Agent(id=10, agent_type=AgentType.NON_EMERGENCY,
              start=(7, 8), goal=(2, 1), budget=20.0, strategy="conservative"),
        Agent(id=11, agent_type=AgentType.NON_EMERGENCY,
              start=(1, 8), goal=(6, 3), budget=70.0, strategy="conservative"),
        Agent(id=12, agent_type=AgentType.NON_EMERGENCY,
              start=(3, 8), goal=(8, 3), budget=52.0, strategy="conservative"),
        Agent(id=13, agent_type=AgentType.NON_EMERGENCY,
              start=(1, 1), goal=(8, 7), budget=60.0, strategy="conservative"),
        Agent(id=14, agent_type=AgentType.NON_EMERGENCY,
              start=(2, 1), goal=(6, 8), budget=66.0, strategy="conservative"),
        Agent(id=15, agent_type=AgentType.NON_EMERGENCY,
              start=(2, 2), goal=(5, 5), budget=46.0, strategy="conservative"),
    ])
    
    # Emergency agent (eastward, but gets priority)
    agents.append(
        Agent(id=100, agent_type=AgentType.EMERGENCY,
              start=(0, 4), goal=(9, 4), budget=0.0, priority=100)
    )
    
    return agents

def demonstrate_layer_assignment():
    """Demonstrate the flight level assignment system"""
    print("\n=== FLIGHT LEVEL ASSIGNMENT DEMONSTRATION ===")
    
    agents = create_sample_agents()
    
    # Create the assignment system
    from double_layer_airspace import FlightLevelAssignmentSystem
    assignment_system = FlightLevelAssignmentSystem()
    
    # Show assignments
    print("\nAgent Layer Assignments:")
    print("-" * 50)
    
    agents_by_layer = assignment_system.assign_agents_to_layers(agents)
    
    for agent in agents:
        assignment = assignment_system.assign_flight_level(agent)
        direction = assignment_system.determine_primary_direction(agent)
        
        agent_type = "EMERGENCY" if agent.is_emergency() else "REGULAR"
        print(f"Agent {agent.id:2d} ({agent_type:9s}): {agent.start} → {agent.goal}")
        print(f"           Direction: {direction.value:10s} → Layer: {assignment.assigned_layer.name}")
        print(f"           Rationale: {assignment.rationale}")
        print()
    
    print(f"Lower Layer: {len(agents_by_layer[FlightLevel.LOWER])} agents")
    print(f"Higher Layer: {len(agents_by_layer[FlightLevel.HIGHER])} agents")

def demonstrate_full_system():
    """Demonstrate the complete double-layer coordination"""
    print("\n=== FULL SYSTEM DEMONSTRATION ===")
    
    # Create system
    system = create_double_layer_system(
        width=10, 
        height=10, 
        max_time=50,
        pathfinder_config=PathfindingConfig(
            heuristic_type=HeuristicType.MANHATTAN,
            max_nodes_expanded=5000,
            max_time_seconds=10.0
        )
    )
    
    system.set_debug_mode(True)
    
    # Create agents
    agents = create_sample_agents()
    
    print(f"\nCoordinating {len(agents)} agents...")
    print(f"Emergency agents: {len([a for a in agents if a.is_emergency()])}")
    print(f"Regular agents: {len([a for a in agents if not a.is_emergency()])}")
    
    # Run coordination
    result = system.coordinate_agents(agents)
    
    print(f"\n=== COORDINATION RESULTS ===")
    print(f"Success: {result.success}")
    print(f"Total Duration: {result.total_duration:.2f} seconds")
    print(f"Agents with Paths: {len(result.agent_paths)}")
    print(f"Unassigned Agents: {len(result.unassigned_agents)}")
    
    if result.failure_reason:
        print(f"Failure Reason: {result.failure_reason}")
    
    print(f"\nTiming Breakdown:")
    print(f"  Assignment: {result.assignment_time:.3f}s")
    print(f"  Lower Layer: {result.lower_layer_time:.3f}s") 
    print(f"  Higher Layer: {result.higher_layer_time:.3f}s")
    print(f"  Integration: {result.integration_time:.3f}s")
    
    print(f"\nAgents by Layer:")
    for layer, count in result.agents_by_layer.items():
        print(f"  {layer.name}: {count} agents")
    
    # Show some example paths
    print(f"\n=== EXAMPLE PATHS ===")
    for agent_id in sorted(list(result.agent_paths.keys())[:3]):  # Show first 3
        path = result.agent_paths[agent_id]
        layer = result.layer_assignments.get(agent_id, FlightLevel.GROUND)
        
        print(f"\nAgent {agent_id} (Layer: {layer.name}):")
        print(f"  Path length: {len(path)} steps")
        
        # Show first few positions to demonstrate vertical movement protocol
        for i, pos in enumerate(path[:5]):
            if i == 0:
                print(f"  t={pos.t}: ({pos.x},{pos.y}) [GROUND - Takeoff]")
            elif i == 1:
                print(f"  t={pos.t}: ({pos.x},{pos.y}) [LOWER LAYER - Ascent]")
            elif i == 2:
                if layer == FlightLevel.HIGHER:
                    print(f"  t={pos.t}: ({pos.x},{pos.y}) [HIGHER LAYER - Final ascent]")
                else:
                    print(f"  t={pos.t}: ({pos.x},{pos.y}) [LOWER LAYER - Remain]")
            else:
                print(f"  t={pos.t}: ({pos.x},{pos.y}) [Horizontal movement]")
        
        if len(path) > 5:
            print(f"  ... {len(path)-5} more steps")
    
    # Validate paths
    validation = validate_double_layer_paths(result)
    print(f"\n=== PATH VALIDATION ===")
    print(f"Valid paths: {len(validation['valid_paths'])}")
    print(f"Invalid paths: {len(validation['invalid_paths'])}")
    
    if validation['protocol_violations']:
        print(f"Protocol violations found:")
        for violation in validation['protocol_violations'][:3]:  # Show first 3
            print(f"  - {violation}")
    
    return result

def demonstrate_conflict_analysis():
    """Demonstrate analysis of layer separation benefits"""
    print("\n=== CONFLICT ANALYSIS DEMONSTRATION ===")
    
    # Simulate some conflict numbers (in real usage these would come from actual runs)
    original_conflicts = 15  # Conflicts if all agents were in single layer
    lower_layer_conflicts = 3  # Conflicts in lower layer only
    higher_layer_conflicts = 2  # Conflicts in higher layer only
    
    analysis = analyze_layer_separation_benefit(
        original_conflicts, lower_layer_conflicts, higher_layer_conflicts
    )
    
    print(f"Conflict Analysis:")
    print(f"  Original (single layer): {analysis['original_conflicts']} conflicts")
    print(f"  Lower layer: {analysis['lower_layer_conflicts']} conflicts")  
    print(f"  Higher layer: {analysis['higher_layer_conflicts']} conflicts")
    print(f"  Total after separation: {analysis['total_layered_conflicts']} conflicts")
    print(f"  Conflicts eliminated: {analysis['conflicts_eliminated']}")
    print(f"  Reduction: {analysis['reduction_percentage']:.1f}%")
    print(f"  Separation effective: {analysis['separation_effective']}")

def main():
    """Run all demonstrations"""
    print("DOUBLE-LAYER AIRSPACE SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_layer_assignment()
        result = demonstrate_full_system()
        demonstrate_conflict_analysis()
        
        print(f"\n=== DEMONSTRATION COMPLETE ===")
        print(f"The double-layer system successfully:")
        print(f"  ✓ Assigned agents to appropriate layers")
        print(f"  ✓ Coordinated pathfinding within each layer")
        print(f"  ✓ Applied vertical movement protocol")
        print(f"  ✓ Integrated results into complete paths")
        
        if result.success:
            print(f"  ✓ All coordination completed successfully")
        else:
            print(f"  ! Some agents could not be assigned paths")
    
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()