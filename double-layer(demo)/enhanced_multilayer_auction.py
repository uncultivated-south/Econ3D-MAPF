"""
Enhanced Multilayer Auction System with Emergency Priority
Handles multi-round bidding for regular agents in two-layer airspace
with emergency agent path reservations as immutable constraints.
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import random
import copy
import time

# Import from enhanced modules
from enhanced_multilayer_grid_system import (
    EnhancedMultilayerGridSystem, LayerType, AgentType, ProcessingPhase
)
from enhanced_astar_pathfinding import (
    EnhancedAStarPathfinder, PathfindingRequest, PathfindingResponse, PathfindingResult
)
from enhanced_cbs import (
    EnhancedCBS, CBSRequest, CBSResponse, CBSResult
)

class BiddingStrategy(Enum):
    """Bidding strategies for regular agents"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"

class AuctionResult(Enum):
    """Results of auction operations"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    NO_BIDDERS = "no_bidders"
    ALL_PATHS_BLOCKED = "all_paths_blocked"
    BUDGET_INSUFFICIENT = "budget_insufficient"
    TIMEOUT = "timeout"

@dataclass
class GridCellBid:
    """Represents a bid on a specific grid cell in space-time"""
    agent_id: int
    position: Tuple[int, int, int]  # (x, y, time)
    layer: LayerType
    bid_amount: float
    
    def __hash__(self):
        return hash((self.agent_id, self.position, self.layer))

@dataclass
class PathBid:
    """Represents a regular agent's bid on a complete path"""
    agent_id: int
    layer: LayerType
    path: List[Tuple[int, int]]  # Path positions (x, y)
    time_steps: List[int]  # Corresponding time steps
    total_bid_amount: float
    grid_cell_bids: List[GridCellBid]  # Breakdown by space-time cell
    budget_remaining: float
    strategy: BiddingStrategy
    emergency_conflicts: List[int] = field(default_factory=list)  # Emergency agents blocking path
    
    def __post_init__(self):
        # Validate path and time_steps alignment
        if len(self.path) != len(self.time_steps):
            raise ValueError("Path length must match time steps length")
        
        # Create space-time positions for validation
        spacetime_positions = set((x, y, t) for (x, y), t in zip(self.path, self.time_steps))
        bid_positions = set(bid.position for bid in self.grid_cell_bids)
        
        if spacetime_positions != bid_positions:
            raise ValueError("Grid cell bids don't match space-time path positions")

@dataclass
class AuctionRoundResult:
    """Result of a single auction round for a specific layer"""
    round_number: int
    layer: LayerType
    winners: Dict[int, PathBid]  # agent_id -> PathBid
    losers: Dict[int, PathBid]   # agent_id -> PathBid
    emergency_blocked: Dict[int, PathBid]  # agent_id -> PathBid (blocked by emergency)
    total_revenue: float
    conflicts_resolved: List[Tuple[int, int]]  # (winner_id, loser_id) pairs
    average_bid_per_agent: float
    
    def get_winner_paths(self) -> Dict[int, Tuple[List[Tuple[int, int]], List[int]]]:
        """Extract winner paths in standard format"""
        return {agent_id: (bid.path, bid.time_steps) for agent_id, bid in self.winners.items()}

@dataclass
class LayerAuctionResult:
    """Complete auction result for a single layer"""
    layer: LayerType
    result: AuctionResult
    total_rounds: int
    final_winners: Dict[int, Tuple[List[Tuple[int, int]], List[int]]]  # agent_id -> (path, time_steps)
    unassigned_agents: List[int]  # agents who didn't get paths
    emergency_blocked_agents: List[int]  # agents blocked by emergency reservations
    total_revenue: float
    round_results: List[AuctionRoundResult]
    computation_time: float
    failure_reason: str = ""
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this layer"""
        total_agents = len(self.final_winners) + len(self.unassigned_agents) + len(self.emergency_blocked_agents)
        return (len(self.final_winners) / total_agents * 100) if total_agents > 0 else 0.0

@dataclass
class MultilayerAuctionResult:
    """Complete auction result across both layers"""
    lower_layer_result: LayerAuctionResult
    upper_layer_result: LayerAuctionResult
    total_computation_time: float
    overall_success_rate: float
    total_revenue: float
    summary: Dict[str, Union[int, float]]

class EnhancedAuctionSystem:
    """
    Enhanced auction system for multilayer airspace with emergency priority.
    
    Only processes regular agents - emergency agents get direct path assignment.
    Respects emergency agent path reservations as immutable constraints.
    """
    
    def __init__(self, grid_system: EnhancedMultilayerGridSystem, 
                 pathfinder: EnhancedAStarPathfinder, cbs_solver: EnhancedCBS):
        """
        Initialize enhanced auction system.
        
        Args:
            grid_system: EnhancedMultilayerGridSystem instance
            pathfinder: EnhancedAStarPathfinder for path generation
            cbs_solver: EnhancedCBS for conflict resolution
        """
        self.grid_system = grid_system
        self.pathfinder = pathfinder
        self.cbs_solver = cbs_solver
        
        # Auction parameters
        self.max_rounds = 5
        self.round_timeout = 30.0  # seconds per round
        
        # Pricing parameters
        self.pricing_alpha = 0.1    # AvgBudget multiplier
        self.pricing_beta = 0.2     # ConflictDensity multiplier  
        self.pricing_gamma = 0.1    # Base price
        
        # Strategy parameters for regular agents
        self.conservative_increment = 0.1  # 10% increment
        self.aggressive_early_multiplier = 1.5  # 150% in early rounds
        self.balanced_increment = 0.3      # 30% increment
        
        # Performance tracking
        self.stats = {
            'total_auctions': 0,
            'successful_auctions': 0,
            'partial_success_auctions': 0,
            'total_revenue': 0.0,
            'average_rounds_per_layer': 0.0,
            'emergency_blockage_rate': 0.0,
            'strategy_performance': {
                BiddingStrategy.CONSERVATIVE: {'attempts': 0, 'successes': 0},
                BiddingStrategy.AGGRESSIVE: {'attempts': 0, 'successes': 0},
                BiddingStrategy.BALANCED: {'attempts': 0, 'successes': 0}
            }
        }
        
        self.debug_mode = False
    
    def calculate_conflict_density(self, layer: LayerType, agents: List[Dict]) -> Dict[Tuple[int, int], int]:
        """
        Calculate spatial conflict density for pricing.
        
        Args:
            layer: Layer to analyze
            agents: List of regular agent information
            
        Returns:
            Dict mapping (x, y) to conflict count
        """
        if not agents:
            return {}
        
        # Get paths for all agents using pathfinder
        agent_paths = {}
        for agent_info in agents:
            request = PathfindingRequest(
                agent_id=agent_info['id'],
                start=agent_info['start'],
                goal=agent_info['goal'],
                start_time=3,  # Regular agents start processing at t=3
                layer=layer,
                agent_type=AgentType.REGULAR,
                max_time_steps=100
            )
            
            response = self.pathfinder.find_path(request)
            if response.result == PathfindingResult.SUCCESS:
                agent_paths[agent_info['id']] = response.path
        
        # Count position usage
        position_usage = defaultdict(int)
        for path in agent_paths.values():
            for x, y in path:
                position_usage[(x, y)] += 1
        
        # Convert to conflict density (usage > 1 = conflict)
        conflict_density = {pos: max(0, count - 1) for pos, count in position_usage.items()}
        
        return conflict_density
    
    def calculate_starting_prices(self, conflict_density: Dict[Tuple[int, int], int],
                                agents: List[Dict]) -> Dict[Tuple[int, int], float]:
        """
        Calculate starting prices for spatial positions based on conflict and budgets.
        
        Args:
            conflict_density: Conflict count per (x, y) position
            agents: List of participating regular agents
            
        Returns:
            Dict mapping (x, y) to starting price
        """
        if not agents:
            return {}
        
        # Calculate average budget
        budgets = [agent['budget'] for agent in agents if agent['budget'] > 0]
        avg_budget = sum(budgets) / len(budgets) if budgets else 50.0
        
        starting_prices = {}
        
        # Calculate price for each position with conflicts
        for (x, y), conflict_count in conflict_density.items():
            if conflict_count > 0:
                price = (avg_budget * self.pricing_alpha) * (conflict_count * self.pricing_beta) + self.pricing_gamma
                starting_prices[(x, y)] = max(price, self.pricing_gamma)
        
        # Set base price for all positions
        for x in range(self.grid_system.width):
            for y in range(self.grid_system.height):
                if (x, y) not in starting_prices:
                    starting_prices[(x, y)] = self.pricing_gamma
        
        return starting_prices
    
    def calculate_bid_amount(self, agent_info: Dict, path: List[Tuple[int, int]], 
                           starting_prices: Dict[Tuple[int, int], float],
                           round_number: int, total_rounds: int) -> float:
        """
        Calculate bid amount based on agent's strategy.
        
        Args:
            agent_info: Agent information dictionary
            path: Path being bid on
            starting_prices: Starting prices for positions
            round_number: Current round (1-based)
            total_rounds: Total auction rounds
            
        Returns:
            Total bid amount
        """
        if not path:
            return 0.0
        
        # Calculate minimum bid (sum of starting prices)
        min_bid = sum(starting_prices.get((x, y), self.pricing_gamma) for x, y in path)
        
        strategy = BiddingStrategy(agent_info['strategy'])
        budget = agent_info['budget']
        
        if strategy == BiddingStrategy.CONSERVATIVE:
            # Conservative: minimal increment above starting price
            increment = min_bid * self.conservative_increment
            return min(min_bid + increment, budget)
        
        elif strategy == BiddingStrategy.AGGRESSIVE:
            # Aggressive: front-load high bids in early rounds
            round_factor = (total_rounds - round_number + 1) / total_rounds
            multiplier = 1.0 + (self.aggressive_early_multiplier - 1.0) * round_factor
            return min(min_bid * multiplier, budget)
        
        elif strategy == BiddingStrategy.BALANCED:
            # Balanced: consistent increments
            increment = min_bid * self.balanced_increment
            return min(min_bid + increment, budget)
        
        else:
            return min(min_bid, budget)
    
    def decompose_path_bid(self, agent_info: Dict, path: List[Tuple[int, int]], 
                          time_steps: List[int], total_bid: float,
                          starting_prices: Dict[Tuple[int, int], float], 
                          layer: LayerType) -> List[GridCellBid]:
        """
        Decompose a total path bid into individual space-time cell bids.
        
        Args:
            agent_info: Agent information
            path: Path positions
            time_steps: Corresponding time steps
            total_bid: Total bid amount
            starting_prices: Starting prices for positions
            layer: Layer for the path
            
        Returns:
            List of grid cell bids
        """
        if not path or len(path) != len(time_steps):
            return []
        
        # Calculate total starting price for the path
        total_starting_price = sum(starting_prices.get((x, y), self.pricing_gamma) 
                                 for x, y in path)
        
        if total_starting_price <= 0:
            return []
        
        # Distribute bid proportionally to starting prices
        grid_cell_bids = []
        for (x, y), t in zip(path, time_steps):
            cell_starting_price = starting_prices.get((x, y), self.pricing_gamma)
            cell_bid_amount = total_bid * (cell_starting_price / total_starting_price)
            
            grid_cell_bids.append(GridCellBid(
                agent_id=agent_info['id'],
                position=(x, y, t),
                layer=layer,
                bid_amount=cell_bid_amount
            ))
        
        return grid_cell_bids
    
    def generate_bids_for_round(self, agents: List[Dict], layer: LayerType,
                              starting_prices: Dict[Tuple[int, int], float],
                              round_number: int, start_time: int) -> List[PathBid]:
        """
        Generate path bids for all regular agents in current round.
        
        Args:
            agents: Regular agent information
            layer: Layer being processed
            starting_prices: Starting prices for positions
            round_number: Current round number
            start_time: Starting time for paths
            
        Returns:
            List of valid path bids
        """
        bids = []
        
        for agent_info in agents:
            # Generate path using enhanced pathfinder (considers emergency reservations)
            request = PathfindingRequest(
                agent_id=agent_info['id'],
                start=agent_info['start'],
                goal=agent_info['goal'],
                start_time=start_time,
                layer=layer,
                agent_type=AgentType.REGULAR,
                max_time_steps=100,
                allow_waiting=True
            )
            
            response = self.pathfinder.find_path(request)
            
            if response.result != PathfindingResult.SUCCESS:
                if self.debug_mode:
                    print(f"Agent {agent_info['id']} couldn't find path in round {round_number}: {response.message}")
                continue
            
            # Calculate bid amount based on strategy
            bid_amount = self.calculate_bid_amount(
                agent_info, response.path, starting_prices, round_number, self.max_rounds
            )
            
            # Check if agent can afford this bid
            if bid_amount > agent_info['budget']:
                if self.debug_mode:
                    print(f"Agent {agent_info['id']} can't afford bid of {bid_amount:.2f} (budget: {agent_info['budget']:.2f})")
                continue
            
            # Decompose into grid cell bids
            grid_cell_bids = self.decompose_path_bid(
                agent_info, response.path, response.time_steps, bid_amount,
                starting_prices, layer
            )
            
            if not grid_cell_bids:
                continue
            
            # Create path bid
            path_bid = PathBid(
                agent_id=agent_info['id'],
                layer=layer,
                path=response.path,
                time_steps=response.time_steps,
                total_bid_amount=bid_amount,
                grid_cell_bids=grid_cell_bids,
                budget_remaining=agent_info['budget'] - bid_amount,
                strategy=BiddingStrategy(agent_info['strategy']),
                emergency_conflicts=response.emergency_conflicts
            )
            
            bids.append(path_bid)
        
        return bids
    
    def select_winners_for_spacetime_cells(self, bids: List[PathBid]) -> Dict[Tuple[int, int, int], GridCellBid]:
        """
        Select highest bidder for each space-time cell.
        
        Args:
            bids: All path bids for this round
            
        Returns:
            Dict mapping (x, y, t) to winning cell bid
        """
        cell_bids = defaultdict(list)
        
        # Group bids by space-time cell
        for path_bid in bids:
            for cell_bid in path_bid.grid_cell_bids:
                cell_bids[cell_bid.position].append(cell_bid)
        
        # Select highest bidder for each cell
        winners = {}
        for position, competing_bids in cell_bids.items():
            if competing_bids:
                winner = max(competing_bids, key=lambda bid: bid.bid_amount)
                winners[position] = winner
        
        return winners
    
    def determine_path_winners(self, bids: List[PathBid]) -> Tuple[List[PathBid], List[PathBid], List[PathBid]]:
        """
        Determine which agents won their complete paths.
        
        Args:
            bids: All path bids for this round
            
        Returns:
            Tuple: (winners, losers, emergency_blocked)
        """
        # Select winners for individual space-time cells
        cell_winners = self.select_winners_for_spacetime_cells(bids)
        
        # Check which agents won all cells in their path
        winners = []
        losers = []
        emergency_blocked = []
        
        for path_bid in bids:
            # Check if agent has emergency conflicts
            if path_bid.emergency_conflicts:
                emergency_blocked.append(path_bid)
                continue
            
            # Check if agent won all required cells
            won_all_cells = True
            for cell_bid in path_bid.grid_cell_bids:
                cell_winner = cell_winners.get(cell_bid.position)
                if not cell_winner or cell_winner.agent_id != path_bid.agent_id:
                    won_all_cells = False
                    break
            
            if won_all_cells:
                winners.append(path_bid)
            else:
                losers.append(path_bid)
        
        return winners, losers, emergency_blocked
    
    def resolve_winner_conflicts_with_cbs(self, winners: List[PathBid], layer: LayerType) -> List[PathBid]:
        """
        Use CBS to resolve conflicts among auction winners.
        
        Args:
            winners: List of auction winners
            layer: Layer being processed
            
        Returns:
            List of conflict-free winners
        """
        if not winners:
            return []
        
        # Convert winners to agent information format for CBS
        winner_agents = []
        for winner in winners:
            winner_agents.append({
                'id': winner.agent_id,
                'start': winner.path[0] if winner.path else (0, 0),
                'goal': winner.path[-1] if winner.path else (0, 0),
                'current_pos': winner.path[0] if winner.path else (0, 0),
                'agent_type': AgentType.REGULAR,
                'budget': winner.budget_remaining,
                'strategy': winner.strategy.value,
                'is_emergency': False
            })
        
        # Run CBS on winners
        cbs_response = self.cbs_solver.solve_regular_agents(layer, winner_agents, 3)
        
        # Return only agents that CBS validated
        if cbs_response.result in [CBSResult.SUCCESS, CBSResult.PARTIAL_SUCCESS]:
            validated_winners = []
            for winner in winners:
                if winner.agent_id in cbs_response.solution:
                    validated_winners.append(winner)
            return validated_winners
        
        return []
    
    def update_agent_budgets(self, winners: List[PathBid], agent_dict: Dict[int, Dict]) -> None:
        """
        Deduct bid amounts from winning agents' budgets.
        
        Args:
            winners: List of winning path bids
            agent_dict: Dictionary of agent information by ID
        """
        for winner in winners:
            if winner.agent_id in agent_dict:
                agent_dict[winner.agent_id]['budget'] -= winner.total_bid_amount
                agent_dict[winner.agent_id]['budget'] = max(0.0, agent_dict[winner.agent_id]['budget'])
    
    def run_auction_round(self, agents: List[Dict], layer: LayerType, round_number: int,
                         conflict_density: Dict[Tuple[int, int], int],
                         start_time: int) -> AuctionRoundResult:
        """
        Run a single auction round for a specific layer.
        
        Args:
            agents: Regular agents participating in this round
            layer: Layer being processed
            round_number: Current round number
            conflict_density: Conflict density for pricing
            start_time: Starting time for paths
            
        Returns:
            Result of this auction round
        """
        round_start_time = time.time()
        
        # Calculate starting prices
        starting_prices = self.calculate_starting_prices(conflict_density, agents)
        
        if self.debug_mode:
            print(f"\nLayer {layer.name}, Round {round_number}")
            print(f"Agents: {len(agents)}, Starting prices calculated")
        
        # Generate bids from all agents
        bids = self.generate_bids_for_round(agents, layer, starting_prices, round_number, start_time)
        
        if not bids:
            return AuctionRoundResult(
                round_number=round_number,
                layer=layer,
                winners={},
                losers={},
                emergency_blocked={},
                total_revenue=0.0,
                conflicts_resolved=[],
                average_bid_per_agent=0.0
            )
        
        # Determine path winners, losers, and emergency blocked
        initial_winners, losers, emergency_blocked = self.determine_path_winners(bids)
        
        if self.debug_mode:
            print(f"Initial winners: {len(initial_winners)}, Losers: {len(losers)}, Emergency blocked: {len(emergency_blocked)}")
        
        # Resolve conflicts among winners using CBS
        final_winners = self.resolve_winner_conflicts_with_cbs(initial_winners, layer)
        
        # Update losers (include initial winners who failed CBS validation)
        for initial_winner in initial_winners:
            if initial_winner not in final_winners:
                losers.append(initial_winner)
        
        # Calculate metrics
        total_revenue = sum(winner.total_bid_amount for winner in final_winners)
        conflicts_resolved = [(w.agent_id, l.agent_id) for w in final_winners for l in initial_winners 
                             if w != l and l not in final_winners]
        avg_bid = sum(bid.total_bid_amount for bid in bids) / len(bids) if bids else 0.0
        
        # Check round timeout
        if time.time() - round_start_time > self.round_timeout:
            if self.debug_mode:
                print(f"Round {round_number} timed out")
        
        return AuctionRoundResult(
            round_number=round_number,
            layer=layer,
            winners={winner.agent_id: winner for winner in final_winners},
            losers={loser.agent_id: loser for loser in losers},
            emergency_blocked={blocked.agent_id: blocked for blocked in emergency_blocked},
            total_revenue=total_revenue,
            conflicts_resolved=conflicts_resolved,
            average_bid_per_agent=avg_bid
        )
    
    def run_layer_auction(self, layer: LayerType, agents: List[Dict], start_time: int) -> LayerAuctionResult:
        """
        Run complete auction for a specific layer.
        
        Args:
            layer: Layer to process
            agents: Regular agents for this layer
            start_time: Starting time for paths
            
        Returns:
            Complete layer auction result
        """
        layer_start_time = time.time()
        
        if not agents:
            return LayerAuctionResult(
                layer=layer,
                result=AuctionResult.NO_BIDDERS,
                total_rounds=0,
                final_winners={},
                unassigned_agents=[],
                emergency_blocked_agents=[],
                total_revenue=0.0,
                round_results=[],
                computation_time=0.0,
                failure_reason="No agents provided for auction"
            )
        
        # Calculate initial conflict density
        conflict_density = self.calculate_conflict_density(layer, agents)
        
        # Initialize auction state
        remaining_agents = agents.copy()
        agent_dict = {agent['id']: agent for agent in agents}
        round_results = []
        final_winners = {}
        total_revenue = 0.0
        emergency_blocked_agents = []
        
        if self.debug_mode:
            print(f"\nStarting {layer.name} layer auction with {len(agents)} regular agents")
        
        # Run auction rounds
        for round_num in range(1, self.max_rounds + 1):
            if not remaining_agents:
                break
            
            # Update conflict density based on current remaining agents
            current_conflict_density = self.calculate_conflict_density(layer, remaining_agents)
            
            # Run this round
            round_result = self.run_auction_round(
                remaining_agents, layer, round_num, current_conflict_density, start_time
            )
            
            round_results.append(round_result)
            total_revenue += round_result.total_revenue
            
            # Process winners
            if round_result.winners:
                # Update budgets
                self.update_agent_budgets(list(round_result.winners.values()), agent_dict)
                
                # Add to final winners
                for agent_id, winner_bid in round_result.winners.items():
                    final_winners[agent_id] = (winner_bid.path, winner_bid.time_steps)
                
                # Remove winners from remaining agents
                remaining_agents = [agent for agent in remaining_agents 
                                  if agent['id'] not in round_result.winners]
            
            # Track emergency blocked agents
            if round_result.emergency_blocked:
                for agent_id in round_result.emergency_blocked:
                    if agent_id not in emergency_blocked_agents:
                        emergency_blocked_agents.append(agent_id)
                
                # Remove emergency blocked agents from remaining
                remaining_agents = [agent for agent in remaining_agents 
                                  if agent['id'] not in round_result.emergency_blocked]
        
        # Determine final result
        unassigned_agents = [agent['id'] for agent in remaining_agents]
        
        if final_winners:
            result = AuctionResult.SUCCESS if not unassigned_agents else AuctionResult.PARTIAL_SUCCESS
        elif emergency_blocked_agents:
            result = AuctionResult.ALL_PATHS_BLOCKED
        else:
            result = AuctionResult.BUDGET_INSUFFICIENT
        
        return LayerAuctionResult(
            layer=layer,
            result=result,
            total_rounds=len(round_results),
            final_winners=final_winners,
            unassigned_agents=unassigned_agents,
            emergency_blocked_agents=emergency_blocked_agents,
            total_revenue=total_revenue,
            round_results=round_results,
            computation_time=time.time() - layer_start_time,
            failure_reason="" if result == AuctionResult.SUCCESS else f"Layer auction result: {result.value}"
        )
    
    def run_multilayer_auction(self, start_time: int = 3) -> MultilayerAuctionResult:
        """
        Run auction across both layers for all regular agents.
        
        Args:
            start_time: Starting time for regular agent processing
            
        Returns:
            Complete multilayer auction result
        """
        auction_start_time = time.time()
        self.stats['total_auctions'] += 1
        
        # Set processing phase
        self.grid_system.set_processing_phase(ProcessingPhase.REGULAR_AUCTION)
        
        # Get regular agents for each layer
        lower_agents = self.grid_system.get_regular_agents_for_processing(LayerType.LOWER)
        upper_agents = self.grid_system.get_regular_agents_for_processing(LayerType.UPPER)
        
        if self.debug_mode:
            print(f"\nStarting multilayer auction")
            print(f"Lower layer: {len(lower_agents)} agents")
            print(f"Upper layer: {len(upper_agents)} agents")
        
        # Run auctions for both layers
        lower_result = self.run_layer_auction(LayerType.LOWER, lower_agents, start_time)
        upper_result = self.run_layer_auction(LayerType.UPPER, upper_agents, start_time)
        
        # Update grid system with winning paths
        if lower_result.final_winners:
            winner_paths = {aid: path for aid, (path, time_steps) in lower_result.final_winners.items()}
            self.grid_system.update_regular_agent_paths(LayerType.LOWER, winner_paths)
        
        if upper_result.final_winners:
            winner_paths = {aid: path for aid, (path, time_steps) in upper_result.final_winners.items()}
            self.grid_system.update_regular_agent_paths(LayerType.UPPER, winner_paths)
        
        # Calculate overall metrics
        total_agents = len(lower_agents) + len(upper_agents)
        total_winners = len(lower_result.final_winners) + len(upper_result.final_winners)
        overall_success_rate = (total_winners / total_agents * 100) if total_agents > 0 else 0.0
        
        total_computation_time = time.time() - auction_start_time
        total_revenue = lower_result.total_revenue + upper_result.total_revenue
        
        # Update statistics
        self.stats['total_revenue'] += total_revenue
        if overall_success_rate == 100.0:
            self.stats['successful_auctions'] += 1
        elif total_winners > 0:
            self.stats['partial_success_auctions'] += 1
        
        # Calculate strategy performance
        for layer_result in [lower_result, upper_result]:
            for round_result in layer_result.round_results:
                for winner_bid in round_result.winners.values():
                    strategy = winner_bid.strategy
                    self.stats['strategy_performance'][strategy]['attempts'] += 1
                    self.stats['strategy_performance'][strategy]['successes'] += 1
        
        # Create summary
        summary = {
            'total_agents': total_agents,
            'total_winners': total_winners,
            'total_unassigned': len(lower_result.unassigned_agents) + len(upper_result.unassigned_agents),
            'total_emergency_blocked': len(lower_result.emergency_blocked_agents) + len(upper_result.emergency_blocked_agents),
            'lower_layer_success_rate': lower_result.get_success_rate(),
            'upper_layer_success_rate': upper_result.get_success_rate(),
            'total_rounds_used': lower_result.total_rounds + upper_result.total_rounds
        }
        
        return MultilayerAuctionResult(
            lower_layer_result=lower_result,
            upper_layer_result=upper_result,
            total_computation_time=total_computation_time,
            overall_success_rate=overall_success_rate,
            total_revenue=total_revenue,
            summary=summary
        )
    
    def get_performance_stats(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Get comprehensive auction system performance statistics.
        
        Returns:
            Dict: Performance metrics and strategy analysis
        """
        total_attempts = self.stats['total_auctions']
        if total_attempts == 0:
            return self.stats.copy()
        
        # Calculate success rates
        success_rate = (self.stats['successful_auctions'] / total_attempts) * 100
        partial_success_rate = (self.stats['partial_success_auctions'] / total_attempts) * 100
        combined_success_rate = ((self.stats['successful_auctions'] + self.stats['partial_success_auctions']) / total_attempts) * 100
        
        # Calculate strategy performance rates
        strategy_stats = {}
        for strategy, perf in self.stats['strategy_performance'].items():
            if perf['attempts'] > 0:
                success_rate_strategy = (perf['successes'] / perf['attempts']) * 100
                strategy_stats[strategy.value] = {
                    'attempts': perf['attempts'],
                    'successes': perf['successes'],
                    'success_rate_percent': success_rate_strategy
                }
            else:
                strategy_stats[strategy.value] = {
                    'attempts': 0,
                    'successes': 0,
                    'success_rate_percent': 0.0
                }
        
        return {
            'total_auctions': total_attempts,
            'successful_auctions': self.stats['successful_auctions'],
            'partial_success_auctions': self.stats['partial_success_auctions'],
            'success_rate_percent': success_rate,
            'partial_success_rate_percent': partial_success_rate,
            'combined_success_rate_percent': combined_success_rate,
            'total_revenue': self.stats['total_revenue'],
            'average_revenue_per_auction': self.stats['total_revenue'] / total_attempts,
            'average_rounds_per_layer': self.stats['average_rounds_per_layer'],
            'emergency_blockage_rate_percent': self.stats['emergency_blockage_rate'],
            'strategy_performance': strategy_stats
        }
    
    def set_auction_parameters(self, max_rounds: int = 5, round_timeout: float = 30.0,
                             pricing_alpha: float = 0.1, pricing_beta: float = 0.2,
                             pricing_gamma: float = 0.1) -> None:
        """
        Update auction system parameters.
        
        Args:
            max_rounds: Maximum auction rounds per layer
            round_timeout: Timeout per round in seconds
            pricing_alpha: Average budget multiplier for pricing
            pricing_beta: Conflict density multiplier for pricing
            pricing_gamma: Base price for all positions
        """
        self.max_rounds = max_rounds
        self.round_timeout = round_timeout
        self.pricing_alpha = pricing_alpha
        self.pricing_beta = pricing_beta
        self.pricing_gamma = pricing_gamma
    
    def set_strategy_parameters(self, conservative_increment: float = 0.1,
                              aggressive_multiplier: float = 1.5,
                              balanced_increment: float = 0.3) -> None:
        """
        Update bidding strategy parameters.
        
        Args:
            conservative_increment: Increment rate for conservative strategy
            aggressive_multiplier: Early round multiplier for aggressive strategy
            balanced_increment: Increment rate for balanced strategy
        """
        self.conservative_increment = conservative_increment
        self.aggressive_early_multiplier = aggressive_multiplier
        self.balanced_increment = balanced_increment
    
    def reset_statistics(self) -> None:
        """Reset all auction statistics."""
        self.stats = {
            'total_auctions': 0,
            'successful_auctions': 0,
            'partial_success_auctions': 0,
            'total_revenue': 0.0,
            'average_rounds_per_layer': 0.0,
            'emergency_blockage_rate': 0.0,
            'strategy_performance': {
                BiddingStrategy.CONSERVATIVE: {'attempts': 0, 'successes': 0},
                BiddingStrategy.AGGRESSIVE: {'attempts': 0, 'successes': 0},
                BiddingStrategy.BALANCED: {'attempts': 0, 'successes': 0}
            }
        }
    
    def set_debug_mode(self, enabled: bool) -> None:
        """Enable or disable debug output."""
        self.debug_mode = enabled
    
    def analyze_layer_competition(self, layer: LayerType) -> Dict[str, Union[int, float, List]]:
        """
        Analyze competition metrics for a specific layer.
        
        Args:
            layer: Layer to analyze
            
        Returns:
            Dict: Competition analysis including density and blockage info
        """
        agents = self.grid_system.get_regular_agents_for_processing(layer)
        if not agents:
            return {
                'layer': layer.name,
                'total_agents': 0,
                'conflict_density': {},
                'emergency_blockage_positions': [],
                'avg_path_length': 0.0,
                'competition_score': 0.0
            }
        
        # Calculate conflict density
        conflict_density = self.calculate_conflict_density(layer, agents)
        
        # Get emergency blockage information
        emergency_obstacles = self.grid_system.get_emergency_dynamic_obstacles(layer)
        emergency_positions = [(x, y) for x, y, t in emergency_obstacles.keys()]
        unique_emergency_positions = list(set(emergency_positions))
        
        # Calculate average path lengths
        total_path_length = 0
        valid_paths = 0
        for agent_info in agents:
            request = PathfindingRequest(
                agent_id=agent_info['id'],
                start=agent_info['start'],
                goal=agent_info['goal'],
                start_time=3,
                layer=layer,
                agent_type=AgentType.REGULAR,
                max_time_steps=100
            )
            response = self.pathfinder.find_path(request)
            if response.result == PathfindingResult.SUCCESS:
                total_path_length += len(response.path)
                valid_paths += 1
        
        avg_path_length = (total_path_length / valid_paths) if valid_paths > 0 else 0.0
        
        # Calculate competition score (higher = more competitive)
        total_conflicts = sum(conflict_density.values())
        competition_score = (total_conflicts / len(agents)) if agents else 0.0
        
        return {
            'layer': layer.name,
            'total_agents': len(agents),
            'conflict_density': dict(conflict_density),
            'total_conflicts': total_conflicts,
            'emergency_blockage_positions': unique_emergency_positions,
            'emergency_blockage_count': len(unique_emergency_positions),
            'avg_path_length': avg_path_length,
            'competition_score': competition_score,
            'agents_with_valid_paths': valid_paths
        }
    
    def export_auction_summary(self, result: MultilayerAuctionResult) -> Dict[str, Union[str, int, float, Dict, List]]:
        """
        Export comprehensive auction result summary.
        
        Args:
            result: Multilayer auction result to summarize
            
        Returns:
            Dict: Detailed summary for analysis or reporting
        """
        # Calculate per-layer summaries
        lower_summary = {
            'result': result.lower_layer_result.result.value,
            'winners': len(result.lower_layer_result.final_winners),
            'unassigned': len(result.lower_layer_result.unassigned_agents),
            'emergency_blocked': len(result.lower_layer_result.emergency_blocked_agents),
            'revenue': result.lower_layer_result.total_revenue,
            'rounds': result.lower_layer_result.total_rounds,
            'computation_time': result.lower_layer_result.computation_time,
            'success_rate': result.lower_layer_result.get_success_rate()
        }
        
        upper_summary = {
            'result': result.upper_layer_result.result.value,
            'winners': len(result.upper_layer_result.final_winners),
            'unassigned': len(result.upper_layer_result.unassigned_agents),
            'emergency_blocked': len(result.upper_layer_result.emergency_blocked_agents),
            'revenue': result.upper_layer_result.total_revenue,
            'rounds': result.upper_layer_result.total_rounds,
            'computation_time': result.upper_layer_result.computation_time,
            'success_rate': result.upper_layer_result.get_success_rate()
        }
        
        # Extract round-by-round details
        lower_rounds = []
        for round_result in result.lower_layer_result.round_results:
            lower_rounds.append({
                'round': round_result.round_number,
                'winners': len(round_result.winners),
                'losers': len(round_result.losers),
                'emergency_blocked': len(round_result.emergency_blocked),
                'revenue': round_result.total_revenue,
                'avg_bid': round_result.average_bid_per_agent
            })
        
        upper_rounds = []
        for round_result in result.upper_layer_result.round_results:
            upper_rounds.append({
                'round': round_result.round_number,
                'winners': len(round_result.winners),
                'losers': len(round_result.losers),
                'emergency_blocked': len(round_result.emergency_blocked),
                'revenue': round_result.total_revenue,
                'avg_bid': round_result.average_bid_per_agent
            })
        
        return {
            'overall_success_rate': result.overall_success_rate,
            'total_computation_time': result.total_computation_time,
            'total_revenue': result.total_revenue,
            'summary': result.summary,
            'lower_layer': lower_summary,
            'upper_layer': upper_summary,
            'lower_layer_rounds': lower_rounds,
            'upper_layer_rounds': upper_rounds,
            'system_performance': self.get_performance_stats()
        }

# Utility functions for integration and testing

def create_enhanced_auction_system(grid_system: EnhancedMultilayerGridSystem,
                                 pathfinder: EnhancedAStarPathfinder,
                                 cbs_solver: EnhancedCBS) -> EnhancedAuctionSystem:
    """
    Create enhanced auction system with all required components.
    
    Args:
        grid_system: Enhanced multilayer grid system
        pathfinder: Enhanced A* pathfinder
        cbs_solver: Enhanced CBS solver
        
    Returns:
        EnhancedAuctionSystem: Ready-to-use auction system
    """
    return EnhancedAuctionSystem(grid_system, pathfinder, cbs_solver)

def prepare_regular_agents_for_auction(grid_system: EnhancedMultilayerGridSystem,
                                     budget_range: Tuple[float, float] = (10.0, 100.0),
                                     strategy_distribution: Optional[Dict[str, float]] = None) -> Dict[LayerType, List[Dict]]:
    """
    Prepare regular agents for auction with budgets and strategies.
    
    Args:
        grid_system: Grid system containing agents
        budget_range: (min, max) budget range for random assignment
        strategy_distribution: Optional strategy distribution weights
        
    Returns:
        Dict mapping LayerType to list of agent information
    """
    if strategy_distribution is None:
        strategy_distribution = {
            'conservative': 0.3,
            'aggressive': 0.3,
            'balanced': 0.4
        }
    
    layer_agents = {LayerType.LOWER: [], LayerType.UPPER: []}
    
    for agent_id, agent_data in grid_system.regular_agents.items():
        # Assign random budget if not set
        if agent_data['budget'] <= 0:
            agent_data['budget'] = random.uniform(budget_range[0], budget_range[1])
        
        # Assign strategy based on distribution
        if agent_data['strategy'] not in strategy_distribution:
            rand_val = random.random()
            cumsum = 0
            for strategy, weight in strategy_distribution.items():
                cumsum += weight
                if rand_val <= cumsum:
                    agent_data['strategy'] = strategy
                    break
        
        # Add to appropriate layer
        layer = grid_system.layer_assignments[agent_id]
        layer_agents[layer].append({
            'id': agent_id,
            'start': agent_data['start'],
            'goal': agent_data['goal'],
            'budget': agent_data['budget'],
            'strategy': agent_data['strategy'],
            'agent_type': AgentType.REGULAR,
            'current_pos': agent_data.get('current_pos'),
            'is_emergency': False
        })
    
    return layer_agents

def run_auction_analysis(auction_system: EnhancedAuctionSystem, num_runs: int = 10) -> Dict[str, Union[int, float, List]]:
    """
    Run multiple auction simulations for statistical analysis.
    
    Args:
        auction_system: Configured auction system
        num_runs: Number of auction runs to perform
        
    Returns:
        Dict: Statistical analysis of auction performance
    """
    results = []
    
    for run_id in range(num_runs):
        result = auction_system.run_multilayer_auction()
        results.append({
            'run_id': run_id,
            'overall_success_rate': result.overall_success_rate,
            'total_revenue': result.total_revenue,
            'computation_time': result.total_computation_time,
            'lower_winners': len(result.lower_layer_result.final_winners),
            'upper_winners': len(result.upper_layer_result.final_winners),
            'total_rounds': result.lower_layer_result.total_rounds + result.upper_layer_result.total_rounds
        })
    
    # Calculate statistics
    success_rates = [r['overall_success_rate'] for r in results]
    revenues = [r['total_revenue'] for r in results]
    computation_times = [r['computation_time'] for r in results]
    
    return {
        'num_runs': num_runs,
        'avg_success_rate': sum(success_rates) / len(success_rates),
        'min_success_rate': min(success_rates),
        'max_success_rate': max(success_rates),
        'avg_revenue': sum(revenues) / len(revenues),
        'min_revenue': min(revenues),
        'max_revenue': max(revenues),
        'avg_computation_time': sum(computation_times) / len(computation_times),
        'detailed_results': results
    }