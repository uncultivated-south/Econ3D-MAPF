"""
Auction System Module for UrbanAirspaceSim
Handles multi-round bidding with CBS validation and conflict resolution
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import random
import copy

# Import from previous modules
from grid_system import Position, Agent, AgentType, GridSystem
from astar_pathfinding import AStarPathfinder, extract_emergency_paths_from_grid
from cbs_module import ConflictBasedSearch, CBSResult, paths_to_tuples, tuples_to_paths

class BiddingStrategy(Enum):
    """Bidding strategies for agents"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"

@dataclass
class GridCellBid:
    """Represents a bid on a specific grid cell"""
    agent_id: int
    position: Tuple[int, int, int]  # (x, y, t)
    bid_amount: float
    
    def __hash__(self):
        return hash((self.agent_id, self.position))

@dataclass
class PathBid:
    """Represents an agent's bid on a complete path"""
    agent_id: int
    path: List[Tuple[int, int, int]]  # Complete path as (x, y, t) tuples
    total_bid_amount: float
    grid_cell_bids: List[GridCellBid]  # Breakdown by grid cell
    
    def __post_init__(self):
        # Validate that grid_cell_bids match the path
        path_positions = set(self.path)
        bid_positions = set(bid.position for bid in self.grid_cell_bids)
        if path_positions != bid_positions:
            raise ValueError("Grid cell bids don't match path positions")

@dataclass
class AuctionRoundResult:
    """Result of a single auction round"""
    round_number: int
    winners: Dict[int, PathBid]  # agent_id -> PathBid
    losers: Dict[int, PathBid]   # agent_id -> PathBid
    total_revenue: float
    conflicts_resolved: List[Tuple[int, int]]  # (winner_id, loser_id) pairs
    
    def get_winner_paths(self) -> Dict[int, List[Tuple[int, int, int]]]:
        """Extract winner paths in standard format"""
        return {agent_id: bid.path for agent_id, bid in self.winners.items()}

@dataclass
class AuctionResult:
    """Final result of the complete auction process"""
    success: bool
    total_rounds: int
    final_winners: Dict[int, List[Tuple[int, int, int]]]  # agent_id -> path
    unassigned_agents: List[int]  # agents who didn't get paths
    total_revenue: float
    round_results: List[AuctionRoundResult]
    failure_reason: str = ""

class AuctionSystem:
    """
    Multi-round auction system for airspace allocation
    """
    
    def __init__(self, grid_system: GridSystem, cbs_solver: ConflictBasedSearch, pathfinder: AStarPathfinder):
        """
        Initialize auction system
        
        Args:
            grid_system: The grid system instance
            cbs_solver: CBS solver for validation
            pathfinder: A* pathfinder for path generation
        """
        self.grid = grid_system
        self.cbs = cbs_solver
        self.pathfinder = pathfinder
        
        # Auction parameters
        self.max_rounds = 5
        self.pricing_alpha = 0.1  # AvgBudget multiplier
        self.pricing_beta = 0.2   # ConflictDensity multiplier
        self.pricing_gamma = 0.1  # Base price
        
        # Strategy parameters
        self.conservative_increment = 0.1  # 10% of starting price
        self.aggressive_multiplier = 1.5   # 150% of starting price
        self.balanced_increment = 0.3      # 30% of starting price
        
        self.debug_mode = False
        
        # Statistics
        self.stats = {
            'total_auctions': 0,
            'successful_auctions': 0,
            'average_rounds': 0,
            'total_revenue': 0.0
        }
    
    def calculate_starting_prices(self, conflict_density: Dict[Tuple[int, int], int],
                                agents: List[Agent]) -> Dict[Tuple[int, int], float]:
        """
        Calculate starting prices for grid cells based on conflict density and agent budgets
        
        Args:
            conflict_density: Conflict count per (x, y) position
            agents: List of participating agents
            
        Returns:
            Dictionary mapping (x, y) to starting price
        """
        if not agents:
            return {}
        
        # Calculate average budget
        budgets = [agent.budget for agent in agents if agent.budget > 0]
        avg_budget = sum(budgets) / len(budgets) if budgets else 50.0
        
        starting_prices = {}
        
        # Calculate price for each grid cell
        for (x, y), conflict_count in conflict_density.items():
            price = (avg_budget * self.pricing_alpha) * (conflict_count * self.pricing_beta) + self.pricing_gamma
            starting_prices[(x, y)] = max(price, self.pricing_gamma)  # Minimum price
        
        # For cells with no conflicts, use base price
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if (x, y) not in starting_prices:
                    starting_prices[(x, y)] = self.pricing_gamma
        
        return starting_prices
    
    def decompose_path_bid(self, agent: Agent, path: List[Tuple[int, int, int]], 
                          total_bid: float, starting_prices: Dict[Tuple[int, int], float]) -> List[GridCellBid]:
        """
        Decompose a total path bid into individual grid cell bids
        
        Args:
            agent: Agent making the bid
            path: Path being bid on
            total_bid: Total bid amount
            starting_prices: Starting prices for grid cells
            
        Returns:
            List of grid cell bids
        """
        if not path:
            return []
        
        # Calculate total starting price for the path
        total_starting_price = 0.0
        for x, y, t in path:
            total_starting_price += starting_prices.get((x, y), self.pricing_gamma)
        
        if total_starting_price == 0:
            return []
        
        # Distribute bid proportionally to starting prices
        grid_cell_bids = []
        for x, y, t in path:
            cell_starting_price = starting_prices.get((x, y), self.pricing_gamma)
            cell_bid_amount = total_bid * (cell_starting_price / total_starting_price)
            
            grid_cell_bids.append(GridCellBid(
                agent_id=agent.id,
                position=(x, y, t),
                bid_amount=cell_bid_amount
            ))
        
        return grid_cell_bids
    
    def calculate_bid_amount(self, agent: Agent, path: List[Tuple[int, int, int]],
                           starting_prices: Dict[Tuple[int, int], float], 
                           round_number: int, total_rounds: int) -> float:
        """
        Calculate bid amount based on agent strategy
        
        Args:
            agent: Agent making the bid
            path: Path being bid on
            starting_prices: Current starting prices
            round_number: Current round (1-based)
            total_rounds: Total number of rounds
            
        Returns:
            Total bid amount
        """
        # Calculate minimum bid (sum of starting prices)
        min_bid = sum(starting_prices.get((x, y), self.pricing_gamma) for x, y, t in path)
        
        strategy = BiddingStrategy(agent.strategy)
        
        if strategy == BiddingStrategy.CONSERVATIVE:
            # Conservative: bid minimum increment above starting price
            increment = min_bid * self.conservative_increment
            return min_bid + increment
        
        elif strategy == BiddingStrategy.AGGRESSIVE:
            # Aggressive: front-load high bids, especially in early rounds
            round_factor = (total_rounds - round_number + 1) / total_rounds  # Higher in early rounds
            multiplier = 1.0 + (self.aggressive_multiplier - 1.0) * round_factor
            return min(min_bid * multiplier, agent.budget)
        
        elif strategy == BiddingStrategy.BALANCED:
            # Balanced: treat every round equally
            increment = min_bid * self.balanced_increment
            return min_bid + increment
        
        else:
            return min_bid
    
    def generate_bids_for_round(self, agents: List[Agent], 
                              starting_prices: Dict[Tuple[int, int], float],
                              emergency_paths: Dict[int, List[Position]],
                              higher_priority_paths: Dict[int, List[Position]],
                              round_number: int) -> List[PathBid]:
        """
        Generate bids for all agents in current round
        
        Args:
            agents: Agents participating in this round
            starting_prices: Starting prices for grid cells
            emergency_paths: Emergency agent paths to avoid
            higher_priority_paths: Paths from previous auction winners
            round_number: Current round number
            
        Returns:
            List of valid path bids
        """
        bids = []
        
        for agent in agents:
            # Find path for this agent
            path = self.pathfinder.find_path_for_agent(
                agent,
                emergency_paths=emergency_paths,
                higher_priority_paths=higher_priority_paths
            )
            
            if not path:
                if self.debug_mode:
                    print(f"Agent {agent.id} couldn't find path in round {round_number}")
                continue
            
            # Convert to tuple format
            path_tuples = [pos.to_tuple() for pos in path]
            
            # Calculate bid amount
            bid_amount = self.calculate_bid_amount(
                agent, path_tuples, starting_prices, round_number, self.max_rounds
            )
            
            # Check if agent can afford this bid
            if bid_amount > agent.budget:
                if self.debug_mode:
                    print(f"Agent {agent.id} can't afford bid of {bid_amount} (budget: {agent.budget})")
                continue
            
            # Decompose into grid cell bids
            grid_cell_bids = self.decompose_path_bid(agent, path_tuples, bid_amount, starting_prices)
            
            # Create path bid
            path_bid = PathBid(
                agent_id=agent.id,
                path=path_tuples,
                total_bid_amount=bid_amount,
                grid_cell_bids=grid_cell_bids
            )
            
            bids.append(path_bid)
        
        return bids
    
    def select_winners_for_grid_cells(self, bids: List[PathBid]) -> Dict[Tuple[int, int, int], GridCellBid]:
        """
        Select highest bidder for each grid cell
        
        Args:
            bids: All path bids for this round
            
        Returns:
            Dictionary mapping position to winning bid
        """
        cell_bids = defaultdict(list)
        
        # Group bids by grid cell
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
    
    def determine_auction_winners(self, bids: List[PathBid]) -> Tuple[List[PathBid], List[PathBid]]:
        """
        Determine auction winners based on complete path acquisition
        
        Args:
            bids: All path bids for this round
            
        Returns:
            Tuple of (winners, losers) as lists of PathBid
        """
        # Select winners for individual grid cells
        cell_winners = self.select_winners_for_grid_cells(bids)
        
        # Check which agents won all cells in their path
        winners = []
        losers = []
        
        for path_bid in bids:
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
        
        return winners, losers
    
    def resolve_winner_conflicts(self, winners: List[PathBid], 
                                emergency_paths: Dict[int, List[Position]]) -> List[PathBid]:
        """
        Resolve conflicts among auction winners using CBS validation
        
        Args:
            winners: List of auction winners
            emergency_paths: Emergency paths to avoid
            
        Returns:
            List of conflict-free winners (sorted by total bid, highest first)
        """
        if not winners:
            return []
        
        # Sort winners by total bid amount (highest first)
        sorted_winners = sorted(winners, key=lambda bid: bid.total_bid_amount, reverse=True)
        
        # Convert to format expected by CBS
        winner_paths = {bid.agent_id: bid.path for bid in sorted_winners}
        
        # Use CBS to validate and resolve conflicts
        validated_paths = self.cbs.validate_auction_winners(winner_paths, emergency_paths)
        
        # Return only validated winners
        validated_winners = []
        for winner in sorted_winners:
            if winner.agent_id in validated_paths:
                validated_winners.append(winner)
        
        return validated_winners
    
    def deduct_budgets(self, winners: List[PathBid], agent_dict: Dict[int, Agent]):
        """
        Deduct bid amounts from winning agents' budgets
        
        Args:
            winners: List of winning bids
            agent_dict: Dictionary of agents by ID
        """
        for winner in winners:
            if winner.agent_id in agent_dict:
                agent_dict[winner.agent_id].budget -= winner.total_bid_amount
                agent_dict[winner.agent_id].budget = max(0, agent_dict[winner.agent_id].budget)
    
    def run_auction_round(self, agents: List[Agent], round_number: int,
                         conflict_density: Dict[Tuple[int, int], int],
                         emergency_paths: Dict[int, List[Position]],
                         higher_priority_paths: Dict[int, List[Position]]) -> AuctionRoundResult:
        """
        Run a single auction round
        
        Args:
            agents: Agents participating in this round
            round_number: Current round number
            conflict_density: Conflict density for pricing
            emergency_paths: Emergency paths to avoid
            higher_priority_paths: Paths from previous winners
            
        Returns:
            Result of this auction round
        """
        # Calculate starting prices based on current CBS analysis
        cbs_result = self.cbs.solve(agents, emergency_paths)
        current_conflict_density = cbs_result.conflict_density if cbs_result.conflict_density else conflict_density
        
        starting_prices = self.calculate_starting_prices(current_conflict_density, agents)
        
        if self.debug_mode:
            print(f"\nRound {round_number} - Starting prices calculated")
            print(f"Conflict density: {dict(list(current_conflict_density.items())[:5])}")  # Show first 5
        
        # Generate bids from all agents
        bids = self.generate_bids_for_round(
            agents, starting_prices, emergency_paths, higher_priority_paths, round_number
        )
        
        if not bids:
            # No valid bids in this round
            return AuctionRoundResult(
                round_number=round_number,
                winners={},
                losers={},
                total_revenue=0.0,
                conflicts_resolved=[]
            )
        
        # Determine initial winners and losers
        initial_winners, initial_losers = self.determine_auction_winners(bids)
        
        if self.debug_mode:
            print(f"Round {round_number} - Initial winners: {len(initial_winners)}, losers: {len(initial_losers)}")
        
        # Resolve conflicts among winners
        final_winners = self.resolve_winner_conflicts(initial_winners, emergency_paths)
        
        # Update losers list (include initial winners who lost to conflicts)
        final_losers = initial_losers.copy()
        for initial_winner in initial_winners:
            if initial_winner not in final_winners:
                final_losers.append(initial_winner)
        
        # Calculate conflicts resolved
        conflicts_resolved = []
        for initial_winner in initial_winners:
            if initial_winner not in final_winners:
                # Find the winner who beat this agent
                for final_winner in final_winners:
                    conflicts_resolved.append((final_winner.agent_id, initial_winner.agent_id))
        
        # Calculate revenue
        total_revenue = sum(winner.total_bid_amount for winner in final_winners)
        
        # Create result
        result = AuctionRoundResult(
            round_number=round_number,
            winners={winner.agent_id: winner for winner in final_winners},
            losers={loser.agent_id: loser for loser in final_losers},
            total_revenue=total_revenue,
            conflicts_resolved=conflicts_resolved
        )
        
        if self.debug_mode:
            print(f"Round {round_number} - Final winners: {len(final_winners)}, total revenue: {total_revenue}")
        
        return result
    
    def run_auction(self, agents: List[Agent], 
                   conflict_density: Dict[Tuple[int, int], int],
                   emergency_paths: Dict[int, List[Position]] = None) -> AuctionResult:
        """
        Run complete multi-round auction
        
        Args:
            agents: Agents to participate in auction
            conflict_density: Initial conflict density from CBS
            emergency_paths: Emergency paths to avoid
            
        Returns:
            Complete auction result
        """
        self.stats['total_auctions'] += 1
        
        if emergency_paths is None:
            emergency_paths = {}
        
        # Initialize auction state
        remaining_agents = agents.copy()
        agent_dict = {agent.id: agent for agent in agents}
        round_results = []
        total_revenue = 0.0
        final_winners = {}
        higher_priority_paths = {}
        
        if self.debug_mode:
            print(f"\nStarting auction with {len(agents)} agents")
            print(f"Initial conflict density: {dict(list(conflict_density.items())[:5])}")
        
        # Run up to max_rounds
        for round_num in range(1, self.max_rounds + 1):
            if not remaining_agents:
                break  # No more agents to process
            
            # Run this round
            round_result = self.run_auction_round(
                remaining_agents, round_num, conflict_density, 
                emergency_paths, higher_priority_paths
            )
            
            round_results.append(round_result)
            total_revenue += round_result.total_revenue
            
            # Process round results
            if round_result.winners:
                # Deduct budgets from winners
                self.deduct_budgets(list(round_result.winners.values()), agent_dict)
                
                # Add winner paths to final results and higher priority paths
                for agent_id, winner_bid in round_result.winners.items():
                    final_winners[agent_id] = winner_bid.path
                    higher_priority_paths[agent_id] = [Position(x, y, t) for x, y, t in winner_bid.path]
                
                # Remove winners from remaining agents
                remaining_agents = [agent for agent in remaining_agents 
                                  if agent.id not in round_result.winners]
            
            # Check if all agents found paths
            if not remaining_agents:
                break
        
        # Determine unassigned agents
        unassigned_agents = [agent.id for agent in remaining_agents]
        
        # Create final result
        result = AuctionResult(
            success=len(final_winners) > 0,
            total_rounds=len(round_results),
            final_winners=final_winners,
            unassigned_agents=unassigned_agents,
            total_revenue=total_revenue,
            round_results=round_results
        )
        
        if result.success:
            self.stats['successful_auctions'] += 1
        
        # Update statistics
        self.stats['average_rounds'] = (
            (self.stats['average_rounds'] * (self.stats['total_auctions'] - 1) + len(round_results)) 
            / self.stats['total_auctions']
        )
        self.stats['total_revenue'] += total_revenue
        
        if self.debug_mode:
            print(f"\nAuction completed:")
            print(f"Winners: {len(final_winners)}")
            print(f"Unassigned: {len(unassigned_agents)}")
            print(f"Total revenue: {total_revenue}")
            print(f"Rounds used: {len(round_results)}")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get auction system statistics"""
        return {
            'total_auctions': self.stats['total_auctions'],
            'successful_auctions': self.stats['successful_auctions'],
            'success_rate': self.stats['successful_auctions'] / max(self.stats['total_auctions'], 1),
            'average_rounds': self.stats['average_rounds'],
            'total_revenue': self.stats['total_revenue'],
            'average_revenue_per_auction': self.stats['total_revenue'] / max(self.stats['total_auctions'], 1)
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
    
    def reset_statistics(self):
        """Reset auction statistics"""
        self.stats = {
            'total_auctions': 0,
            'successful_auctions': 0,
            'average_rounds': 0,
            'total_revenue': 0.0
        }

# Utility functions for integration

def create_auction_system(grid_system: GridSystem, cbs_solver: ConflictBasedSearch) -> AuctionSystem:
    """Create auction system with required components"""
    pathfinder = AStarPathfinder(grid_system)
    return AuctionSystem(grid_system, cbs_solver, pathfinder)

def prepare_agents_for_auction(grid_system: GridSystem, budget_range: Tuple[float, float] = (1.0, 100.0)) -> List[Agent]:
    """
    Prepare non-emergency agents for auction with random budgets
    
    Args:
        grid_system: Grid system containing agents
        budget_range: (min, max) budget range for random assignment
        
    Returns:
        List of agents ready for auction
    """
    auction_agents = []
    
    for agent in grid_system.agents.values():
        if agent.agent_type == AgentType.NON_EMERGENCY and not agent.path:
            # Assign random budget if not already set
            if agent.budget <= 0:
                agent.budget = random.uniform(budget_range[0], budget_range[1])
            
            # Assign random strategy if not already set
            if not agent.strategy or agent.strategy not in [s.value for s in BiddingStrategy]:
                agent.strategy = random.choice(list(BiddingStrategy)).value
            
            auction_agents.append(agent)
    
    return auction_agents

def update_grid_with_auction_winners(grid_system: GridSystem, auction_result: AuctionResult):
    """
    Update grid system with auction winner paths
    
    Args:
        grid_system: Grid system to update
        auction_result: Result of auction with winner paths
    """
    for agent_id, path_tuples in auction_result.final_winners.items():
        # Convert tuples to Position objects
        path = [Position(x, y, t) for x, y, t in path_tuples]
        grid_system.set_agent_path(agent_id, path)

def extract_conflict_density_from_cbs(cbs_result) -> Dict[Tuple[int, int], int]:
    """Extract conflict density from CBS result for auction"""
    return cbs_result.conflict_density if cbs_result.conflict_density else {}

def get_auction_summary(auction_result: AuctionResult) -> Dict:
    """Get summary statistics for an auction result"""
    return {
        'success': auction_result.success,
        'total_rounds': auction_result.total_rounds,
        'winners_count': len(auction_result.final_winners),
        'unassigned_count': len(auction_result.unassigned_agents),
        'total_revenue': auction_result.total_revenue,
        'average_revenue_per_winner': (
            auction_result.total_revenue / len(auction_result.final_winners) 
            if auction_result.final_winners else 0
        ),
        'success_rate': len(auction_result.final_winners) / (
            len(auction_result.final_winners) + len(auction_result.unassigned_agents)
        ) if (auction_result.final_winners or auction_result.unassigned_agents) else 0
    }