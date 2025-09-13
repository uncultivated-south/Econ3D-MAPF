"""
Auction System Module for UrbanAirspaceSim
Handles multi-round bidding with CBS validation and conflict resolution
REFACTORED VERSION - Addresses identified bugs and integrates with improved modules
"""

from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy
import time
import threading
from decimal import Decimal, ROUND_HALF_UP
import logging

# Import from refactored modules
from grid_system import Position, Agent, AgentType, GridSystem
from astar_pathfinding import AStarPathfinder, PathfindingConfig, PathfindingResult, create_fast_config
from cbs_module import ConflictBasedSearch, CBSResult, convert_positions_to_tuple_paths

class BiddingStrategy(Enum):
    """Bidding strategies for agents"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive" 
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"  # New: adapts based on success rate

class AuctionPhase(Enum):
    """Phases of auction execution"""
    INITIALIZATION = "initialization"
    BIDDING = "bidding"
    WINNER_SELECTION = "winner_selection"
    CONFLICT_RESOLUTION = "conflict_resolution"
    BUDGET_PROCESSING = "budget_processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BudgetTransaction:
    """Represents a budget transaction with rollback capability"""
    agent_id: int
    amount: Decimal
    transaction_type: str  # "debit", "credit", "reserve"
    timestamp: float = field(default_factory=time.time)
    committed: bool = False
    
    def __post_init__(self):
        # Ensure amount is Decimal for precision
        if not isinstance(self.amount, Decimal):
            self.amount = Decimal(str(self.amount))

@dataclass
class GridCellBid:
    """Represents a bid on a specific grid cell with precise pricing"""
    agent_id: int
    position: Tuple[int, int, int]  # (x, y, t)
    bid_amount: Decimal
    cell_priority: float = 1.0  # Priority weight for this cell
    
    def __post_init__(self):
        if not isinstance(self.bid_amount, Decimal):
            self.bid_amount = Decimal(str(self.bid_amount))
    
    def __hash__(self):
        return hash((self.agent_id, self.position))

@dataclass
class PathBid:
    """Represents an agent's bid on a complete path with validation"""
    agent_id: int
    path: List[Position]  # Complete path
    total_bid_amount: Decimal
    grid_cell_bids: List[GridCellBid]
    bid_strategy: BiddingStrategy
    confidence_score: float = 1.0  # Agent's confidence in this path
    
    def __post_init__(self):
        if not isinstance(self.total_bid_amount, Decimal):
            self.total_bid_amount = Decimal(str(self.total_bid_amount))
        
        # Validate that grid cell bids match the path
        path_positions = {(pos.x, pos.y, pos.t) for pos in self.path}
        bid_positions = {bid.position for bid in self.grid_cell_bids}
        
        if path_positions != bid_positions:
            raise ValueError(f"Grid cell bids don't match path positions for agent {self.agent_id}")
        
        # Validate total bid matches sum of cell bids
        cell_bid_sum = sum(bid.bid_amount for bid in self.grid_cell_bids)
        if abs(cell_bid_sum - self.total_bid_amount) > Decimal('0.001'):
            raise ValueError(f"Total bid amount doesn't match sum of cell bids for agent {self.agent_id}")

@dataclass
class AuctionRoundResult:
    """Result of a single auction round with detailed tracking"""
    round_number: int
    phase: AuctionPhase = AuctionPhase.COMPLETED
    
    # Bidding results
    initial_bids: Dict[int, PathBid] = field(default_factory=dict)
    winning_bids: Dict[int, PathBid] = field(default_factory=dict)
    losing_bids: Dict[int, PathBid] = field(default_factory=dict)
    
    # Financial tracking
    total_revenue: Decimal = Decimal('0')
    budget_transactions: List[BudgetTransaction] = field(default_factory=list)
    
    # Conflict resolution
    conflicts_before_resolution: int = 0
    conflicts_after_resolution: int = 0
    agents_eliminated_by_conflicts: List[int] = field(default_factory=list)
    
    # Performance metrics
    round_duration: float = 0.0
    pathfinding_time: float = 0.0
    conflict_resolution_time: float = 0.0
    
    def get_winner_paths(self) -> Dict[int, List[Position]]:
        """Extract winner paths in standard format"""
        return {agent_id: bid.path for agent_id, bid in self.winning_bids.items()}

@dataclass
class AuctionResult:
    """Comprehensive result of the complete auction process"""
    # Core results
    success: bool = False
    total_rounds: int = 0
    final_winners: Dict[int, List[Position]] = field(default_factory=dict)
    unassigned_agents: List[int] = field(default_factory=list)
    
    # Financial summary
    total_revenue: Decimal = Decimal('0')
    revenue_by_round: List[Decimal] = field(default_factory=list)
    average_winning_bid: Decimal = Decimal('0')
    
    # Performance metrics
    total_duration: float = 0.0
    total_pathfinding_time: float = 0.0
    total_conflict_resolution_time: float = 0.0
    
    # Detailed results
    round_results: List[AuctionRoundResult] = field(default_factory=list)
    budget_transactions: List[BudgetTransaction] = field(default_factory=list)
    
    # Failure analysis
    failure_reason: str = ""
    agents_by_failure_cause: Dict[str, List[int]] = field(default_factory=dict)

class BudgetManager:
    """Manages agent budgets with transaction support and rollback capability"""
    
    def __init__(self):
        self.agent_budgets: Dict[int, Decimal] = {}
        self.reserved_amounts: Dict[int, Decimal] = defaultdict(lambda: Decimal('0'))
        self.transaction_log: List[BudgetTransaction] = []
        self._lock = threading.RLock()
    
    def initialize_agent_budget(self, agent_id: int, initial_budget: Union[float, Decimal]):
        """Initialize budget for an agent"""
        with self._lock:
            if not isinstance(initial_budget, Decimal):
                initial_budget = Decimal(str(initial_budget))
            self.agent_budgets[agent_id] = initial_budget
    
    def get_available_budget(self, agent_id: int) -> Decimal:
        """Get available budget (total - reserved)"""
        with self._lock:
            total = self.agent_budgets.get(agent_id, Decimal('0'))
            reserved = self.reserved_amounts.get(agent_id, Decimal('0'))
            return max(Decimal('0'), total - reserved)
    
    def get_total_budget(self, agent_id: int) -> Decimal:
        """Get total budget for an agent"""
        with self._lock:
            return self.agent_budgets.get(agent_id, Decimal('0'))
    
    def reserve_budget(self, agent_id: int, amount: Union[float, Decimal]) -> bool:
        """Reserve budget for a potential bid"""
        with self._lock:
            if not isinstance(amount, Decimal):
                amount = Decimal(str(amount))
            
            if amount <= Decimal('0'):
                return False
            
            available = self.get_available_budget(agent_id)
            if available >= amount:
                self.reserved_amounts[agent_id] += amount
                
                transaction = BudgetTransaction(
                    agent_id=agent_id,
                    amount=amount,
                    transaction_type="reserve"
                )
                self.transaction_log.append(transaction)
                return True
            return False
    
    def release_reservation(self, agent_id: int, amount: Union[float, Decimal]) -> bool:
        """Release a budget reservation"""
        with self._lock:
            if not isinstance(amount, Decimal):
                amount = Decimal(str(amount))
            
            if self.reserved_amounts[agent_id] >= amount:
                self.reserved_amounts[agent_id] -= amount
                
                transaction = BudgetTransaction(
                    agent_id=agent_id,
                    amount=amount,
                    transaction_type="credit"
                )
                self.transaction_log.append(transaction)
                return True
            return False
    
    def commit_transaction(self, agent_id: int, amount: Union[float, Decimal]) -> bool:
        """Commit a transaction (deduct from total budget)"""
        with self._lock:
            if not isinstance(amount, Decimal):
                amount = Decimal(str(amount))
            
            # Release reservation first
            if not self.release_reservation(agent_id, amount):
                return False
            
            # Deduct from total budget
            if self.agent_budgets.get(agent_id, Decimal('0')) >= amount:
                self.agent_budgets[agent_id] -= amount
                
                transaction = BudgetTransaction(
                    agent_id=agent_id,
                    amount=amount,
                    transaction_type="debit",
                    committed=True
                )
                self.transaction_log.append(transaction)
                return True
            return False
    
    def get_transaction_history(self, agent_id: Optional[int] = None) -> List[BudgetTransaction]:
        """Get transaction history for an agent or all agents"""
        with self._lock:
            if agent_id is None:
                return self.transaction_log.copy()
            return [t for t in self.transaction_log if t.agent_id == agent_id]

class PricingEngine:
    """Handles dynamic pricing based on conflict density and market conditions"""
    
    def __init__(self, base_price: Decimal = Decimal('1.0')):
        self.base_price = base_price
        self.price_history: Dict[Tuple[int, int], List[Decimal]] = defaultdict(list)
        self.demand_multipliers = {
            'very_low': Decimal('0.8'),
            'low': Decimal('0.9'), 
            'normal': Decimal('1.0'),
            'high': Decimal('1.2'),
            'very_high': Decimal('1.5'),
            'extreme': Decimal('2.0')
        }
    
    def calculate_cell_prices(self, conflict_density: Dict[Tuple[int, int], int],
                            agent_budgets: List[Decimal],
                            round_number: int = 1) -> Dict[Tuple[int, int], Decimal]:
        """
        Calculate starting prices for grid cells based on multiple factors
        
        Args:
            conflict_density: Number of conflicts per spatial cell
            agent_budgets: List of available agent budgets
            round_number: Current round number
            
        Returns:
            Dictionary mapping spatial position to price
        """
        if not agent_budgets:
            return {}
        
        # Calculate market parameters
        avg_budget = sum(agent_budgets) / len(agent_budgets)
        max_budget = max(agent_budgets) if agent_budgets else Decimal('0')
        
        # Base pricing factors
        budget_factor = avg_budget / Decimal('100')  # Normalize to reasonable range
        round_factor = Decimal('1') + Decimal(str(round_number - 1)) * Decimal('0.1')  # Increase with rounds
        
        cell_prices = {}
        
        # Get conflict statistics for normalization
        if conflict_density:
            max_conflicts = max(conflict_density.values())
            avg_conflicts = sum(conflict_density.values()) / len(conflict_density)
        else:
            max_conflicts = avg_conflicts = 0
        
        # Price each cell based on demand
        for (x, y), conflict_count in conflict_density.items():
            # Normalize conflict density to demand level
            if max_conflicts > 0:
                conflict_intensity = conflict_count / max_conflicts
            else:
                conflict_intensity = 0
            
            # Determine demand level
            if conflict_intensity == 0:
                demand_level = 'very_low'
            elif conflict_intensity < 0.2:
                demand_level = 'low'
            elif conflict_intensity < 0.4:
                demand_level = 'normal'
            elif conflict_intensity < 0.6:
                demand_level = 'high'
            elif conflict_intensity < 0.8:
                demand_level = 'very_high'
            else:
                demand_level = 'extreme'
            
            # Calculate price
            demand_multiplier = self.demand_multipliers[demand_level]
            
            price = (self.base_price * budget_factor * demand_multiplier * round_factor)
            
            # Ensure minimum price and reasonable maximum
            price = max(price, self.base_price)
            price = min(price, max_budget / Decimal('2'))  # Don't price out agents completely
            
            # Round to reasonable precision
            cell_prices[(x, y)] = price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            
            # Track price history
            self.price_history[(x, y)].append(price)
        
        return cell_prices
    
    def get_price_trend(self, position: Tuple[int, int]) -> str:
        """Get price trend for a position"""
        history = self.price_history.get(position, [])
        if len(history) < 2:
            return "stable"
        
        recent_change = (history[-1] - history[-2]) / history[-2]
        
        if recent_change > Decimal('0.1'):
            return "increasing"
        elif recent_change < Decimal('-0.1'):
            return "decreasing"
        else:
            return "stable"

class AuctionSystem:
    """
    Enhanced multi-round auction system with proper budget management and conflict resolution
    """
    
    def __init__(self, grid_system: GridSystem, cbs_solver: ConflictBasedSearch, 
                 pathfinder: AStarPathfinder):
        """
        Initialize auction system with required components
        
        Args:
            grid_system: Grid system instance
            cbs_solver: CBS solver for conflict validation
            pathfinder: A* pathfinder for path generation
        """
        self.grid = grid_system
        self.cbs = cbs_solver
        self.pathfinder = pathfinder
        
        # Core components
        self.budget_manager = BudgetManager()
        self.pricing_engine = PricingEngine()
        
        # Auction parameters
        self.max_rounds = 5
        self.bid_increment_minimum = Decimal('0.01')
        self.reserve_price_multiplier = Decimal('0.5')  # Minimum bid as fraction of starting price
        
        # Strategy parameters
        self.strategy_configs = {
            BiddingStrategy.CONSERVATIVE: {
                'increment_factor': Decimal('0.05'),  # 5% increments
                'max_bid_ratio': Decimal('0.3'),      # Use up to 30% of budget
                'risk_tolerance': Decimal('0.2')
            },
            BiddingStrategy.AGGRESSIVE: {
                'increment_factor': Decimal('0.25'),  # 25% increments
                'max_bid_ratio': Decimal('0.8'),      # Use up to 80% of budget
                'risk_tolerance': Decimal('0.8')
            },
            BiddingStrategy.BALANCED: {
                'increment_factor': Decimal('0.15'),  # 15% increments
                'max_bid_ratio': Decimal('0.5'),      # Use up to 50% of budget
                'risk_tolerance': Decimal('0.5')
            },
            BiddingStrategy.ADAPTIVE: {
                'increment_factor': Decimal('0.10'),  # Adapts based on success
                'max_bid_ratio': Decimal('0.6'),      # Adapts based on competition
                'risk_tolerance': Decimal('0.4')
            }
        }
        
        # Thread safety and error handling
        self._auction_lock = threading.RLock()
        self.debug_mode = False
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.stats = {
            'total_auctions': 0,
            'successful_auctions': 0,
            'total_revenue': Decimal('0'),
            'average_rounds': 0.0,
            'average_duration': 0.0,
            'conflict_resolution_success_rate': 0.0
        }
    
    def prepare_agents_for_auction(self, agents: List[Agent], 
                                 budget_range: Optional[Tuple[Decimal, Decimal]] = None) -> List[Agent]:
        """
        Prepare agents for auction with budget validation and strategy assignment
        
        Args:
            agents: Agents to prepare
            budget_range: Optional range for random budget assignment
            
        Returns:
            List of prepared agents
        """
        prepared_agents = []
        
        if budget_range is None:
            budget_range = (Decimal('10.0'), Decimal('100.0'))
        
        for agent in agents:
            # Skip agents that already have paths
            if agent.has_path():
                continue
            
            # Initialize budget if not set
            if agent.budget <= 0:
                # Assign budget based on agent priority and random factor
                base_budget = budget_range[0] + (budget_range[1] - budget_range[0]) * Decimal(str(0.5))
                priority_multiplier = Decimal(str(agent.priority / 100))  # Scale priority
                agent.budget = float(base_budget * (Decimal('1') + priority_multiplier))
            
            # Ensure budget is Decimal in budget manager
            self.budget_manager.initialize_agent_budget(agent.id, Decimal(str(agent.budget)))
            
            # Assign strategy if not set
            if not hasattr(agent, 'strategy') or not agent.strategy:
                agent.strategy = BiddingStrategy.BALANCED.value
            
            prepared_agents.append(agent)
        
        return prepared_agents
    
    def _generate_path_for_agent(self, agent: Agent, emergency_paths: Dict[int, List[Position]],
                                higher_priority_paths: Dict[int, List[Position]]) -> Optional[List[Position]]:
        """Generate path for agent avoiding higher priority paths"""
        
        # Use fast pathfinding config for auction
        old_config = self.pathfinder.config
        self.pathfinder.config = create_fast_config()
        
        try:
            # Clear existing constraints
            self.pathfinder.clear_all_constraints()
            
            # Add emergency path constraints
            for emergency_id, emergency_path in emergency_paths.items():
                if emergency_id != agent.id:
                    for pos in emergency_path:
                        from astar_pathfinding import Constraint, ConstraintType
                        constraint = Constraint(
                            constraint_type=ConstraintType.VERTEX,
                            position=pos
                        )
                        self.pathfinder.add_constraint(constraint)
            
            # Add higher priority path constraints
            for priority_id, priority_path in higher_priority_paths.items():
                if priority_id != agent.id:
                    for pos in priority_path:
                        constraint = Constraint(
                            constraint_type=ConstraintType.VERTEX,
                            position=pos
                        )
                        self.pathfinder.add_constraint(constraint)
            
            # Find path
            result = self.pathfinder.find_path(agent)
            
            if result.success:
                return result.path
            else:
                if self.debug_mode:
                    self.logger.warning(f"Agent {agent.id} pathfinding failed: {result.failure_reason}")
                return None
        
        finally:
            # Restore original config and clean up
            self.pathfinder.config = old_config
            self.pathfinder.clear_all_constraints()
    
    def _calculate_bid_amount(self, agent: Agent, path: List[Position],
                            cell_prices: Dict[Tuple[int, int], Decimal],
                            round_number: int, historical_success: Dict[int, float]) -> Decimal:
        """
        Calculate bid amount based on agent strategy and market conditions
        
        Args:
            agent: Agent making the bid
            path: Path being bid on
            cell_prices: Current cell prices
            round_number: Current round number
            historical_success: Success rates by strategy for adaptation
            
        Returns:
            Total bid amount
        """
        strategy = BiddingStrategy(agent.strategy)
        config = self.strategy_configs[strategy]
        
        # Calculate minimum bid (sum of reserve prices)
        min_bid = Decimal('0')
        for pos in path:
            spatial_pos = (pos.x, pos.y)
            cell_price = cell_prices.get(spatial_pos, self.pricing_engine.base_price)
            reserve_price = cell_price * self.reserve_price_multiplier
            min_bid += reserve_price
        
        # Get agent's available budget
        available_budget = self.budget_manager.get_available_budget(agent.id)
        max_affordable = available_budget * config['max_bid_ratio']
        
        # Calculate base bid
        if strategy == BiddingStrategy.CONSERVATIVE:
            # Conservative: bid minimum plus small increment
            increment = min_bid * config['increment_factor']
            bid_amount = min_bid + increment
        
        elif strategy == BiddingStrategy.AGGRESSIVE:
            # Aggressive: front-load high bids, especially early
            round_factor = Decimal(str((self.max_rounds - round_number + 1) / self.max_rounds))
            multiplier = Decimal('1') + config['increment_factor'] * round_factor
            bid_amount = min_bid * multiplier
        
        elif strategy == BiddingStrategy.BALANCED:
            # Balanced: consistent incremental bidding
            increment = min_bid * config['increment_factor']
            bid_amount = min_bid + increment
        
        elif strategy == BiddingStrategy.ADAPTIVE:
            # Adaptive: adjust based on historical success rates
            success_rate = historical_success.get(agent.id, 0.5)
            if success_rate < 0.3:  # Low success, bid more aggressively
                multiplier = Decimal('1.3')
            elif success_rate > 0.7:  # High success, can be more conservative
                multiplier = Decimal('1.1')
            else:  # Medium success, balanced approach
                multiplier = Decimal('1.2')
            
            bid_amount = min_bid * multiplier
        
        else:
            bid_amount = min_bid
        
        # Ensure bid is within budget constraints
        bid_amount = min(bid_amount, max_affordable)
        bid_amount = max(bid_amount, min_bid)
        
        # Round to reasonable precision
        return bid_amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def _create_path_bid(self, agent: Agent, path: List[Position],
                        cell_prices: Dict[Tuple[int, int], Decimal],
                        total_bid: Decimal) -> PathBid:
        """
        Create a path bid with properly distributed cell bids
        
        Args:
            agent: Agent making the bid
            path: Path being bid on
            cell_prices: Current cell prices
            total_bid: Total bid amount to distribute
            
        Returns:
            Complete path bid
        """
        # Calculate total starting price for proportional distribution
        total_starting_price = Decimal('0')
        cell_prices_for_path = {}
        
        for pos in path:
            spatial_pos = (pos.x, pos.y)
            cell_price = cell_prices.get(spatial_pos, self.pricing_engine.base_price)
            cell_prices_for_path[pos] = cell_price
            total_starting_price += cell_price
        
        # Distribute total bid proportionally
        grid_cell_bids = []
        distributed_amount = Decimal('0')
        
        for i, pos in enumerate(path):
            if i == len(path) - 1:  # Last position gets remainder to avoid rounding errors
                cell_bid_amount = total_bid - distributed_amount
            else:
                cell_price = cell_prices_for_path[pos]
                if total_starting_price > 0:
                    proportion = cell_price / total_starting_price
                    cell_bid_amount = (total_bid * proportion).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
                else:
                    cell_bid_amount = total_bid / len(path)  # Equal distribution fallback
                distributed_amount += cell_bid_amount
            
            grid_cell_bid = GridCellBid(
                agent_id=agent.id,
                position=(pos.x, pos.y, pos.t),
                bid_amount=cell_bid_amount,
                cell_priority=1.0  # Could be enhanced with priority calculations
            )
            grid_cell_bids.append(grid_cell_bid)
        
        return PathBid(
            agent_id=agent.id,
            path=path,
            total_bid_amount=total_bid,
            grid_cell_bids=grid_cell_bids,
            bid_strategy=BiddingStrategy(agent.strategy)
        )
    
    def _generate_bids_for_round(self, agents: List[Agent], round_number: int,
                                cell_prices: Dict[Tuple[int, int], Decimal],
                                emergency_paths: Dict[int, List[Position]],
                                higher_priority_paths: Dict[int, List[Position]],
                                historical_success: Dict[int, float]) -> List[PathBid]:
        """
        Generate all bids for the current round with proper validation
        """
        bids = []
        pathfinding_start = time.time()
        
        for agent in agents:
            try:
                # Check if agent has sufficient budget
                available_budget = self.budget_manager.get_available_budget(agent.id)
                if available_budget <= Decimal('0.01'):  # Minimum viable budget
                    if self.debug_mode:
                        self.logger.info(f"Agent {agent.id} has insufficient budget: {available_budget}")
                    continue
                
                # Generate path
                path = self._generate_path_for_agent(agent, emergency_paths, higher_priority_paths)
                if not path:
                    if self.debug_mode:
                        self.logger.warning(f"Agent {agent.id} couldn't find path in round {round_number}")
                    continue
                
                # Calculate bid amount
                bid_amount = self._calculate_bid_amount(
                    agent, path, cell_prices, round_number, historical_success
                )
                
                # Verify agent can afford the bid
                if bid_amount > available_budget:
                    # Scale down bid to available budget
                    bid_amount = available_budget * Decimal('0.9')  # Leave small buffer
                    if bid_amount < self.pricing_engine.base_price:
                        continue  # Can't afford even minimum bid
                
                # Reserve budget for this bid
                if not self.budget_manager.reserve_budget(agent.id, bid_amount):
                    if self.debug_mode:
                        self.logger.warning(f"Agent {agent.id} couldn't reserve budget for bid: {bid_amount}")
                    continue
                
                # Create path bid
                path_bid = self._create_path_bid(agent, path, cell_prices, bid_amount)
                bids.append(path_bid)
                
            except Exception as e:
                self.logger.error(f"Error generating bid for agent {agent.id}: {e}")
                # Release any reserved budget
                try:
                    available = self.budget_manager.get_available_budget(agent.id)
                    if available < self.budget_manager.get_total_budget(agent.id):
                        reserved = self.budget_manager.get_total_budget(agent.id) - available
                        self.budget_manager.release_reservation(agent.id, reserved)
                except:
                    pass  # Best effort cleanup
                continue
        
        pathfinding_time = time.time() - pathfinding_start
        return bids, pathfinding_time
    
    def _determine_auction_winners(self, bids: List[PathBid]) -> Tuple[List[PathBid], List[PathBid]]:
        """
        Determine auction winners based on cell-by-cell highest bid
        
        Args:
            bids: All path bids for this round
            
        Returns:
            Tuple of (winners, losers)
        """
        # Group bids by grid cell
        cell_competitions = defaultdict(list)
        
        for path_bid in bids:
            for cell_bid in path_bid.grid_cell_bids:
                cell_competitions[cell_bid.position].append((cell_bid, path_bid))
        
        # Determine winner for each cell
        cell_winners = {}
        for position, competing_bids in cell_competitions.items():
            if competing_bids:
                # Sort by bid amount (highest first), then by agent priority as tiebreaker
                competing_bids.sort(
                    key=lambda x: (x[0].bid_amount, self.grid.agents[x[0].agent_id].priority),
                    reverse=True
                )
                winning_cell_bid, winning_path_bid = competing_bids[0]
                cell_winners[position] = winning_cell_bid.agent_id
        
        # Determine which agents won all cells in their paths
        winners = []
        losers = []
        
        for path_bid in bids:
            won_all_cells = True
            
            for cell_bid in path_bid.grid_cell_bids:
                if cell_winners.get(cell_bid.position) != path_bid.agent_id:
                    won_all_cells = False
                    break
            
            if won_all_cells:
                winners.append(path_bid)
            else:
                losers.append(path_bid)
        
        return winners, losers
    
    def _resolve_winner_conflicts(self, winners: List[PathBid],
                                 emergency_paths: Dict[int, List[Position]]) -> List[PathBid]:
        """
        Use CBS to resolve conflicts among auction winners
        
        Args:
            winners: List of winning bids
            emergency_paths: Emergency paths to avoid
            
        Returns:
            List of conflict-free winners (priority-ordered)
        """
        if not winners:
            return []
        
        conflict_resolution_start = time.time()
        
        try:
            # Sort winners by total bid amount (highest first)
            sorted_winners = sorted(winners, key=lambda bid: bid.total_bid_amount, reverse=True)
            
            # Convert to format expected by CBS
            winner_paths = {bid.agent_id: bid.path for bid in sorted_winners}
            
            # Use CBS to validate and resolve conflicts
            validated_paths = self.cbs.validate_auction_winners(winner_paths, emergency_paths)
            
            # Return only validated winners in original order
            validated_winners = []
            for winner in sorted_winners:
                if winner.agent_id in validated_paths:
                    validated_winners.append(winner)
            
            return validated_winners
            
        except Exception as e:
            self.logger.error(f"Error in conflict resolution: {e}")
            # Fallback: return winners sorted by bid amount
            return sorted(winners, key=lambda bid: bid.total_bid_amount, reverse=True)
    
    def _process_round_transactions(self, winners: List[PathBid], losers: List[PathBid]) -> List[BudgetTransaction]:
        """
        Process budget transactions for round results
        
        Args:
            winners: Winning bids
            losers: Losing bids
            
        Returns:
            List of completed transactions
        """
        transactions = []
        
        # Process winners - commit their transactions
        for winner in winners:
            success = self.budget_manager.commit_transaction(winner.agent_id, winner.total_bid_amount)
            if success:
                transaction = BudgetTransaction(
                    agent_id=winner.agent_id,
                    amount=winner.total_bid_amount,
                    transaction_type="debit",
                    committed=True
                )
                transactions.append(transaction)
            else:
                self.logger.error(f"Failed to commit transaction for winner {winner.agent_id}")
        
        # Process losers - release their reservations
        for loser in losers:
            self.budget_manager.release_reservation(loser.agent_id, loser.total_bid_amount)
            transaction = BudgetTransaction(
                agent_id=loser.agent_id,
                amount=loser.total_bid_amount,
                transaction_type="credit"
            )
            transactions.append(transaction)
        
        return transactions
    
    def _calculate_historical_success(self, round_results: List[AuctionRoundResult]) -> Dict[int, float]:
        """Calculate historical success rates for adaptive bidding"""
        success_counts = defaultdict(int)
        total_attempts = defaultdict(int)
        
        for round_result in round_results:
            for agent_id in round_result.initial_bids:
                total_attempts[agent_id] += 1
                if agent_id in round_result.winning_bids:
                    success_counts[agent_id] += 1
        
        return {
            agent_id: success_counts[agent_id] / total_attempts[agent_id]
            for agent_id in total_attempts
            if total_attempts[agent_id] > 0
        }
    
    def run_auction_round(self, agents: List[Agent], round_number: int,
                         conflict_density: Dict[Tuple[int, int], int],
                         emergency_paths: Dict[int, List[Position]],
                         higher_priority_paths: Dict[int, List[Position]],
                         historical_success: Dict[int, float]) -> AuctionRoundResult:
        """
        Execute a single auction round with comprehensive error handling
        
        Args:
            agents: Agents participating in this round
            round_number: Current round number
            conflict_density: Conflict density for pricing
            emergency_paths: Emergency paths to avoid
            higher_priority_paths: Paths from previous winners
            historical_success: Historical success rates for adaptive strategies
            
        Returns:
            Comprehensive round result
        """
        round_start_time = time.time()
        result = AuctionRoundResult(round_number=round_number)
        
        if self.debug_mode:
            self.logger.info(f"Starting auction round {round_number} with {len(agents)} agents")
        
        try:
            result.phase = AuctionPhase.INITIALIZATION
            
            # Calculate current market prices
            agent_budgets = [self.budget_manager.get_available_budget(a.id) for a in agents]
            cell_prices = self.pricing_engine.calculate_cell_prices(
                conflict_density, agent_budgets, round_number
            )
            
            if self.debug_mode:
                total_cells_priced = len(cell_prices)
                avg_price = sum(cell_prices.values()) / len(cell_prices) if cell_prices else Decimal('0')
                self.logger.info(f"Round {round_number}: Priced {total_cells_priced} cells, avg price: {avg_price}")
            
            # Generate bids
            result.phase = AuctionPhase.BIDDING
            bids, pathfinding_time = self._generate_bids_for_round(
                agents, round_number, cell_prices, emergency_paths, 
                higher_priority_paths, historical_success
            )
            result.pathfinding_time = pathfinding_time
            result.initial_bids = {bid.agent_id: bid for bid in bids}
            
            if not bids:
                result.phase = AuctionPhase.FAILED
                if self.debug_mode:
                    self.logger.warning(f"Round {round_number}: No valid bids generated")
                return result
            
            # Determine winners
            result.phase = AuctionPhase.WINNER_SELECTION
            initial_winners, initial_losers = self._determine_auction_winners(bids)
            
            if self.debug_mode:
                self.logger.info(f"Round {round_number}: Initial winners: {len(initial_winners)}, losers: {len(initial_losers)}")
            
            # Resolve conflicts
            result.phase = AuctionPhase.CONFLICT_RESOLUTION
            conflict_resolution_start = time.time()
            
            if initial_winners:
                result.conflicts_before_resolution = len([
                    conflict for winner in initial_winners 
                    for conflict in self.cbs.detect_conflicts({winner.agent_id: winner.path})
                ])
                
                final_winners = self._resolve_winner_conflicts(initial_winners, emergency_paths)
                
                result.conflicts_after_resolution = len([
                    conflict for winner in final_winners
                    for conflict in self.cbs.detect_conflicts({winner.agent_id: winner.path})
                ])
                
                # Identify agents eliminated by conflict resolution
                eliminated_agents = [
                    winner.agent_id for winner in initial_winners 
                    if winner not in final_winners
                ]
                result.agents_eliminated_by_conflicts = eliminated_agents
            else:
                final_winners = []
            
            result.conflict_resolution_time = time.time() - conflict_resolution_start
            
            # Update losers list
            final_losers = initial_losers.copy()
            for initial_winner in initial_winners:
                if initial_winner not in final_winners:
                    final_losers.append(initial_winner)
            
            # Process budget transactions
            result.phase = AuctionPhase.BUDGET_PROCESSING
            transactions = self._process_round_transactions(final_winners, final_losers)
            result.budget_transactions = transactions
            
            # Calculate financial results
            result.total_revenue = sum(winner.total_bid_amount for winner in final_winners)
            result.winning_bids = {winner.agent_id: winner for winner in final_winners}
            result.losing_bids = {loser.agent_id: loser for loser in final_losers}
            
            result.phase = AuctionPhase.COMPLETED
            
            if self.debug_mode:
                self.logger.info(f"Round {round_number}: Final winners: {len(final_winners)}, revenue: {result.total_revenue}")
            
        except Exception as e:
            result.phase = AuctionPhase.FAILED
            self.logger.error(f"Error in auction round {round_number}: {e}")
            
            # Cleanup: release all reservations for this round
            for agent in agents:
                try:
                    available = self.budget_manager.get_available_budget(agent.id)
                    total = self.budget_manager.get_total_budget(agent.id)
                    if available < total:
                        reserved = total - available
                        self.budget_manager.release_reservation(agent.id, reserved)
                except:
                    pass  # Best effort cleanup
        
        finally:
            result.round_duration = time.time() - round_start_time
        
        return result
    
    def run_auction(self, agents: List[Agent], 
                   conflict_density: Dict[Tuple[int, int], int],
                   emergency_paths: Dict[int, List[Position]] = None) -> AuctionResult:
        """
        Execute complete multi-round auction with comprehensive tracking
        
        Args:
            agents: Agents to participate in auction
            conflict_density: Initial conflict density from CBS
            emergency_paths: Emergency paths to avoid
            
        Returns:
            Comprehensive auction result
        """
        with self._auction_lock:
            auction_start_time = time.time()
            self.stats['total_auctions'] += 1
            
            result = AuctionResult()
            
            if emergency_paths is None:
                emergency_paths = {}
            
            if self.debug_mode:
                self.logger.info(f"Starting auction with {len(agents)} agents")
            
            try:
                # Prepare agents
                prepared_agents = self.prepare_agents_for_auction(agents)
                
                if not prepared_agents:
                    result.failure_reason = "No agents prepared for auction"
                    result.unassigned_agents = [a.id for a in agents]
                    return result
                
                # Initialize auction state
                remaining_agents = prepared_agents.copy()
                round_results = []
                higher_priority_paths = {}
                historical_success = {}
                
                # Run auction rounds
                for round_num in range(1, self.max_rounds + 1):
                    if not remaining_agents:
                        break  # All agents assigned
                    
                    # Run round
                    round_result = self.run_auction_round(
                        remaining_agents, round_num, conflict_density,
                        emergency_paths, higher_priority_paths, historical_success
                    )
                    
                    round_results.append(round_result)
                    
                    # Process round results
                    if round_result.phase == AuctionPhase.COMPLETED and round_result.winning_bids:
                        # Add winners to final results
                        for agent_id, winning_bid in round_result.winning_bids.items():
                            result.final_winners[agent_id] = winning_bid.path
                            higher_priority_paths[agent_id] = winning_bid.path
                        
                        # Remove winners from remaining agents
                        remaining_agents = [
                            agent for agent in remaining_agents 
                            if agent.id not in round_result.winning_bids
                        ]
                        
                        # Update historical success for adaptive strategies
                        historical_success = self._calculate_historical_success(round_results)
                    
                    # Update financial tracking
                    result.total_revenue += round_result.total_revenue
                    result.revenue_by_round.append(round_result.total_revenue)
                    result.total_pathfinding_time += round_result.pathfinding_time
                    result.total_conflict_resolution_time += round_result.conflict_resolution_time
                    result.budget_transactions.extend(round_result.budget_transactions)
                
                # Finalize results
                result.success = len(result.final_winners) > 0
                result.total_rounds = len(round_results)
                result.round_results = round_results
                result.unassigned_agents = [agent.id for agent in remaining_agents]
                
                if result.final_winners:
                    result.average_winning_bid = result.total_revenue / len(result.final_winners)
                
                # Update statistics
                if result.success:
                    self.stats['successful_auctions'] += 1
                
                self.stats['total_revenue'] += result.total_revenue
                self.stats['average_rounds'] = (
                    (self.stats['average_rounds'] * (self.stats['total_auctions'] - 1) + result.total_rounds) /
                    self.stats['total_auctions']
                )
                
                if self.debug_mode:
                    self.logger.info(f"Auction completed: {len(result.final_winners)} winners, "
                                   f"{len(result.unassigned_agents)} unassigned, "
                                   f"revenue: {result.total_revenue}")
            
            except Exception as e:
                result.failure_reason = f"Auction system error: {e}"
                result.unassigned_agents = [a.id for a in agents]
                self.logger.error(f"Auction failed with error: {e}")
            
            finally:
                result.total_duration = time.time() - auction_start_time
                self.stats['average_duration'] = (
                    (self.stats['average_duration'] * (self.stats['total_auctions'] - 1) + result.total_duration) /
                    self.stats['total_auctions']
                )
            
            return result
    
    def get_statistics(self) -> Dict:
        """Get comprehensive auction system statistics"""
        return {
            'total_auctions': self.stats['total_auctions'],
            'successful_auctions': self.stats['successful_auctions'],
            'success_rate': self.stats['successful_auctions'] / max(self.stats['total_auctions'], 1),
            'total_revenue': float(self.stats['total_revenue']),
            'average_revenue_per_auction': float(self.stats['total_revenue']) / max(self.stats['total_auctions'], 1),
            'average_rounds': self.stats['average_rounds'],
            'average_duration': self.stats['average_duration'],
            'conflict_resolution_success_rate': self.stats['conflict_resolution_success_rate']
        }
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
        if enabled:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def reset_statistics(self):
        """Reset auction statistics"""
        self.stats = {
            'total_auctions': 0,
            'successful_auctions': 0,
            'total_revenue': Decimal('0'),
            'average_rounds': 0.0,
            'average_duration': 0.0,
            'conflict_resolution_success_rate': 0.0
        }
        self.budget_manager = BudgetManager()
        self.pricing_engine = PricingEngine()

# Utility functions for integration with other modules

def create_auction_system(grid_system: GridSystem, cbs_solver: ConflictBasedSearch,
                         pathfinder: AStarPathfinder = None) -> AuctionSystem:
    """
    Create auction system with required components
    
    Args:
        grid_system: Grid system instance
        cbs_solver: CBS solver for conflict resolution
        pathfinder: Optional pathfinder (creates default if None)
        
    Returns:
        Configured auction system
    """
    if pathfinder is None:
        # Create pathfinder with fast config for auctions
        pathfinder = AStarPathfinder(grid_system, create_fast_config())
    
    return AuctionSystem(grid_system, cbs_solver, pathfinder)

def update_grid_with_auction_winners(grid_system: GridSystem, auction_result: AuctionResult) -> Dict[int, bool]:
    """
    Update grid system with auction winner paths
    
    Args:
        grid_system: Grid system to update
        auction_result: Result of auction with winner paths
        
    Returns:
        Dictionary mapping agent_id to success status
    """
    results = {}
    
    for agent_id, path in auction_result.final_winners.items():
        success = grid_system.set_agent_path(agent_id, path)
        results[agent_id] = success
        
        if not success:
            logging.getLogger(__name__).warning(f"Failed to set path for auction winner {agent_id}")
    
    return results

def extract_conflict_density_from_cbs(cbs_result: CBSResult) -> Dict[Tuple[int, int], int]:
    """Extract conflict density from CBS result for auction pricing"""
    return dict(cbs_result.conflict_density) if cbs_result.conflict_density else {}

def analyze_auction_performance(auction_result: AuctionResult) -> Dict:
    """
    Analyze auction performance and provide insights
    
    Args:
        auction_result: Result from auction execution
        
    Returns:
        Performance analysis dictionary
    """
    analysis = {
        'efficiency_rating': 'high',
        'financial_performance': 'good',
        'bottlenecks': [],
        'recommendations': []
    }
    
    # Efficiency analysis
    if auction_result.total_duration > 60:
        analysis['efficiency_rating'] = 'low'
        analysis['bottlenecks'].append('high_execution_time')
        analysis['recommendations'].append('Consider reducing max_rounds or using faster pathfinding')
    elif auction_result.total_duration > 20:
        analysis['efficiency_rating'] = 'medium'
    
    # Success rate analysis
    total_agents = len(auction_result.final_winners) + len(auction_result.unassigned_agents)
    success_rate = len(auction_result.final_winners) / total_agents if total_agents > 0 else 0
    
    if success_rate < 0.5:
        analysis['bottlenecks'].append('low_assignment_rate')
        analysis['recommendations'].append('Consider adjusting pricing strategy or increasing max_rounds')
    
    # Financial performance
    if auction_result.average_winning_bid > Decimal('0'):
        if auction_result.total_revenue / len(auction_result.final_winners) < Decimal('10'):
            analysis['financial_performance'] = 'poor'
            analysis['recommendations'].append('Consider increasing base prices')
        elif auction_result.total_revenue / len(auction_result.final_winners) > Decimal('100'):
            analysis['financial_performance'] = 'excellent'
    
    # Conflict resolution analysis
    total_conflicts_before = sum(r.conflicts_before_resolution for r in auction_result.round_results)
    total_conflicts_after = sum(r.conflicts_after_resolution for r in auction_result.round_results)
    
    if total_conflicts_before > 0:
        resolution_rate = 1 - (total_conflicts_after / total_conflicts_before)
        if resolution_rate < 0.8:
            analysis['bottlenecks'].append('poor_conflict_resolution')
            analysis['recommendations'].append('Consider improving CBS configuration or pathfinding quality')
    
    analysis['metrics'] = {
        'success_rate': success_rate,
        'average_bid': float(auction_result.average_winning_bid),
        'revenue_per_agent': float(auction_result.total_revenue) / max(total_agents, 1),
        'rounds_used': auction_result.total_rounds,
        'conflict_resolution_rate': 1 - (total_conflicts_after / max(total_conflicts_before, 1))
    }
    
    return analysis

def get_auction_summary(auction_result: AuctionResult) -> Dict:
    """Get concise summary of auction results"""
    total_agents = len(auction_result.final_winners) + len(auction_result.unassigned_agents)
    
    return {
        'success': auction_result.success,
        'total_rounds': auction_result.total_rounds,
        'winners_count': len(auction_result.final_winners),
        'unassigned_count': len(auction_result.unassigned_agents),
        'success_rate': len(auction_result.final_winners) / max(total_agents, 1),
        'total_revenue': float(auction_result.total_revenue),
        'average_winning_bid': float(auction_result.average_winning_bid),
        'total_duration': auction_result.total_duration,
        'pathfinding_efficiency': (
            auction_result.total_pathfinding_time / auction_result.total_duration 
            if auction_result.total_duration > 0 else 0
        )
    }