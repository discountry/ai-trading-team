"""Dynamic single-direction grid trading strategy."""

import time
from typing import List, Optional, Dict, Any
from weex_sdk import WeexClient, WeexWebSocket
from weex_sdk.logger import get_logger, setup_logger
from weex_sdk.exceptions import WeexAPIError

from .grid_level import GridLevel, GridState
from .order_manager import OrderManager
from .state_manager import StateManager
from .price_calculator import PriceCalculator
from .websocket_manager import WebSocketManager
from .exceptions import InvalidGridDirectionError, GridStrategyError

logger = get_logger("grid_strategy")


class GridStrategy:
    """Dynamic single-direction grid trading strategy."""

    STRATEGY_VERSION = "1.0"

    def __init__(
        self,
        client: WeexClient,
        ws: WeexWebSocket,
        symbol: str,
        direction: str,
        grid_count: int,
        size_per_grid: float,
        price_range_percent: float = 0.03,
        stop_loss_amount: float = 0.0,
        margin_mode: int = 1,
        state_file: Optional[str] = None,
        log_level: Optional[int] = None,
    ):
        """Initialize GridStrategy.

        Args:
            client: WeexClient instance
            ws: WeexWebSocket instance (must be private for order/position updates)
            symbol: Trading pair symbol
            direction: 'long' or 'short'
            grid_count: Number of grid levels
            size_per_grid: Size per grid level
            price_range_percent: Price range percentage (default: 3%)
            stop_loss_amount: Stop loss amount (default: 0, no stop loss)
            margin_mode: Margin mode (1: Cross, 3: Isolated)
            state_file: Path to state file for persistence
            log_level: Logging level (None uses default)
        """
        if direction not in ["long", "short"]:
            raise InvalidGridDirectionError(f"Invalid direction: {direction}")

        if grid_count < 2:
            raise GridStrategyError("grid_count must be at least 2")

        if size_per_grid <= 0:
            raise GridStrategyError("size_per_grid must be positive")

        if price_range_percent <= 0 or price_range_percent > 1:
            raise GridStrategyError("price_range_percent must be between 0 and 1")

        if log_level is not None:
            setup_logger(level=log_level)

        self.client = client
        self.ws = ws
        self.symbol = symbol
        self.direction = direction
        self.grid_count = grid_count
        self.size_per_grid = size_per_grid
        self.price_range_percent = price_range_percent
        self.stop_loss_amount = stop_loss_amount
        self.margin_mode = margin_mode

        self._running = False
        self._current_price: float = 0.0
        self._upper_bound: float = 0.0
        self._lower_bound: float = 0.0
        self._grid_levels: List[GridLevel] = []
        self._realized_pnl: float = 0.0
        self._start_time: int = 0
        self._tick_size: float = 0.01
        self._stop_loss_triggered: bool = False

        self.order_manager = OrderManager(client, symbol, margin_mode)
        self.ws_manager = WebSocketManager(ws)

        if state_file is None:
            state_file = f"grid_strategy_{symbol}_{direction}_{int(time.time())}.json"
        self.state_manager = StateManager(state_file)

        logger.info(
            f"GridStrategy initialized: {symbol} {direction} "
            f"{grid_count} grids, {size_per_grid} per grid, "
            f"range {price_range_percent * 100}%, stop_loss {stop_loss_amount}"
        )

    def start(self) -> None:
        """Start grid strategy."""
        if self._running:
            logger.warning("Strategy is already running")
            return

        logger.info("=" * 60)
        logger.info("Starting Grid Strategy")
        logger.info("=" * 60)

        self._running = True
        self._start_time = int(time.time() * 1000)

        self._fetch_tick_size()

        if not self._restore_or_initialize():
            logger.error("Failed to initialize strategy")
            self.stop()
            return

        self.ws_manager.connect()

        self._setup_websocket_subscriptions()

        logger.info("Grid Strategy started successfully")
        logger.info(f"Current price: {self._current_price}")
        logger.info(f"Grid range: [{self._lower_bound}, {self._upper_bound}]")
        logger.info(f"Stop loss: {self.stop_loss_amount} USDT")
        logger.info("=" * 60)

    def stop(self) -> None:
        """Stop grid strategy."""
        if not self._running:
            return

        logger.info("=" * 60)
        logger.info("Stopping Grid Strategy")
        logger.info("=" * 60)

        self._running = False

        self.ws_manager.close()

        self._save_state()

        self._print_summary()

        logger.info("Grid Strategy stopped")
        logger.info("=" * 60)

    def _fetch_tick_size(self) -> None:
        """Fetch tick size from contract info."""
        try:
            contracts = self.client.market.get_contracts(symbol=self.symbol)
            if contracts and len(contracts) > 0:
                self._tick_size = float(contracts[0].get("tick_size", "0.01"))
                logger.info(f"Fetched tick_size for {self.symbol}: {self._tick_size}")
            else:
                logger.warning(f"Could not fetch tick_size, using default: 0.01")
                self._tick_size = 0.01
        except Exception as e:
            logger.error(f"Error fetching tick_size: {e}, using default: 0.01")
            self._tick_size = 0.01

    def _restore_or_initialize(self) -> bool:
        """Restore strategy from state or initialize from scratch."""
        state = self.state_manager.load()

        if state:
            return self._restore_from_state(state)
        else:
            return self._initialize_new()

    def _restore_from_state(self, state: Dict) -> bool:
        """Restore strategy from saved state."""
        logger.info("Restoring strategy from saved state...")

        self._current_price = state.get("current_price", 0.0)
        self._upper_bound = state.get("upper_bound", 0.0)
        self._lower_bound = state.get("lower_bound", 0.0)
        self._realized_pnl = state.get("realized_pnl", 0.0)
        self._grid_levels = state.get("grid_levels", [])
        self._start_time = state.get("start_time", int(time.time() * 1000))

        if not self._verify_grid_levels():
            logger.error("Grid level verification failed, reinitializing...")
            return self._initialize_new()

        logger.info(f"Restored: price={self._current_price}, realized_pnl={self._realized_pnl}")
        return True

    def _initialize_new(self) -> bool:
        """Initialize strategy from scratch."""
        logger.info("Initializing new strategy...")

        try:
            ticker = self.client.market.get_ticker(self.symbol)
            self._current_price = float(ticker.get("last", 0))
            if self._current_price == 0:
                logger.error("Failed to get current price")
                return False
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return False

        self._lower_bound, self._upper_bound, grid_prices = PriceCalculator.calculate_grid_prices(
            self._current_price,
            self.direction,
            self.grid_count,
            self.price_range_percent,
            self._tick_size,
        )

        self._grid_levels = [
            GridLevel(price=price, size=self.size_per_grid, direction=self.direction)
            for price in grid_prices
        ]

        success = self._place_initial_orders()

        if success:
            self._save_state()
            logger.info(f"Initialized {len(self._grid_levels)} grid levels")

        return success

    def _verify_grid_levels(self) -> bool:
        """Verify that grid levels are still valid after restore."""
        modified = False
        for grid in self._grid_levels:
            if grid.state == GridState.OPENING:
                if grid.open_order_id:
                    detail = self.order_manager.client.trade.get_order_detail(grid.open_order_id)
                    status = detail.get("status")
                    if not status or status not in ["live", "open", "partially_filled"]:
                        logger.warning(
                            f"Grid level at {grid.price} has invalid order state: {status}"
                        )
                        grid.state = GridState.EMPTY
                        grid.open_order_id = None
                        modified = True

            elif grid.state == GridState.CLOSING:
                if grid.close_order_id:
                    detail = self.order_manager.client.trade.get_order_detail(grid.close_order_id)
                    status = detail.get("status")
                    if not status or status not in ["live", "open", "partially_filled"]:
                        logger.warning(
                            f"Grid level at {grid.price} has invalid close order state: {status}"
                        )
                        grid.state = GridState.HOLDING
                        grid.close_order_id = None
                        modified = True

        if modified:
            self._save_state()

        return True

    def _place_initial_orders(self) -> bool:
        """Place initial opening orders for all grid levels."""
        logger.info("Placing initial opening orders...")

        success_count = 0
        for grid in self._grid_levels:
            if grid.state == GridState.EMPTY:
                order_id = self.order_manager.place_open_order(grid)
                if order_id:
                    success_count += 1
                else:
                    logger.error(f"Failed to place order at {grid.price}")

        logger.info(f"Placed {success_count}/{len(self._grid_levels)} initial orders")
        return success_count == len(self._grid_levels)

    def _setup_websocket_subscriptions(self) -> None:
        """Setup WebSocket subscriptions for ticker, order, and position updates."""
        self.ws_manager.subscribe_ticker(self.symbol, callback=self._on_ticker_update)

        self.ws_manager.subscribe_order(callback=self._on_order_update)

        self.ws_manager.subscribe_position(callback=self._on_position_update)

    def _on_ticker_update(self, data: Dict[str, Any]) -> None:
        """Handle ticker update from WebSocket."""
        if not self._running:
            return

        try:
            new_price = float(data.get("data", {}).get("last", 0))
            if new_price > 0:
                old_price = self._current_price
                self._current_price = new_price

                logger.debug(f"Price update: {old_price} -> {new_price}")

                if PriceCalculator.should_rebuild_grid(
                    new_price, self._upper_bound, self._lower_bound
                ):
                    logger.info(
                        f"Price {new_price} outside range [{self._lower_bound}, {self._upper_bound}], "
                        "rebuilding grid..."
                    )
                    self._rebuild_grid()

        except (ValueError, TypeError) as e:
            logger.error(f"Error processing ticker update: {e}")

    def _on_order_update(self, data: Dict[str, Any]) -> None:
        """Handle order update from WebSocket."""
        if not self._running or self._stop_loss_triggered:
            return

        try:
            order_data = data.get("data", {})
            if not order_data:
                logger.warning("Order update received without data")
                return

            order_id = order_data.get("orderId")
            if not order_id:
                logger.warning("Order update received without orderId")
                return

            self._process_single_order_update(order_id, order_data)
            self._save_state()

        except Exception as e:
            logger.error(f"Error processing order update: {e}")

    def _on_position_update(self, data: Dict[str, Any]) -> None:
        """Handle position update from WebSocket."""
        if not self._running:
            return

        if self.stop_loss_amount > 0:
            floating_pnl = self.order_manager.get_floating_pnl()

            if floating_pnl <= -self.stop_loss_amount:
                logger.warning(
                    f"Stop loss triggered! Floating PnL: {floating_pnl} <= -{self.stop_loss_amount}"
                )
                self._trigger_stop_loss()

    def _process_single_order_update(self, order_id: str, order_data: Dict) -> None:
        """Process single order update from WebSocket event (avoids race condition)."""
        for grid in self._grid_levels:
            if grid.state == GridState.OPENING and grid.open_order_id == order_id:
                status = order_data.get("status")
                if status == "filled":
                    grid.filled_price = float(
                        order_data.get("priceAvg", order_data.get("price", grid.price))
                    )
                    grid.filled_time = int(order_data.get("createTime", int(time.time() * 1000)))
                    grid.open_order_id = None

                    logger.info(
                        f"Opening order filled at {grid.filled_price} (target: {grid.price})"
                    )

                    next_price = PriceCalculator.find_next_grid_price(
                        grid.filled_price, [g.price for g in self._grid_levels], self.direction
                    )

                    if next_price:
                        close_order_id = self.order_manager.place_close_order(grid, next_price)
                        if not close_order_id:
                            logger.error(f"Failed to place closing order at {next_price}")
                            grid.state = GridState.HOLDING
                    else:
                        logger.warning(f"No next grid price found for {grid.filled_price}")
                        grid.state = GridState.HOLDING

                elif status in ["canceled", "expired", "rejected"]:
                    logger.warning(f"Opening order at {grid.price} failed with status: {status}")
                    grid.state = GridState.EMPTY
                    grid.open_order_id = None

            elif grid.state == GridState.CLOSING and grid.close_order_id == order_id:
                status = order_data.get("status")
                if status == "filled":
                    close_price = float(
                        order_data.get("priceAvg", order_data.get("price", grid.close_target_price))
                    )
                    realized_pnl = float(order_data.get("totalProfits", 0))

                    grid.realized_pnl = realized_pnl
                    self._realized_pnl += realized_pnl
                    grid.close_order_id = None

                    logger.info(
                        f"Closing order filled at {close_price}, "
                        f"realized PnL: {realized_pnl}, "
                        f"total realized PnL: {self._realized_pnl}"
                    )

                    grid.state = GridState.EMPTY
                    grid.filled_price = None
                    grid.filled_time = None
                    grid.close_target_price = None

                    new_order_id = self.order_manager.place_open_order(grid)
                    if new_order_id:
                        logger.info(f"New opening order placed at {grid.price}")
                    else:
                        logger.error(f"Failed to place new opening order at {grid.price}")

                elif status in ["canceled", "expired", "rejected"]:
                    logger.warning(
                        f"Closing order at {grid.close_target_price} failed with status: {status}"
                    )
                    grid.state = GridState.HOLDING
                    grid.close_order_id = None

    def _rebuild_grid(self) -> None:
        """Rebuild grid when price moves outside current range."""
        logger.info("Rebuilding grid...")

        cancel_failed = False
        for grid in self._grid_levels:
            if grid.state in [GridState.EMPTY, GridState.OPENING]:
                if grid.open_order_id:
                    cancel_success = self.order_manager.cancel_order(grid.open_order_id)
                    if not cancel_success:
                        logger.warning(f"Failed to cancel order {grid.open_order_id}")
                        cancel_failed = True
                    grid.open_order_id = None
                grid.state = GridState.EMPTY

        if cancel_failed:
            logger.warning("Some orders failed to cancel during grid rebuild")

        new_lower, new_upper, new_prices = PriceCalculator.calculate_grid_prices(
            self._current_price,
            self.direction,
            self.grid_count,
            self.price_range_percent,
            self._tick_size,
        )

        self._lower_bound = new_lower
        self._upper_bound = new_upper

        existing_close_prices = {
            g.close_target_price
            for g in self._grid_levels
            if g.state in [GridState.HOLDING, GridState.CLOSING]
            and g.close_target_price is not None
        }

        tolerance = self._tick_size / 2
        filtered_prices = [
            p
            for p in new_prices
            if not any(abs(p - existing) < tolerance for existing in existing_close_prices)
        ]

        existing_grids = {g.price: g for g in self._grid_levels}

        new_grid_levels = []
        for price in filtered_prices:
            if price in existing_grids:
                new_grid_levels.append(existing_grids[price])
            else:
                new_grid = GridLevel(price=price, size=self.size_per_grid, direction=self.direction)
                new_grid_levels.append(new_grid)

        for grid in self._grid_levels:
            if grid.price not in filtered_prices and grid.state in [
                GridState.HOLDING,
                GridState.CLOSING,
            ]:
                new_grid_levels.append(grid)

        self._grid_levels = new_grid_levels

        for grid in self._grid_levels:
            if grid.state == GridState.EMPTY:
                self.order_manager.place_open_order(grid)

        self._save_state()

        logger.info(
            f"Grid rebuilt: new range [{self._lower_bound}, {self._upper_bound}], "
            f"{len(self._grid_levels)} levels"
        )

    def _trigger_stop_loss(self) -> None:
        """Trigger stop loss: close all positions and cancel all orders."""
        if self._stop_loss_triggered:
            return

        logger.warning("=" * 60)
        logger.warning("STOP LOSS TRIGGERED")
        logger.warning("=" * 60)

        cancel_success = self.order_manager.cancel_all_orders()
        close_success = self.order_manager.close_all_positions()

        if not (cancel_success and close_success):
            logger.error("Stop loss execution failed! Orders or positions not properly handled")
            logger.warning("=" * 60)
            return

        self._stop_loss_triggered = True

        for grid in self._grid_levels:
            grid.state = GridState.EMPTY
            grid.open_order_id = None
            grid.close_order_id = None

        self._save_state()

        logger.warning("Stop loss executed: all positions closed, all orders canceled")
        logger.warning("=" * 60)

        self.stop()

    def _save_state(self) -> None:
        """Save current strategy state."""
        try:
            self.state_manager.save(
                version=self.STRATEGY_VERSION,
                symbol=self.symbol,
                direction=self.direction,
                grid_count=self.grid_count,
                size_per_grid=self.size_per_grid,
                price_range_percent=self.price_range_percent,
                stop_loss_amount=self.stop_loss_amount,
                current_price=self._current_price,
                upper_bound=self._upper_bound,
                lower_bound=self._lower_bound,
                grid_levels=self._grid_levels,
                realized_pnl=self._realized_pnl,
                start_time=self._start_time,
                margin_mode=self.margin_mode,
            )
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _print_summary(self) -> None:
        """Print strategy summary."""
        runtime = (int(time.time() * 1000) - self._start_time) / 1000 / 60

        logger.info("=" * 60)
        logger.info("Strategy Summary")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Direction: {self.direction}")
        logger.info(f"Runtime: {runtime:.2f} minutes")
        logger.info(f"Grid count: {self.grid_count}")
        logger.info(f"Realized PnL: {self._realized_pnl}")
        logger.info(f"Stop loss triggered: {self._stop_loss_triggered}")

        state_counts = {}
        for grid in self._grid_levels:
            state_counts[grid.state] = state_counts.get(grid.state, 0) + 1

        for state, count in state_counts.items():
            logger.info(f"{state.value}: {count}")

        logger.info("=" * 60)

    @property
    def is_running(self) -> bool:
        """Check if strategy is running."""
        return self._running

    @property
    def realized_pnl(self) -> float:
        """Get total realized PnL."""
        return self._realized_pnl

    @property
    def grid_levels(self) -> List[GridLevel]:
        """Get all grid levels."""
        return self._grid_levels.copy()
