import time
import asyncio
import numpy as np
import json
import random
import math
import numpy as np
from strategy import StrategyBaseClass
from utils import get_kill_switch, generate_random_string, get_run_key_status, ORDER_FILLED
from .utils_strategy import is_run_time_strategy
from .numba_model import *

class DMM002_03(StrategyBaseClass):
    """
    Dynamic Spread Market Volatility + Inventory Skew + Orderbook Imbalance + Grid Orders
    """
    
    def __init__(self, symbol: str, quote: str, exchange_name: str, strategy_name: str, strategy_info: dict, account: list, exchange_data: object, run_mode: str, run_key: str):
        super().__init__(symbol, quote, strategy_name, strategy_info, exchange_data, exchange_name, run_mode, run_key)
        self.account = account
        self.logger = self.logger_strategy
        self.r = exchange_data

        params = self.strategy_info

        self.start_base_value = float(params.get("start_base_value", 0.0))
        self.trade_value = float(params.get("trade_value", 5))
        
        self.allow_taker = bool(params.get("allow_taker", False))
        self.skew_factor = float(params.get("skew_factor", 0.1))
        self.volatility_multiplier = float(params.get("volatility_multiplier", 1.0))
        self.orderbook_imbalance_threshold = float(params.get("orderbook_imbalance_threshold", 0.5))

        self.half_spread_tick = int(params.get("half_spread_tick", 5))
        self.half_spread_tick_max = int(params.get("half_spread_tick_max", 10))
        self.tick_move_tolerance = int(params.get("tick_move_tolerance", 5))
        
        self.max_orders_pending = int(params.get("max_orders_pending", 5))
        self.max_position = float(params.get("max_position", 5))
        
        self.grid_level_orders = int(params.get("grid_level_orders", 5))
        self.grid_level_factor = float(params.get("grid_level_factor", 1.0))
        
        self.order_life_time = float(params.get("order_life_time", 5))
        self.new_order_interval = float(params.get("new_order_interval", 5))
        self.sleep_time = float(params.get("sleep_time", 5))
        self.cancel_all_interval = float(params.get("cancel_all_interval", 5))
        
        self.alpha, self.beta, self.mu, self.gamma = 0.2, 0.2, 0.1, 0.001
        
        capacity = 110
        
        self.net_position = 0.0
        self.mid_price_chg = np.full(capacity, np.nan, dtype=np.float64)
        self.mid_price_array = np.full(capacity, np.nan, dtype=np.float64)
        self.mid_price = 0.0
        self.orders_placed = []
        
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.lasted_mid_price = 0.0
        
        capacity_ = 10
        depth = 50
        self.bids = np.full((capacity_, depth, 2), np.nan, dtype=np.float32)  # [snapshot, level, (price,size)]
        self.asks = np.full((capacity_, depth, 2), np.nan, dtype=np.float32)
        self.ob_imbalance = np.full(capacity_, np.nan, dtype=np.float32)

        self.params_log = set()

    async def _run_synthetic_base(self):
        """Run synthetic base in a separate thread"""
        try:
            loop = asyncio.get_event_loop()
            
            ticker = self.get_ticker()
            if not ticker:
                return
            
            mid_price = (float(ticker["bestBid"]) + float(ticker["bestAsk"])) / 2
            base_count = self.start_base_value / mid_price
            base_count = round(base_count, self.price_scale)
            
            buy_res, sell_res = await loop.run_in_executor(
                None,
                lambda: self.synthetic_base(
                    self.account["client"],
                    self.base,
                    self.quote,
                    base_count,
                    "market",
                    3
                )
            )
            self.logger_strategy.info(f"Synthetic base success: {buy_res} | {sell_res}")
        except Exception as e:
            self.logger_strategy.error(f"Synthetic base error: {e}")
            self.stop_flag = True
        
    async def update_mid_price_chg(self):
        try:
            if self.lasted_mid_price == 0.0:
                self.lasted_mid_price = self.mid_price

            arr = self.mid_price_chg
            arr_mid = self.mid_price_array
            n = arr.shape[0]

            # fast in-place roll: shift view instead of copying
            arr[: n - 1] = arr[1:]
            arr[-1] = self.mid_price - self.lasted_mid_price
            
            arr_mid[: n - 1] = arr_mid[1:]
            arr_mid[-1] = self.mid_price

            self.lasted_mid_price = self.mid_price

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__update_mid_price_chg error: "
                f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
            )

    async def update_mid_price(self):
        try:
            now = int(time.time() * 1000)
            key = f"{self.symbol_redis}_{self.exchange_name}_ticker"
            raw = await self.r.get(key)
            if not raw:
                return

            ticker = json.loads(raw)
            ts = float(ticker.get("ts", 0))
            if now - ts > 500:
                # refresh in background thread
                ticker = await asyncio.to_thread(self.account["client"].get_ticker)

            pb = float(ticker["bestBid"])
            pa = float(ticker["bestAsk"])

            self.best_bid = pb
            self.best_ask = pa
            self.mid_price = (pb + pa) * 0.5  

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__update_mid_price error: "
                f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
            )

    async def update_orderbooks(self):
        try:
            now = int(time.time() * 1000)
            key = f"{self.symbol_redis}_{self.exchange_name}_orderbook"

            raw = await self.r.get(key)
            if not raw:
                return

            books = json.loads(raw)
            ts = float(books.get("ts", 0))

            # Refresh if stale
            if now - ts > 500:
                books = await asyncio.to_thread(self.account["client"].get_orderbook)

            bids_items = books.get("bids", {})
            asks_items = books.get("asks", {})
            if not bids_items or not asks_items:
                return

            depth = 50
            bids_sorted = sorted(((float(p), float(s)) for p, s in bids_items.items()), key=lambda x: -x[0])
            asks_sorted = sorted(((float(p), float(s)) for p, s in asks_items.items()), key=lambda x: x[0])

            # Clone to new arrays (avoid mutating shared buffers)
            bids_new = np.full((depth, 2), np.nan, dtype=np.float32)
            asks_new = np.full((depth, 2), np.nan, dtype=np.float32)

            bids_len = min(depth, len(bids_sorted))
            asks_len = min(depth, len(asks_sorted))
            if bids_len > 0:
                bids_new[:bids_len] = bids_sorted[:bids_len]
            if asks_len > 0:
                asks_new[:asks_len] = asks_sorted[:asks_len]

            # Compute order book imbalance
            ob = ob_imbalance(bids_new, asks_new)

            # Clone arrays first to preserve existing values during roll
            bids_clone = self.bids.copy()
            asks_clone = self.asks.copy()
            imbalance_clone = self.ob_imbalance.copy()

            # Roll arrays in-place
            bids_clone[:-1] = bids_clone[1:]
            asks_clone[:-1] = asks_clone[1:]
            imbalance_clone[:-1] = imbalance_clone[1:]

            # Append new snapshot
            bids_clone[-1] = bids_new
            asks_clone[-1] = asks_new
            imbalance_clone[-1] = ob

            # Commit
            self.bids = bids_clone
            self.asks = asks_clone
            self.ob_imbalance = imbalance_clone

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__update_orderbook error: "
                f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
            )

    async def run_orderbooks_updater(self, interval_ms=1000):
        """Continuously update mid_price at a specified interval (ms)."""
        interval = interval_ms / 1000
        while not getattr(self, "stop_flag", False):
            await self.update_orderbooks()
            await asyncio.sleep(interval)
            
    async def run_mid_price_updater(self, interval_ms=200):
        """Continuously update mid_price at a specified interval (ms)."""
        interval = interval_ms / 1000
        while not getattr(self, "stop_flag", False):
            await self.update_mid_price()
            await asyncio.sleep(interval)


    async def run_mid_price_chg_updater(self, interval_ms=1000):
        """Continuously update mid_price_chg at precise intervals (ms)."""
        interval = interval_ms / 1000
        next_run = time.time() + interval
        while not getattr(self, "stop_flag", False):
            now = time.time()
            if now >= next_run:
                try:
                    await self.update_mid_price_chg()
                except Exception as e:
                    self.logger.error(
                        f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__mid_price_chg_updater error: "
                        f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
                    )
                next_run += interval  # keep fixed timing
            await asyncio.sleep(0.05)

    async def handle_orders(self):
        """
        Cancel expired orders only.
        Concurrent cancellation for speed.
        """
        client = self.account["client"]
        now_ms = int(time.time() * 1000)
        cutoff_ms = self.order_life_time * 1000
        tick_size_move_tolerance = self.tick_move_tolerance * self.tick_price

        try:
            # Reset if mid price moved too far
            if abs(self.mid_price_chg[-1]) >= tick_size_move_tolerance:
                await self.close_old_orders()
                return

            expired, valid = [], []
            for placed_time, _, order_id in self.orders_placed:
                if now_ms - placed_time >= cutoff_ms:
                    expired.append(order_id)
                else:
                    valid.append((placed_time, _, order_id))

            if expired:
                cancel_tasks = [
                    asyncio.to_thread(client.cancel_order, oid) for oid in expired
                ]
                results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

                for oid, res in zip(expired, results):
                    if isinstance(res, Exception):
                        self.logger.error(
                            f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancel_order error: {res}"
                        )
                    else:
                        self.logger.info(
                            f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancelled expired order: {oid}"
                        )

            # keep only valid (non-expired) orders
            self.orders_placed = valid

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__handle_orders error: {e}_{e.__traceback__.tb_lineno}"
            )


    async def update_position(self):
        """
        Update normalized position from account balances and open positions.
        """
        try:
            client = self.account["client"]
            norm_value_entry = self.trade_value or 1e-12  # avoid div by zero

            # parallel requests
            balance_task = asyncio.to_thread(client.get_account_balance)
            positions_task = asyncio.to_thread(client.get_open_positions, category="USDT-FUTURES")
            balance_data, positions_data = await asyncio.gather(balance_task, positions_task)

            balance_data = balance_data.get("data", {})
            positions_data = positions_data.get("data", [])

            total_base_balance = float(balance_data.get(self.base, {}).get("total", 0))
            hedge_base_balance = 0.0
            symbol_upper = f"{self.base}{self.quote}".upper()

            for pos in positions_data:
                if pos.get("symbol", "").upper() == symbol_upper:
                    hedge_base_balance = float(pos.get("total") or pos.get("available") or 0)
                    break

            net_base_pos = total_base_balance - hedge_base_balance
            filled_value = net_base_pos * self.mid_price
            self.net_position = filled_value / norm_value_entry
            
            self.logger.info(f"Net Position: {self.net_position} | Open Orders: {len(self.orders_placed)} | Trade Params (s, k, m, v): {self.params_log}")
            
        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__update_position error: {e}_{e.__traceback__.tb_lineno}"
            )

    async def cancel_all_orders(self, close_all: bool = False):
        """
        Cancel all orders asynchronously.
        If close_all is True, fetch all open orders from the exchange and cancel them.
        Otherwise, cancel orders in self.orders_placed.
        """
        try:
            client = self.account["client"]

            if close_all:
                self.logger.info(
                    f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancel_all_orders")
                try:
                    open_orders = await asyncio.to_thread(client.get_open_orders)
                    order_data = open_orders.get("data", []) if open_orders else []
                    order_ids = [order["orderId"] for order in order_data]

                    for oid in order_ids:
                        self.logger.info(
                            f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancelling order: {oid}"
                        )

                    cancel_tasks = [
                        asyncio.to_thread(client.cancel_order, order_id) for order_id in order_ids
                    ]
                    if cancel_tasks:
                        await asyncio.gather(*cancel_tasks)
                except Exception as e:
                    self.logger.error(
                        f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancel_all_orders fetch error: {e}_{e.__traceback__.tb_lineno}"
                    )
                return

            # Cancel only orders in self.orders_placed
            for _, _, order_id in self.orders_placed:
                self.logger.info(
                    f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancelling order: {order_id}"
                )

            cancel_tasks = [
                asyncio.to_thread(client.cancel_order, order_id)
                for _, _, order_id in self.orders_placed
            ]
            if cancel_tasks:
                await asyncio.gather(*cancel_tasks)

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__cancel_all_orders error: {e}_{e.__traceback__.tb_lineno}"
            )
            
        finally:
            # Clear local orders list
            self.orders_placed.clear()
            
    async def close_old_orders(self):
        """
        Cancel only orders older than self.order_life_time (in seconds).
        Clears local state after.
        """
        client = self.account["client"]
        now_ms = int(time.time() * 1000)
        cutoff_ms = self.order_life_time * 1000

        try:
            self.logger.info(f"{self.symbol_redis}__checking open orders for expiry")

            open_orders = await asyncio.to_thread(client.get_open_orders)
            order_data = open_orders.get("data", []) if open_orders else []

            expired = []
            for o in order_data:
                created_at = o.get("createdAt")
                if created_at and now_ms - created_at >= cutoff_ms:
                    expired.append(o["orderId"])

            if not expired:
                self.logger.info(f"{self.symbol_redis}__no expired orders to cancel")
                return

            self.logger.info(f"{self.symbol_redis}__cancelling {len(expired)} expired orders")

            cancel_tasks = [asyncio.to_thread(client.cancel_order, oid) for oid in expired]
            results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

            for oid, res in zip(expired, results):
                if isinstance(res, Exception):
                    self.logger.error(f"{self.symbol_redis}__cancel error {oid}: {res}")
                else:
                    self.logger.info(f"{self.symbol_redis}__cancelled expired order: {oid}")

        except Exception as e:
            self.logger.error(
                f"{self.symbol_redis}__close_orders_force error: {e}_{e.__traceback__.tb_lineno}"
            )

        finally:
            expired_ids = set(expired)
            self.orders_placed = [
                (t, s, oid) for (t, s, oid) in self.orders_placed if oid not in expired_ids
            ]

    async def run_position_updater(self, interval_ms=1000):
        """Continuously update account position at precise intervals (ms)."""
        interval = interval_ms / 1000
        next_run = time.time() + interval

        while not getattr(self, "stop_flag", False):
            now = time.time()
            if now >= next_run:
                try:
                    await self.update_position()
                except Exception as e:
                    self.logger.error(
                        f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__position_updater error: "
                        f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
                    )
                next_run += interval 
                await self.close_old_orders()
            await asyncio.sleep(0.1)            

    async def run_order_handler(self, interval_ms=1000):
        """Continuously cancel expired orders at precise intervals (ms)."""
        interval = interval_ms / 1000
        next_run = time.time() + interval

        while not getattr(self, "stop_flag", False):
            now = time.time()
            if now >= next_run:
                try:
                    await self.handle_orders()
                except Exception as e:
                    self.logger.error(
                        f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__order_handler error: "
                        f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
                    )
                next_run += interval
            await asyncio.sleep(0.05)

    def get_market_params(self):
        """Compute market parameters for the DMM strategy."""
        tick_size = self.tick_price
        mu = 0.0
        skew = 0.0
        vola = 0.0
        obi_mean = 0.0
        try:
            # Use pre-cleaned array stored once
            mchg = self.mid_price_chg  
            np.nan_to_num(mchg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            
            pos = self.net_position
            skew_factor = self.skew_factor
            volatility_multiplier = self.volatility_multiplier

            half_spread_tick_size = self.half_spread_tick * tick_size
            half_spread_tick_size_max = self.half_spread_tick_max * tick_size

            half_spread, skew, mu, vola = compute_trade_params(
                mchg,
                pos,
                half_spread_tick_size,
                half_spread_tick_size_max,
                skew_factor,
                volatility_multiplier,
                tick_size,
            )
            mp_adj = self.mid_price + skew + mu
            obi_mean = np.nanmean(self.ob_imbalance)
                        
            if half_spread < tick_size:
                half_spread = tick_size

            return mp_adj, half_spread, obi_mean

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__get_market_params error: "
                f"{e}_{getattr(e.__traceback__, 'tb_lineno', '?')}"
            )
            return self.mid_price, self.half_spread_tick * tick_size, 0
        
        finally:
            self.params_log = (half_spread, skew, mu, vola, obi_mean)

            
    async def create_order_params(self, client, order_side, quantity, order_type, price=None):
        """Create a single order param asynchronously."""
        return await asyncio.to_thread(
            client.create_order_params,
            order_side=order_side,
            quantity=quantity,
            price=price,
            order_type=order_type
        )

    async def make_batch_order(self, api_key, private_client, order_params: list):
        """Place batch orders concurrently."""
        try:
            result = await asyncio.to_thread(
                private_client.place_batch_order,
                order_params=order_params
            )

            if result and result.get("code") == "00000":
                order_ids = result["data"]
                now_ms = int(time.time() * 1000)
                for i, oid in enumerate(order_ids):
                    oid = oid['orderId']
                    side = order_params[i]["side"]
                    qty = order_params[i]["qty"]
                    price = order_params[i]["price"]
                    self.orders_placed.append((now_ms, side, oid))
                    self.logger.info(f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__Placed {side} {qty} @ {price} (id={oid})")
                return order_ids
            else:
                self.logger.error(f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__Batch order failed: {result}")
        except Exception as e:
            self.logger.error(f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__make_batch_order error: {e}_{e.__traceback__.tb_lineno}")
        return None


    async def make_order(self, api_key, private_client, order_side, quantity, order_type, price=None):
        """
        Place an order
        """
        try:
            result = await asyncio.to_thread(
                private_client.place_order,
                order_side=order_side,
                quantity=quantity,
                price=price,
                order_type=order_type
            )
            if result and result.get("code") == "00000":
                order_id = result["data"]["orderId"]
                self.orders_placed.append((time.time() * 1000, order_side, order_id))
                self.logger.info(f"Placed order {order_side} {quantity} @ {price} with order_id {result}")
                return order_id
        except Exception as e:
            self.logger.error(f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__make_order error: {e}_{e.__traceback__.tb_lineno}")
        
        return None

    async def check_run_conditions(self):
        """
        Check various conditions to determine whether to run the strategy.
        Stops the strategy and cancels orders if any condition fails.
        """
        try:            
            # Stop flag triggered: cancel all orders
            if self.stop_flag:
                await self.cancel_all_orders(close_all=True)
                return

            # Check run key status
            run_key_status = get_run_key_status(self.run_key) if self.run_key != "terminal" else 1
            if run_key_status != 1:
                self.stop_flag = True
                return

            # Check if current time is within strategy run time
            if not is_run_time_strategy(self.from_time, self.to_end_time):
                await asyncio.sleep(60) 
                return

        except Exception as e:
            self.logger.error(
                f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__check_run_conditions error: {e}_{e.__traceback__.tb_lineno}"
            )
            
    async def run_dmm(self):
        """Run the DMM strategy asynchronously with precise scheduling."""
        self.logger.info(f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__run_dmm started")

        # --- Synthetic Base Initialization ---
        if self.start_base_value > 0:
            await self._run_synthetic_base()

        # --- Background Tasks ---
        asyncio.create_task(self.run_mid_price_updater(10))        # fast market tracking
        asyncio.create_task(self.run_mid_price_chg_updater(990))   # slower momentum tracking
        asyncio.create_task(self.run_position_updater(5000))       # slower account updates
        asyncio.create_task(self.run_order_handler(100))           # cancel stale orders quickly
        asyncio.create_task(self.run_orderbooks_updater(1000))

        # --- Warm-up phase ---
        await asyncio.sleep(5)

        api_key = self.account["api_key"]
        client = self.account["client"]

        allow_taker = self.allow_taker
        min_tick = self.tick_price
        price_scale = self.price_scale
        qty_scale = self.qty_scale

        self.logger.info(
            f"{self.exchange_name}__{self.strategy_name}__{self.symbol_redis}__Starting DMM strategy..."
        )
        self.logger.info(
            f"Allow Taker: {allow_taker} | Min Tick: {min_tick} | Price Scale: {price_scale} | Quantity Scale: {qty_scale}"
        )
        
        cancel_all_interval = self.cancel_all_interval
        new_order_interval = self.new_order_interval
        sleep_time = self.sleep_time
        max_position = self.max_position
        max_orders_pending = self.max_orders_pending
        grid_level_orders = self.grid_level_orders
        grid_level_factor = self.grid_level_factor
        orderbook_imbalance_threshold = self.orderbook_imbalance_threshold

        last_placed = 0.0
        last_cancel_all = 0.0

        try:
            while not getattr(self, "stop_flag", False):
                loop_start = time.monotonic()

                try:
                    if time.monotonic() - last_cancel_all >= cancel_all_interval:
                        await self.cancel_all_orders(close_all=True)
                        last_cancel_all = time.monotonic()
                    
                    if time.monotonic() - last_placed >= new_order_interval:
                        net_pos = self.net_position
                        trade_value = self.trade_value / self.grid_level_orders
                        best_bid, best_ask = self.best_bid, self.best_ask

                        qty = round(trade_value / self.mid_price, qty_scale)

                        mid_price_adj, half_spread_tick, obi = await asyncio.to_thread(self.get_market_params)
                        
                        if not mid_price_adj or len(self.orders_placed) >= max_orders_pending:
                            await asyncio.sleep(sleep_time)
                            continue

                        order_params = []

                        # Precompute prices
                        bid_prices = [
                            round(mid_price_adj - half_spread_tick * (i + 1) * grid_level_factor, price_scale)
                            for i in range(grid_level_orders)
                        ] if net_pos < max_position and obi > - orderbook_imbalance_threshold else []

                        ask_prices = [
                            round(mid_price_adj + half_spread_tick * (i + 1) * grid_level_factor, price_scale)
                            for i in range(grid_level_orders)
                        ] if net_pos > -max_position and obi < orderbook_imbalance_threshold else []

                        # Apply taker constraints
                        if not allow_taker:
                            bid_prices = [min(p, best_ask - min_tick * 2) for p in bid_prices]
                            ask_prices = [max(p, best_bid + min_tick * 2) for p in ask_prices]

                        # Build tasks concurrently
                        tasks = (
                            [self.create_order_params(client, "BUY", qty, "LIMIT", p) for p in bid_prices] +
                            [self.create_order_params(client, "SELL", qty, "LIMIT", p) for p in ask_prices]
                        )

                        # Run all at once
                        order_params = await asyncio.gather(*tasks, return_exceptions=True)

                        if order_params:
                            await self.make_batch_order(api_key, client, order_params)
                            last_placed = time.monotonic()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    tb_line = getattr(e.__traceback__, "tb_lineno", "?")
                    self.logger.error(f"{self.symbol_redis}__run_dmm error: {e} @line {tb_line}")

                # --- Adaptive sleep to maintain precise timing ---
                elapsed = time.monotonic() - loop_start
                if elapsed < sleep_time:
                    await asyncio.sleep(sleep_time - elapsed)

        except KeyboardInterrupt:
            self.logger.info(f"{self.symbol_redis}__run_dmm interrupted")

        finally:
            self.stop_flag = True
            self.logger.info(f"{self.symbol_redis}__run_dmm shutting down, cancelling orders...")
            try:
                await self.cancel_all_orders(close_all=True)
            except Exception as e:
                self.logger.error(f"{self.symbol_redis}__cancel_all_orders failed: {e}")
            self.logger.info(f"{self.symbol_redis}__run_dmm shutdown complete")
