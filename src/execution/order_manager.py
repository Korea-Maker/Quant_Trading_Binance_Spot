# src/execution/order_manager.py

from src.utils.logger import get_logger
from src.data_collection.binance_api import BinanceAPI
from src.data_collection.websocket_client import BinanceWebSocketClient
from binance.exceptions import BinanceAPIException
from binance.enums import *

import time
import uuid
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
from dataclasses import dataclass, field

class OrderType(Enum):
    """주문 유형"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"

class OrderSide(Enum):
    """선물 주문 방향"""
    LONG = "BUY"
    SHORT = "SELL"

class PositionType(Enum):
    """포지션 유형"""
    ISOLATED = "ISOLATED"
    CROSS = "CROSS"

class OrderStatus(Enum):
    """주문 상태"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class FuturesPosition:
    """선물 포지션 상세 정보"""
    symbol: str
    side: OrderSide
    entry_price: float = 0.0
    quantity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin: float = 0.0
    leverage: float = 0.0
    position_type: PositionType = PositionType.ISOLATED
    margin_type: str = "ISOLATED"
    entry_time: Optional[datetime] = None

    # 손절 / 익절 설정
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    def update(self, current_price: float):
        """현재 가격 기준 포지션 정보 업데이트"""
        if self.side == OrderSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

@dataclass
class Order:
    pass
