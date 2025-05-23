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
    """선물 주문 정보"""
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    side: Optional[OrderSide] = None
    order_type: Optional[OrderType] = None
    quantity: float = 0.0
    price: Optional[float] = None

    # 주문 상태 관련
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_price: float = 0.0

    # 시간 관련
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: Optional[datetime] = None

    # 추가 옵션
    reduce_only: bool = False
    close_position: bool = False
    time_in_force: str = "GTC"


class FuturesOrderManager:
    """바이낸스 선물 거래 주문 관리자"""

    def __init__(
            self,
            symbol: str,
            initial_leverage: float = 1.0,
            position_type: PositionType = PositionType.ISOLATED
    ):
        """
        Args:
            symbol: 거래 심볼 (예: "BTCUSDT")
            initial_leverage: 초기 레버리지 설정
            position_type: 포지션 모드 설정
        """
        self.logger = get_logger(__name__)
        self.api = BinanceAPI()

        # 기본 설정
        self.symbol = symbol
        self.leverage = initial_leverage
        self.position_type = position_type

        # 포지션 및 주문 관리
        self.current_position: Optional[FuturesPosition] = None
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

        # 위험 관리 설정
        self.max_position_size = 0.1  # 총 자본의 10%
        self.stop_loss_percentage = 0.02  # 2% 손절
        self.take_profit_percentage = 0.05  # 5% 익절

        # 초기화 작업
        self._initialize_leverage()
        self._set_margin_type()

        self.logger.info(f"FuturesOrderManager 초기화: {symbol}")

    def _initialize_leverage(self):
        """레버리지 설정"""
        try:
            self.api.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=int(self.leverage)
            )
            self.logger.info(f"{self.symbol} 레버리지 {self.leverage}배 설정")
        except Exception as e:
            self.logger.error(f"레버리지 설정 실패: {e}")

    def _set_margin_type(self):
        """마진 타입 설정"""
        try:
            margin_type = self.position_type.value
            self.api.client.futures_change_margin_type(
                symbol=self.symbol,
                marginType=margin_type
            )
            self.logger.info(f"{self.symbol} 마진 타입: {margin_type}")
        except Exception as e:
            self.logger.error(f"마진 타입 설정 실패: {e}")

    def calculate_position_size(self, total_capital: float, risk_percentage: float = 0.02) -> float:
        """
        포지션 크기 계산

        Args:
            total_capital: 총 자본
            risk_percentage: 리스크 퍼센트 (기본 2%)

        Returns:
            계산된 포지션 크기
        """
        try:
            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])
            max_loss = total_capital * risk_percentage
            position_size = max_loss / (current_price * self.leverage * risk_percentage)

            # 최대 포지션 크기 제한
            max_allowed_size = total_capital * self.max_position_size
            return min(position_size, max_allowed_size)

        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {e}")
            return 0.0

    def open_position(
            self,
            side: OrderSide,
            total_capital: float,
            risk_percentage: float = 0.02
    ) -> Optional[FuturesPosition]:
        """
        선물 포지션 오픈

        Args:
            side: 포지션 방향 (LONG/SHORT)
            total_capital: 총 자본
            risk_percentage: 리스크 퍼센트

        Returns:
            생성된 포지션 객체
        """
        try:
            # 포지션 크기 계산
            quantity = self.calculate_position_size(total_capital, risk_percentage)
            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])

            # 주문 실행
            order = self.api.client.futures_create_order(
                symbol=self.symbol,
                side=side.value,
                type='MARKET',
                quantity=quantity,
                leverage=int(self.leverage)
            )

            # 포지션 객체 생성
            position = FuturesPosition(
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                quantity=quantity,
                leverage=self.leverage,
                position_type=self.position_type,
                entry_time=datetime.now()
            )

            # 손절/익절 가격 설정
            if side == OrderSide.LONG:
                position.stop_loss_price = current_price * (1 - self.stop_loss_percentage)
                position.take_profit_price = current_price * (1 + self.take_profit_percentage)
            else:
                position.stop_loss_price = current_price * (1 + self.stop_loss_percentage)
                position.take_profit_price = current_price * (1 - self.take_profit_percentage)

            self.current_position = position
            self.logger.info(f"포지션 오픈: {side.name} {quantity} {self.symbol} @ {current_price}")

            return position

        except Exception as e:
            self.logger.error(f"포지션 오픈 실패: {e}")
            return None

    def close_position(self, reason: str = "") -> bool:
        """현재 포지션 청산"""
        if not self.current_position:
            self.logger.warning("청산할 포지션이 없습니다.")
            return False

        try:
            # 반대 방향으로 포지션 청산
            close_side = OrderSide.SHORT if self.current_position.side == OrderSide.LONG else OrderSide.LONG

            close_order = self.api.client.futures_create_order(
                symbol=self.symbol,
                side=close_side.value,
                type='MARKET',
                quantity=self.current_position.quantity,
                reduceOnly=True
            )

            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])

            # 실현 손익 계산
            if self.current_position.side == OrderSide.LONG:
                realized_pnl = (current_price - self.current_position.entry_price) * self.current_position.quantity
            else:
                realized_pnl = (self.current_position.entry_price - current_price) * self.current_position.quantity

            self.current_position.realized_pnl = realized_pnl

            self.logger.info(f"포지션 청산: {self.current_position.side.name} {reason}")
            self.logger.info(f"실현 손익: {realized_pnl}")

            # 포지션 초기화
            self.current_position = None

            return True

        except Exception as e:
            self.logger.error(f"포지션 청산 실패: {e}")
            return False

    def check_risk_management(self, current_price: float):
        """위험 관리 상태 확인"""
        if not self.current_position:
            return

            # 손절 확인
        if self.current_position.side == OrderSide.LONG:
            if current_price <= self.current_position.stop_loss_price:
                self.close_position("Stop Loss")
            elif current_price >= self.current_position.take_profit_price:
                self.close_position("Take Profit")
        else:  # SHORT
            if current_price >= self.current_position.stop_loss_price:
                self.close_position("Stop Loss")
            elif current_price <= self.current_position.take_profit_price:
                self.close_position("Take Profit")

    def get_current_position(self) -> Optional[FuturesPosition]:
        """현재 포지션 상태 조회"""
        return self.current_position


if __name__ == "__main__":
    # 사용 예시
    order_manager = FuturesOrderManager(symbol="BTCUSDT", initial_leverage=5)

    # 총 자본 예시 (예: $10,000)
    total_capital = 10000

    try:
        # 롱 포지션 오픈
        long_position = order_manager.open_position(
            side=OrderSide.LONG,
            total_capital=total_capital
        )

        # 현재 포지션 상태 확인
        current_position = order_manager.get_current_position()
        if current_position:
            print(f"포지션 상세: {current_position}")

            # 시뮬레이션을 위한 현재 가격 조회
        current_price = float(order_manager.api.client.futures_symbol_ticker(symbol="BTCUSDT")['price'])

        # 위험 관리 확인
        order_manager.check_risk_management(current_price)

    except Exception as e:
        print(f"오류 발생: {e}")
