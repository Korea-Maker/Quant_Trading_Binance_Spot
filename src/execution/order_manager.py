# src/execution/order_manager.py

from src.utils.logger import get_logger
from src.data_collection.binance_api import BinanceAPI
from src.data_collection.websocket_client import BinanceWebSocketClient
from src.config.settings import TEST_MODE
from binance.exceptions import BinanceAPIException
from binance.enums import *

import time
import uuid
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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

    def __init__(self, symbol: str, initial_leverage: float = 1.0,
                 position_type: PositionType = PositionType.ISOLATED):
        """
        Args:
            symbol: 거래 심볼
            initial_leverage: 초기 레버리지
            position_type: 포지션 타입 (격리/교차)
        """
        self.logger = get_logger(__name__)

        # BinanceAPI는 자동으로 settings.py의 TEST_MODE를 사용
        self.api = BinanceAPI()

        # 현재 테스트 모드 상태 로깅
        test_mode_str = "테스트넷" if TEST_MODE else "실제 환경"
        self.logger.info(f"FuturesOrderManager 초기화 - 환경: {test_mode_str}")

        # 실제 환경에서 추가 경고
        if not TEST_MODE:
            self.logger.warning("⚠️ 실제 거래 환경에서 실행 중입니다! 신중하게 사용하세요.")

        self.symbol = symbol
        self.leverage = initial_leverage
        self.position_type = position_type

        # 심볼 정보 초기화
        self.min_qty = 0.0
        self.max_qty = 0.0
        self.step_size = 0.0
        self.qty_precision = 0
        self.min_price = 0.0
        self.max_price = 0.0
        self.tick_size = 0.0
        self.price_precision = 0

        # 포지션 및 주문 관리
        self.current_position: Optional[FuturesPosition] = None
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

        # 위험 관리 설정 (테스트넷에서 더 보수적으로)
        if TEST_MODE:
            self.max_position_size = 0.05  # 테스트넷: 5%
            self.logger.info("테스트넷 모드 - 보수적 리스크 설정 적용")
        else:
            self.max_position_size = 0.1  # 실제: 10%

        self.stop_loss_percentage = 0.02
        self.take_profit_percentage = 0.05

        # 초기화 작업
        try:
            if self._get_symbol_info():  # 심볼 정보 먼저 조회
                # API 키가 있는 경우에만 레버리지/마진 설정
                if self._has_api_credentials():
                    self._initialize_leverage()
                    self._set_margin_type()
                else:
                    self.logger.warning("API 키가 없어 레버리지/마진 타입 설정을 건너뜁니다.")
            else:
                raise ValueError(f"{symbol} 심볼 정보 조회 실패")
        except Exception as e:
            self.logger.error(f"초기화 중 오류 발생: {e}")
            raise

        self.logger.info(f"FuturesOrderManager 초기화 완료: {symbol}")

    def _has_api_credentials(self) -> bool:
        """API 인증 정보 확인"""
        try:
            # API 키가 있고 계정 정보 조회가 가능한지 확인
            self.api.client.futures_account()
            return True
        except Exception:
            return False

    def _get_symbol_info(self) -> bool:
        """심볼 정보 조회 및 정밀도 설정"""
        try:
            exchange_info = self.api.client.futures_exchange_info()

            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    # 수량 정밀도 확인
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            self.min_qty = float(filter_info['minQty'])
                            self.max_qty = float(filter_info['maxQty'])
                            self.step_size = float(filter_info['stepSize'])

                            # step_size로부터 정밀도 계산
                            self.qty_precision = len(str(self.step_size).split('.')[-1]) if '.' in str(
                                self.step_size) else 0

                        elif filter_info['filterType'] == 'PRICE_FILTER':
                            self.min_price = float(filter_info['minPrice'])
                            self.max_price = float(filter_info['maxPrice'])
                            self.tick_size = float(filter_info['tickSize'])

                            # tick_size로부터 가격 정밀도 계산
                            self.price_precision = len(str(self.tick_size).split('.')[-1]) if '.' in str(
                                self.tick_size) else 0

                    self.logger.info(f"{self.symbol} 수량 정밀도: {self.qty_precision}, 가격 정밀도: {self.price_precision}")
                    return True

            self.logger.error(f"{self.symbol} 정보를 찾을 수 없습니다.")
            return False

        except Exception as e:
            self.logger.error(f"심볼 정보 조회 실패: {e}")
            return False

    def _initialize_leverage(self):
        """레버리지 설정"""
        try:
            result = self.api.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=int(self.leverage)
            )

            env_str = "테스트넷" if TEST_MODE else "실제 환경"
            self.logger.info(f"{env_str} - {self.symbol} 레버리지 {self.leverage}배 설정 완료")

        except Exception as e:
            self.logger.error(f"레버리지 설정 실패: {e}")

    def _set_margin_type(self):
        """마진 타입 설정 (기존 포지션 확인 후 설정)"""
        try:
            # 현재 포지션 확인
            positions = self.api.client.futures_position_information(symbol=self.symbol)
            active_position = None

            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    active_position = pos
                    break

            if active_position:
                self.logger.warning(f"{self.symbol}에 기존 포지션이 있습니다. 마진 타입 변경을 건너뜁니다.")
                return

                # 포지션이 없으면 마진 타입 설정
            margin_type = self.position_type.value
            self.api.client.futures_change_margin_type(
                symbol=self.symbol,
                marginType=margin_type
            )

            env_str = "테스트넷" if TEST_MODE else "실제 환경"
            self.logger.info(f"{env_str} - {self.symbol} 마진 타입: {margin_type}")

        except Exception as e:
            if "No need to change margin type" in str(e):
                self.logger.info(f"{self.symbol} 마진 타입이 이미 설정되어 있습니다.")
            else:
                self.logger.error(f"마진 타입 설정 실패: {e}")

    def _adjust_quantity_precision(self, quantity: float) -> float:
        """수량을 올바른 정밀도로 조정"""
        # step_size에 맞춰 수량 조정
        adjusted_qty = round(quantity / self.step_size) * self.step_size

        # 정밀도에 맞춰 반올림
        adjusted_qty = round(adjusted_qty, self.qty_precision)

        # 최소/최대 수량 확인
        if adjusted_qty < self.min_qty:
            adjusted_qty = self.min_qty
        elif adjusted_qty > self.max_qty:
            adjusted_qty = self.max_qty

        return adjusted_qty

    def calculate_position_size(self, total_capital: float, risk_percentage: float = 0.02) -> float:
        """정밀도를 고려한 포지션 크기 계산"""
        try:
            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])
            max_loss = total_capital * risk_percentage
            position_size = max_loss / (current_price * self.leverage * risk_percentage)

            # 최대 포지션 크기 제한
            max_allowed_size = total_capital * self.max_position_size
            raw_quantity = min(position_size, max_allowed_size)

            # 정밀도 조정
            adjusted_quantity = self._adjust_quantity_precision(raw_quantity)

            self.logger.info(f"계산된 수량: {raw_quantity} -> 조정된 수량: {adjusted_quantity}")
            return adjusted_quantity

        except Exception as e:
            self.logger.error(f"포지션 크기 계산 실패: {e}")
            return 0.0

    def open_position(
            self,
            side: OrderSide,
            total_capital: float,
            risk_percentage: float = 0.02
    ) -> Optional[FuturesPosition]:
        """선물 포지션 오픈"""
        if not self._has_api_credentials():
            self.logger.error("API 인증 정보가 없어 포지션을 열 수 없습니다.")
            return None

        try:
            # 환경 확인 로깅
            env_str = "테스트넷" if TEST_MODE else "실제 환경"
            self.logger.info(f"{env_str}에서 {side.name} 포지션 오픈 시도")

            # 포지션 크기 계산
            quantity = self.calculate_position_size(total_capital, risk_percentage)
            if quantity <= 0:
                self.logger.error("유효하지 않은 포지션 크기")
                return None

            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])

            # 주문 실행
            order = self.api.client.futures_create_order(
                symbol=self.symbol,
                side=side.value,
                type='MARKET',
                quantity=quantity
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
            self.logger.info(f"✅ {env_str} 포지션 오픈 성공: {side.name} {quantity} {self.symbol} @ {current_price}")

            return position

        except Exception as e:
            self.logger.error(f"포지션 오픈 실패: {e}")
            return None

    def close_position(self, reason: str = "") -> bool:
        """현재 포지션 청산"""
        if not self.current_position:
            self.logger.warning("청산할 포지션이 없습니다.")
            return False

        if not self._has_api_credentials():
            self.logger.error("API 인증 정보가 없어 포지션을 청산할 수 없습니다.")
            return False

        try:
            env_str = "테스트넷" if TEST_MODE else "실제 환경"
            self.logger.info(f"{env_str}에서 포지션 청산 시도: {reason}")

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

            self.logger.info(f"✅ {env_str} 포지션 청산 성공: {self.current_position.side.name} {reason}")
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

    def get_account_info(self) -> Optional[Dict]:
        """계정 정보 조회"""
        if not self._has_api_credentials():
            self.logger.warning("API 키가 없어 계정 정보를 조회할 수 없습니다.")
            return None

        try:
            account_info = self.api.client.futures_account()
            env_str = "테스트넷" if TEST_MODE else "실제 환경"
            self.logger.debug(f"{env_str} 계정 정보 조회 성공")
            return account_info
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {e}")
            return None

    def get_position_info(self) -> Optional[List[Dict]]:
        """포지션 정보 조회"""
        if not self._has_api_credentials():
            return None

        try:
            return self.api.client.futures_position_information(symbol=self.symbol)
        except Exception as e:
            self.logger.error(f"포지션 정보 조회 실패: {e}")
            return None


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
