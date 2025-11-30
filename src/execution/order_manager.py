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
from typing import Dict, Optional, List
import pandas as pd
from dataclasses import dataclass, field

# 리스크 관리 모듈 import
from src.risk_management import (
    create_stop_loss_manager,
    create_position_sizer,
    ExposureManager
)

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
            position_type: PositionType = PositionType.ISOLATED,
            exposure_manager: Optional[ExposureManager] = None
    ):
        """
        Args:
            symbol: 거래 심볼 (예: "BTCUSDT")
            initial_leverage: 초기 레버리지 설정
            position_type: 포지션 모드 설정
            exposure_manager: ExposureManager 인스턴스 (선택적)
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

        # 리스크 관리 모듈 초기화
        self.stop_loss_manager = create_stop_loss_manager(
            trading_type='futures',
            symbol=self.symbol,
            leverage=self.leverage
        )
        self.position_sizer = create_position_sizer(
            trading_type='futures',
            symbol=self.symbol,
            leverage=self.leverage
        )
        
        # ExposureManager 설정 (전달받거나 None)
        self.exposure_manager = exposure_manager

        # 초기화 작업
        self._initialize_leverage()
        self._set_margin_type()

        self.logger.info(f"FuturesOrderManager 초기화: {symbol} (리스크 관리 모듈 통합)")

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

    def _get_symbol_precision(self) -> tuple:
        """
        심볼의 수량 정밀도 정보 조회
        
        Returns:
            (step_size, min_qty, max_qty): stepSize, 최소 수량, 최대 수량
        """
        try:
            # 선물 거래 exchange info 조회
            exchange_info = self.api.client.futures_exchange_info()
            symbol_info = next(
                (s for s in exchange_info['symbols'] if s['symbol'] == self.symbol),
                None
            )
            
            if symbol_info:
                filters = {f['filterType']: f for f in symbol_info['filters']}
                
                # LOT_SIZE 필터에서 stepSize 확인
                if 'LOT_SIZE' in filters:
                    step_size = float(filters['LOT_SIZE']['stepSize'])
                    min_qty = float(filters['LOT_SIZE']['minQty'])
                    max_qty = float(filters['LOT_SIZE']['maxQty'])
                    return step_size, min_qty, max_qty
                
                # PRICE_FILTER에서도 확인 (가격 정밀도)
                if 'PRICE_FILTER' in filters:
                    tick_size = float(filters['PRICE_FILTER']['tickSize'])
                    self.logger.debug(f"가격 정밀도: {tick_size}")
            
            # 기본값 반환 (정밀도 정보를 찾을 수 없는 경우)
            self.logger.warning(f"심볼 {self.symbol}의 정밀도 정보를 찾을 수 없습니다. 기본값 사용")
            return 0.001, 0.001, 1000000.0  # 기본 stepSize
            
        except Exception as e:
            self.logger.warning(f"거래소 정보 조회 실패, 기본값 사용: {e}")
            return 0.001, 0.001, 1000000.0  # 기본 stepSize
    
    def _adjust_quantity_precision(self, quantity: float) -> float:
        """
        수량을 바이낸스 정밀도 요구사항에 맞게 조정
        
        Args:
            quantity: 원본 수량
            
        Returns:
            조정된 수량
        """
        try:
            step_size, min_qty, max_qty = self._get_symbol_precision()
            
            # stepSize에 맞게 수량 조정 (내림)
            adjusted_quantity = (quantity // step_size) * step_size
            
            # 최소 수량 확인
            if adjusted_quantity < min_qty:
                self.logger.warning(
                    f"조정된 수량 {adjusted_quantity}이 최소 수량 {min_qty}보다 작습니다. "
                    f"최소 수량으로 조정합니다."
                )
                adjusted_quantity = min_qty
            
            # 최대 수량 확인
            if adjusted_quantity > max_qty:
                self.logger.warning(
                    f"조정된 수량 {adjusted_quantity}이 최대 수량 {max_qty}보다 큽니다. "
                    f"최대 수량으로 조정합니다."
                )
                adjusted_quantity = max_qty
            
            # 소수점 자릿수 계산 (stepSize의 소수점 자릿수)
            precision = len(str(step_size).rstrip('0').split('.')[-1]) if '.' in str(step_size) else 0
            
            # 최종 수량 반올림 (문자열 포맷팅으로 정밀도 보장)
            if precision > 0:
                adjusted_quantity = float(f"{adjusted_quantity:.{precision}f}")
            else:
                adjusted_quantity = float(int(adjusted_quantity))
            
            self.logger.debug(
                f"수량 정밀도 조정: {quantity} -> {adjusted_quantity} "
                f"(stepSize={step_size}, precision={precision})"
            )
            
            return adjusted_quantity
            
        except Exception as e:
            self.logger.error(f"수량 정밀도 조정 실패: {e}")
            # 기본값으로 반올림 (3자리)
            return round(quantity, 3)
    
    def calculate_position_size(self, total_capital: float, risk_percentage: float = 0.02) -> float:
        """
        포지션 크기 계산 (새 리스크 관리 모듈 사용)

        Args:
            total_capital: 총 자본
            risk_percentage: 리스크 퍼센트 (기본 2%)

        Returns:
            계산된 포지션 크기 (정밀도 조정됨)
        """
        try:
            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])
            
            # 손절가 계산 (새 모듈 사용)
            entry_price = current_price  # 진입 가격은 현재 가격으로 가정
            side = 'LONG'  # 기본값 (실제로는 포지션 방향에 따라 결정)
            stop_loss_price = self.stop_loss_manager.calculate_stop_loss(
                entry_price=entry_price,
                side=side,
                current_price=current_price
            )
            
            # 새 포지션 사이징 모듈 사용
            position_size = self.position_sizer.calculate_position_size(
                total_capital=total_capital,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                risk_per_trade_pct=risk_percentage,
                signal_strength=1.0,
                confidence=0.5
            )
            
            # ExposureManager 체크 (있는 경우)
            if self.exposure_manager:
                can_open, reason = self.exposure_manager.can_open_new_position(
                    symbol=self.symbol,
                    trading_type='futures',
                    position_size=position_size,
                    entry_price=entry_price,
                    leverage=self.leverage
                )
                if not can_open:
                    self.logger.warning(f"ExposureManager: 새 포지션 개설 불가 - {reason}")
                    return 0.0
            
            # 정밀도 조정
            adjusted_quantity = self._adjust_quantity_precision(position_size)
            
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
        """
        선물 포지션 오픈 (ExposureManager 연동)

        Args:
            side: 포지션 방향 (LONG/SHORT)
            total_capital: 총 자본
            risk_percentage: 리스크 퍼센트

        Returns:
            생성된 포지션 객체
        """
        try:
            # 이미 같은 방향 포지션이 있는지 확인
            current_position = self.get_current_position()
            if current_position:
                if current_position.side == side:
                    self.logger.warning(
                        f"이미 {side.name} 포지션이 있습니다. 중복 포지션 오픈 방지: "
                        f"{current_position.quantity:.6f} @ {current_position.entry_price:.2f}"
                    )
                    return current_position
                else:
                    self.logger.warning(
                        f"반대 방향 포지션이 있습니다 ({current_position.side.name}). "
                        f"새로운 {side.name} 포지션 오픈 전에 기존 포지션을 먼저 청산하세요."
                    )
                    return None
            
            current_price = float(self.api.client.futures_symbol_ticker(symbol=self.symbol)['price'])
            
            # 포지션 크기 계산 (정밀도 조정 포함)
            quantity = self.calculate_position_size(total_capital, risk_percentage)
            
            # 수량이 0이면 주문 실행 불가
            if quantity <= 0:
                self.logger.error(f"계산된 수량이 0 이하입니다: {quantity}")
                return None
            
            # ExposureManager: 주문 전 노출 체크 및 추가
            if self.exposure_manager:
                success = self.exposure_manager.add_position(
                    symbol=self.symbol,
                    trading_type='futures',
                    position_size=quantity,
                    entry_price=current_price,
                    current_price=current_price,
                    leverage=self.leverage
                )
                if not success:
                    self.logger.warning("ExposureManager: 노출 제한으로 인해 포지션 오픈 불가")
                    return None
            
            # 최종 수량 정밀도 재확인 (안전장치)
            quantity = self._adjust_quantity_precision(quantity)
            
            self.logger.info(f"주문 실행: {side.name} {quantity} {self.symbol} @ {current_price}")

            # 주문 실행
            try:
                order = self.api.client.futures_create_order(
                    symbol=self.symbol,
                    side=side.value,
                    type='MARKET',
                    quantity=quantity,
                    leverage=int(self.leverage)
                )
            except Exception as order_error:
                # 주문 실패 시 ExposureManager 롤백
                if self.exposure_manager:
                    self.exposure_manager.remove_position(
                        symbol=self.symbol,
                        trading_type='futures'
                    )
                raise order_error

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

            # 손절/익절 가격 설정 (새 모듈 사용)
            side_str = 'LONG' if side == OrderSide.LONG else 'SHORT'
            position.stop_loss_price = self.stop_loss_manager.calculate_stop_loss(
                entry_price=current_price,
                side=side_str,
                current_price=current_price
            )
            position.take_profit_price = self.stop_loss_manager.calculate_take_profit(
                entry_price=current_price,
                side=side_str
            )

            self.current_position = position
            self.logger.info(
                f"포지션 오픈: {side.name} {quantity} {self.symbol} @ {current_price} "
                f"(손절: {position.stop_loss_price:.2f}, 익절: {position.take_profit_price:.2f})"
            )

            return position

        except Exception as e:
            self.logger.error(f"포지션 오픈 실패: {e}")
            return None

    def close_position(self, reason: str = "") -> bool:
        """현재 포지션 청산 (ExposureManager 연동)"""
        if not self.current_position:
            self.logger.warning("청산할 포지션이 없습니다.")
            return False

        try:
            # 반대 방향으로 포지션 청산
            close_side = OrderSide.SHORT if self.current_position.side == OrderSide.LONG else OrderSide.LONG
            
            # 청산 수량 정밀도 조정
            close_quantity = self._adjust_quantity_precision(self.current_position.quantity)
            
            if close_quantity <= 0:
                self.logger.error(f"청산 수량이 0 이하입니다: {close_quantity}")
                return False

            self.logger.info(f"포지션 청산: {close_side.name} {close_quantity} {self.symbol} (이유: {reason})")

            close_order = self.api.client.futures_create_order(
                symbol=self.symbol,
                side=close_side.value,
                type='MARKET',
                quantity=close_quantity,
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

            # ExposureManager: 포지션 제거
            if self.exposure_manager:
                self.exposure_manager.remove_position(
                    symbol=self.symbol,
                    trading_type='futures'
                )

            # 포지션 초기화
            self.current_position = None

            return True

        except Exception as e:
            self.logger.error(f"포지션 청산 실패: {e}")
            return False

    def check_risk_management(self, current_price: float):
        """위험 관리 상태 확인 (새 리스크 관리 모듈 사용)"""
        if not self.current_position:
            return
        
        # 현재 가격이 유효하지 않으면 리턴
        if current_price is None or current_price <= 0:
            return
        
        # 진입 가격이 없으면 리스크 관리 불가
        if self.current_position.entry_price <= 0:
            return
        
        # 포지션 방향을 문자열로 변환
        side_str = 'LONG' if self.current_position.side == OrderSide.LONG else 'SHORT'
        
        # 손절/익절 가격이 설정되지 않은 경우 계산 (새 모듈 사용)
        if self.current_position.stop_loss_price is None or self.current_position.take_profit_price is None:
            self.current_position.stop_loss_price = self.stop_loss_manager.calculate_stop_loss(
                entry_price=self.current_position.entry_price,
                side=side_str,
                current_price=current_price
            )
            self.current_position.take_profit_price = self.stop_loss_manager.calculate_take_profit(
                entry_price=self.current_position.entry_price,
                side=side_str
            )

        # 손절 확인 (새 모듈 사용)
        if self.stop_loss_manager.check_stop_loss(
            current_price=current_price,
            side=side_str,
            entry_price=self.current_position.entry_price
        ):
            self.logger.warning(f"손절 조건 도달: {current_price} (손절가: {self.current_position.stop_loss_price})")
            self.close_position("Stop Loss")
            return
        
        # 익절 확인 (새 모듈 사용)
        if self.stop_loss_manager.check_take_profit(
            current_price=current_price,
            side=side_str,
            entry_price=self.current_position.entry_price
        ):
            self.logger.info(f"익절 조건 도달: {current_price} (익절가: {self.current_position.take_profit_price})")
            self.close_position("Take Profit")
            return
        
        # 청산 리스크 확인 (Futures 전용)
        try:
            # 마진 정보 조회 (간단한 구현)
            account_info = self.api.client.futures_account()
            margin_balance = float(account_info.get('totalWalletBalance', 0))
            margin_used = float(account_info.get('totalInitialMargin', 0))
            
            liquidation_risk = self.stop_loss_manager.check_liquidation_risk(
                current_price=current_price,
                side=side_str,
                entry_price=self.current_position.entry_price,
                margin_used=margin_used,
                margin_balance=margin_balance
            )
            
            if liquidation_risk.get('is_critical', False):
                self.logger.critical(
                    f"청산 리스크 CRITICAL: 청산가까지 {liquidation_risk.get('distance_to_liquidation_pct', 0):.2f}% "
                    f"(청산가: {liquidation_risk.get('liquidation_price', 0):.2f})"
                )
                # CRITICAL 리스크인 경우 즉시 청산 고려
                if liquidation_risk.get('distance_to_liquidation_pct', 100) < 1.0:
                    self.logger.critical("청산 위험이 매우 높습니다. 포지션 청산 권장")
                    self.close_position("Liquidation Risk")
            elif liquidation_risk.get('risk_level') == 'HIGH':
                self.logger.warning(
                    f"청산 리스크 HIGH: 청산가까지 {liquidation_risk.get('distance_to_liquidation_pct', 0):.2f}%"
                )
        except Exception as e:
            self.logger.debug(f"청산 리스크 확인 중 오류 (무시): {e}")

    def get_current_position(self) -> Optional[FuturesPosition]:
        """
        현재 포지션 상태 조회
        실제 계정에서 포지션 정보를 조회하여 동기화합니다.
        """
        try:
            # 실제 계정에서 포지션 정보 조회
            positions = self.api.client.futures_position_information(symbol=self.symbol)
            
            if positions and len(positions) > 0:
                # 포지션 정보가 있는 경우
                pos_data = positions[0]
                position_amt = float(pos_data['positionAmt'])
                
                if abs(position_amt) > 0.0001:  # 포지션이 있는 경우
                    # 실제 포지션 정보로 업데이트
                    if not self.current_position or abs(self.current_position.quantity - abs(position_amt)) > 0.0001:
                        # 포지션이 변경되었거나 새로 생성된 경우
                        side = OrderSide.LONG if position_amt > 0 else OrderSide.SHORT
                        entry_price = float(pos_data['entryPrice'])
                        
                        # 손절/익절 가격 계산 (새 모듈 사용)
                        side_str = 'LONG' if side == OrderSide.LONG else 'SHORT'
                        stop_loss_price = self.stop_loss_manager.calculate_stop_loss(
                            entry_price=entry_price,
                            side=side_str,
                            current_price=entry_price
                        )
                        take_profit_price = self.stop_loss_manager.calculate_take_profit(
                            entry_price=entry_price,
                            side=side_str
                        )
                        
                        self.current_position = FuturesPosition(
                            symbol=self.symbol,
                            side=side,
                            entry_price=entry_price,
                            quantity=abs(position_amt),
                            realized_pnl=float(pos_data.get('unRealizedProfit', 0)),
                            unrealized_pnl=float(pos_data.get('unRealizedProfit', 0)),
                            margin=float(pos_data.get('isolatedMargin', 0)),
                            leverage=self.leverage,
                            position_type=self.position_type,
                            entry_time=datetime.now(),  # 실제 진입 시간은 API에서 가져올 수 없으므로 현재 시간 사용
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price
                        )
                        self.logger.info(
                            f"실제 포지션 정보 동기화: {side.name} {abs(position_amt):.6f} @ {entry_price:.2f} "
                            f"(손절: {stop_loss_price:.2f}, 익절: {take_profit_price:.2f})"
                        )
                    else:
                        # 포지션 정보 업데이트 (손익 등)
                        self.current_position.unrealized_pnl = float(pos_data['unRealizedProfit'])
                        self.current_position.realized_pnl = float(pos_data['unRealizedProfit'])
                    
                    return self.current_position
                else:
                    # 포지션이 없는 경우
                    if self.current_position:
                        self.logger.info("실제 계정에 포지션이 없습니다. 로컬 포지션 정보 초기화")
                        self.current_position = None
                    return None
            else:
                # 포지션 정보가 없는 경우
                if self.current_position:
                    self.logger.info("실제 계정에 포지션 정보가 없습니다. 로컬 포지션 정보 초기화")
                    self.current_position = None
                return None
                
        except Exception as e:
            self.logger.warning(f"실제 포지션 정보 조회 실패, 로컬 포지션 정보 사용: {e}")
            # 조회 실패 시 로컬 포지션 정보 반환
            return self.current_position
    
    def get_current_price(self) -> float:
        """현재 가격 조회 (선물 거래)"""
        try:
            ticker = self.api.client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"선물 거래 현재 가격 조회 실패: {e}")
            return 0.0


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
