"""
바이낸스 스팟 거래 주문 관리자

이 모듈은 바이낸스 스팟 거래를 위한 주문 관리 기능을 제공합니다.
"""

from src.utils.logger import get_logger
from src.data_collection.binance_api import BinanceAPI
from src.config.settings import TEST_MODE
from binance.exceptions import BinanceAPIException
from binance.enums import *

import time
import uuid
from enum import Enum
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field


class SpotOrderType(Enum):
    """스팟 주문 유형"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    LIMIT_MAKER = "LIMIT_MAKER"


class SpotOrderSide(Enum):
    """스팟 주문 방향"""
    BUY = "BUY"
    SELL = "SELL"


class SpotOrderStatus(Enum):
    """주문 상태"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class SpotPosition:
    """스팟 포지션 정보"""
    symbol: str
    base_asset: str  # 예: BTC
    quote_asset: str  # 예: USDT
    quantity: float = 0.0  # 보유 수량
    avg_price: float = 0.0  # 평균 매수 가격
    entry_time: Optional[datetime] = None
    
    # 손절/익절 설정
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    
    def get_current_value(self, current_price: float) -> float:
        """현재 포지션 가치 계산"""
        return self.quantity * current_price
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산"""
        if self.avg_price == 0:
            return 0.0
        return (current_price - self.avg_price) * self.quantity
    
    def get_unrealized_pnl_pct(self, current_price: float) -> float:
        """미실현 손익률 계산 (%)"""
        if self.avg_price == 0:
            return 0.0
        return ((current_price - self.avg_price) / self.avg_price) * 100


@dataclass
class SpotOrder:
    """스팟 주문 정보"""
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    side: Optional[SpotOrderSide] = None
    order_type: Optional[SpotOrderType] = None
    quantity: float = 0.0
    price: Optional[float] = None
    
    # 주문 상태 관련
    status: SpotOrderStatus = SpotOrderStatus.NEW
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    
    # 시간 관련
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: Optional[datetime] = None
    
    # 추가 옵션
    time_in_force: str = "GTC"  # GTC, IOC, FOK


class SpotOrderManager:
    """바이낸스 스팟 거래 주문 관리자"""
    
    def __init__(self, symbol: str):
        """
        Args:
            symbol: 거래 심볼 (예: "BTCUSDT")
        """
        self.logger = get_logger(__name__)
        self.api = BinanceAPI()
        
        # 기본 설정
        self.symbol = symbol
        self.base_asset = symbol.replace("USDT", "").replace("BUSD", "")  # 예: BTC
        self.quote_asset = "USDT" if "USDT" in symbol else "BUSD"
        
        # 포지션 및 주문 관리
        self.current_position: Optional[SpotPosition] = None
        self.active_orders: Dict[str, SpotOrder] = {}
        self.order_history: List[SpotOrder] = []
        
        # 위험 관리 설정
        self.max_position_size_pct = 0.1  # 총 자산의 10%
        self.stop_loss_percentage = 0.02  # 2% 손절
        self.take_profit_percentage = 0.05  # 5% 익절
        
        # 테스트넷/메인넷 모드 설정
        self.use_testnet = TEST_MODE
        if self.use_testnet:
            self.logger.info("[바이낸스 테스트넷] 테스트넷에서 실제 주문이 실행됩니다 (메인넷에는 영향 없음)")
        else:
            self.logger.warning("[바이낸스 메인넷] 메인넷에서 실제 주문이 실행됩니다. 주의하세요!")
        
        # 계정 정보 캐시
        self._account_info_cache = None
        self._account_info_cache_time = None
        
        self.logger.info(f"SpotOrderManager 초기화: {symbol}")
    
    def get_account_balance(self, asset: str = None) -> Dict[str, float]:
        """
        계정 잔액 조회
        
        Args:
            asset: 특정 자산만 조회 (None이면 전체)
            
        Returns:
            잔액 딕셔너리 {asset: balance}
        """
        try:
            # 캐시 확인 (5초 이내)
            if (self._account_info_cache and 
                self._account_info_cache_time and 
                (datetime.now() - self._account_info_cache_time).total_seconds() < 5):
                account_info = self._account_info_cache
            else:
                account_info = self.api.client.get_account()
                self._account_info_cache = account_info
                self._account_info_cache_time = datetime.now()
            
            balances = {}
            for balance in account_info['balances']:
                asset_name = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:  # 잔액이 있는 것만
                    balances[asset_name] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
            
            if asset:
                return balances.get(asset, {'free': 0.0, 'locked': 0.0, 'total': 0.0})
            
            return balances
            
        except BinanceAPIException as e:
            error_code = getattr(e, 'code', None)
            if error_code == -2015:
                self.logger.error(
                    f"API 키 오류 (code={error_code}): Invalid API-key, IP, or permissions for action.\n"
                    f"계정 잔액 조회 실패: {e}\n"
                    f"해결 방법:\n"
                    f"1. 바이낸스 {'테스트넷' if self.use_testnet else '메인넷'}에서 API 키가 올바르게 생성되었는지 확인\n"
                    f"2. API 키 권한 확인: 'Enable Reading' 및 'Enable Spot & Margin Trading' 활성화\n"
                    f"3. IP 제한 설정 확인 (필요시 현재 IP 추가 또는 IP 제한 비활성화)\n"
                    f"4. {'테스트넷' if self.use_testnet else '메인넷'} API 키 사용 확인 (혼동 방지)\n"
                    f"5. API 키가 만료되지 않았는지 확인\n"
                    f"6. 테스트넷 URL 확인: https://testnet.binance.vision/"
                )
            else:
                self.logger.error(f"계정 잔액 조회 실패: {e}")
            return {}
    
    def get_current_price(self) -> float:
        """현재 가격 조회"""
        try:
            ticker = self.api.get_ticker(self.symbol)
            return float(ticker['lastPrice'])
        except Exception as e:
            self.logger.error(f"현재 가격 조회 실패: {e}")
            return 0.0
    
    def calculate_buy_quantity(self, 
                              quote_amount: float,
                              current_price: Optional[float] = None) -> float:
        """
        매수 수량 계산
        
        Args:
            quote_amount: 사용할 USDT 금액
            current_price: 현재 가격 (None이면 조회)
            
        Returns:
            매수할 수량
        """
        if current_price is None:
            current_price = self.get_current_price()
        
        if current_price == 0:
            self.logger.error("가격을 조회할 수 없습니다.")
            return 0.0
        
        # 최소 주문 금액 및 수량 확인
        try:
            exchange_info = self.api.get_exchange_info()
            symbol_info = next(
                (s for s in exchange_info['symbols'] if s['symbol'] == self.symbol),
                None
            )
            
            if symbol_info:
                filters = {f['filterType']: f for f in symbol_info['filters']}
                
                # 최소 주문 금액 확인
                if 'MIN_NOTIONAL' in filters:
                    min_notional = float(filters['MIN_NOTIONAL']['minNotional'])
                    if quote_amount < min_notional:
                        self.logger.warning(
                            f"주문 금액이 최소 금액보다 작습니다: "
                            f"{quote_amount} < {min_notional}"
                        )
                        return 0.0
                
                # 수량 정밀도 확인
                if 'LOT_SIZE' in filters:
                    step_size = float(filters['LOT_SIZE']['stepSize'])
                    quantity = (quote_amount / current_price) // step_size * step_size
                    return quantity
        
        except Exception as e:
            self.logger.warning(f"거래소 정보 조회 실패, 기본 계산 사용: {e}")
        
        # 기본 계산
        return quote_amount / current_price
    
    def calculate_sell_quantity(self, 
                               base_quantity: float,
                               sell_percentage: float = 1.0) -> float:
        """
        매도 수량 계산
        
        Args:
            base_quantity: 보유 수량
            sell_percentage: 매도 비율 (0.0 ~ 1.0)
            
        Returns:
            매도할 수량
        """
        try:
            exchange_info = self.api.get_exchange_info()
            symbol_info = next(
                (s for s in exchange_info['symbols'] if s['symbol'] == self.symbol),
                None
            )
            
            if symbol_info:
                filters = {f['filterType']: f for f in symbol_info['filters']}
                
                # 수량 정밀도 확인
                if 'LOT_SIZE' in filters:
                    step_size = float(filters['LOT_SIZE']['stepSize'])
                    sell_quantity = (base_quantity * sell_percentage) // step_size * step_size
                    return sell_quantity
        
        except Exception as e:
            self.logger.warning(f"거래소 정보 조회 실패, 기본 계산 사용: {e}")
        
        # 기본 계산
        return base_quantity * sell_percentage
    
    def place_market_buy_order(self, 
                              quote_amount: float,
                              current_price: Optional[float] = None) -> Optional[SpotOrder]:
        """
        시장가 매수 주문
        
        Args:
            quote_amount: 사용할 USDT 금액
            current_price: 현재 가격 (None이면 조회)
            
        Returns:
            주문 객체 또는 None
        """
        try:
            # 이미 포지션이 있는지 확인
            current_position = self.get_current_position()
            if current_position and current_position.quantity > 0:
                self.logger.warning(
                    f"이미 보유 중입니다. 중복 매수 방지: "
                    f"{current_position.quantity:.6f} {self.base_asset} @ {current_position.avg_price:.2f}"
                )
                return None
            
            # 잔액 확인
            balance = self.get_account_balance(self.quote_asset)
            available = balance.get('free', 0.0)
            
            if available < quote_amount:
                self.logger.error(
                    f"잔액 부족: {available} < {quote_amount} {self.quote_asset}\n"
                    f"해결 방법:\n"
                    f"1. {'테스트넷' if self.use_testnet else '메인넷'} 계정에 충분한 {self.quote_asset} 잔액 확인\n"
                    f"2. {'테스트넷의 경우 테스트 자산 받기: https://testnet.binance.vision/' if self.use_testnet else '메인넷 계정에 자금 입금'}\n"
                    f"3. 계정 잔액 조회 권한 확인 (Enable Reading)"
                )
                return None
            
            # 수량 계산
            quantity = self.calculate_buy_quantity(quote_amount, current_price)
            if quantity == 0:
                return None
            
            # 실제 주문 실행 (테스트넷 또는 메인넷)
            network_type = "테스트넷" if self.use_testnet else "메인넷"
            self.logger.info(f"[바이낸스 {network_type}] 매수 주문 실행: {quote_amount} USDT")
            
            order_response = self.api.client.create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quoteOrderQty=quote_amount  # USDT 금액으로 주문
            )
            
            # 주문 객체 생성
            order = SpotOrder(
                client_order_id=order_response.get('clientOrderId', ''),
                symbol=self.symbol,
                side=SpotOrderSide.BUY,
                order_type=SpotOrderType.MARKET,
                quantity=float(order_response.get('executedQty', 0)),
                status=SpotOrderStatus.FILLED if order_response.get('status') == 'FILLED' else SpotOrderStatus.NEW,
                filled_quantity=float(order_response.get('executedQty', 0)),
                avg_price=float(order_response.get('price', 0)) or self.get_current_price(),
                created_time=datetime.fromtimestamp(order_response.get('transactTime', datetime.now().timestamp() * 1000) / 1000)
            )
            
            # 포지션 업데이트
            self._update_position_after_buy(order)
            
            self.active_orders[order.client_order_id] = order
            self.order_history.append(order)
            
            self.logger.info(
                f"매수 주문 완료: {order.filled_quantity:.6f} {self.base_asset} "
                f"@ {order.avg_price:.2f} {self.quote_asset}"
            )
            
            return order
            
        except BinanceAPIException as e:
            self.logger.error(f"매수 주문 실패: {e}")
            return None
        except Exception as e:
            self.logger.error(f"매수 주문 중 오류: {e}", exc_info=True)
            return None
    
    def place_market_sell_order(self, 
                               quantity: Optional[float] = None,
                               sell_percentage: float = 1.0) -> Optional[SpotOrder]:
        """
        시장가 매도 주문
        
        Args:
            quantity: 매도할 수량 (None이면 보유량의 sell_percentage만큼)
            sell_percentage: 매도 비율 (quantity가 None일 때 사용)
            
        Returns:
            주문 객체 또는 None
        """
        try:
            # 보유량 확인
            balance = self.get_account_balance(self.base_asset)
            available = balance.get('free', 0.0)
            
            if quantity is None:
                quantity = self.calculate_sell_quantity(available, sell_percentage)
            
            if available < quantity:
                self.logger.error(
                    f"보유량 부족: {available} < {quantity} {self.base_asset}"
                )
                return None
            
            if quantity == 0:
                self.logger.warning("매도할 수량이 없습니다.")
                return None
            
            # 실제 주문 실행 (테스트넷 또는 메인넷)
            network_type = "테스트넷" if self.use_testnet else "메인넷"
            self.logger.info(f"[바이낸스 {network_type}] 매도 주문 실행: {quantity} {self.base_asset}")
            
            order_response = self.api.client.create_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # 주문 객체 생성
            order = SpotOrder(
                client_order_id=order_response.get('clientOrderId', ''),
                symbol=self.symbol,
                side=SpotOrderSide.SELL,
                order_type=SpotOrderType.MARKET,
                quantity=float(order_response.get('executedQty', 0)),
                status=SpotOrderStatus.FILLED if order_response.get('status') == 'FILLED' else SpotOrderStatus.NEW,
                filled_quantity=float(order_response.get('executedQty', 0)),
                avg_price=float(order_response.get('price', 0)) or self.get_current_price(),
                created_time=datetime.fromtimestamp(order_response.get('transactTime', datetime.now().timestamp() * 1000) / 1000)
            )
            
            # 포지션 업데이트
            self._update_position_after_sell(order)
            
            self.active_orders[order.client_order_id] = order
            self.order_history.append(order)
            
            self.logger.info(
                f"매도 주문 완료: {order.filled_quantity:.6f} {self.base_asset} "
                f"@ {order.avg_price:.2f} {self.quote_asset}"
            )
            
            return order
            
        except BinanceAPIException as e:
            self.logger.error(f"매도 주문 실패: {e}")
            return None
        except Exception as e:
            self.logger.error(f"매도 주문 중 오류: {e}", exc_info=True)
            return None
    
    def _update_position_after_buy(self, order: SpotOrder):
        """매수 후 포지션 업데이트"""
        if not self.current_position:
            # 새 포지션 생성
            self.current_position = SpotPosition(
                symbol=self.symbol,
                base_asset=self.base_asset,
                quote_asset=self.quote_asset,
                quantity=order.filled_quantity,
                avg_price=order.avg_price,
                entry_time=order.created_time
            )
        else:
            # 기존 포지션에 추가 (평균 가격 재계산)
            total_cost = (self.current_position.quantity * self.current_position.avg_price +
                         order.filled_quantity * order.avg_price)
            total_quantity = self.current_position.quantity + order.filled_quantity
            self.current_position.avg_price = total_cost / total_quantity
            self.current_position.quantity = total_quantity
        
        # 손절/익절 가격 설정
        current_price = self.get_current_price()
        self.current_position.stop_loss_price = current_price * (1 - self.stop_loss_percentage)
        self.current_position.take_profit_price = current_price * (1 + self.take_profit_percentage)
    
    def _update_position_after_sell(self, order: SpotOrder):
        """매도 후 포지션 업데이트"""
        if not self.current_position:
            self.logger.warning("매도할 포지션이 없습니다.")
            return
        
        # 포지션 수량 감소
        self.current_position.quantity -= order.filled_quantity
        
        # 포지션이 모두 매도되면 초기화
        if self.current_position.quantity <= 0.0001:  # 소수점 오차 고려
            realized_pnl = self.current_position.get_unrealized_pnl(order.avg_price)
            self.logger.info(
                f"포지션 청산 완료. 실현 손익: {realized_pnl:.2f} {self.quote_asset} "
                f"({self.current_position.get_unrealized_pnl_pct(order.avg_price):.2f}%)"
            )
            self.current_position = None
    
    def check_risk_management(self, current_price: Optional[float] = None):
        """위험 관리 상태 확인 (손절/익절)"""
        if not self.current_position:
            return
        
        if current_price is None:
            current_price = self.get_current_price()
        
        # 손절 확인
        if (self.current_position.stop_loss_price and 
            current_price <= self.current_position.stop_loss_price):
            self.logger.warning(f"손절가 도달: {current_price} <= {self.current_position.stop_loss_price}")
            self.place_market_sell_order(quantity=self.current_position.quantity)
        
        # 익절 확인
        elif (self.current_position.take_profit_price and 
              current_price >= self.current_position.take_profit_price):
            self.logger.info(f"익절가 도달: {current_price} >= {self.current_position.take_profit_price}")
            self.place_market_sell_order(quantity=self.current_position.quantity)
    
    def get_current_position(self) -> Optional[SpotPosition]:
        """현재 포지션 상태 조회"""
        if self.current_position:
            # 최신 가격으로 업데이트
            current_price = self.get_current_price()
            self.current_position.stop_loss_price = current_price * (1 - self.stop_loss_percentage)
            self.current_position.take_profit_price = current_price * (1 + self.take_profit_percentage)
        
        return self.current_position
    
    def get_order_status(self, order_id: str) -> Optional[SpotOrder]:
        """주문 상태 조회"""
        try:
            order_info = self.api.client.get_order(symbol=self.symbol, orderId=order_id)
            
            order = SpotOrder(
                client_order_id=order_info.get('clientOrderId', ''),
                symbol=self.symbol,
                side=SpotOrderSide.BUY if order_info['side'] == 'BUY' else SpotOrderSide.SELL,
                order_type=SpotOrderType[order_info['type']],
                quantity=float(order_info['origQty']),
                price=float(order_info.get('price', 0)) if order_info.get('price') else None,
                status=SpotOrderStatus[order_info['status']],
                filled_quantity=float(order_info['executedQty']),
                avg_price=float(order_info.get('price', 0)) or float(order_info.get('cummulativeQuoteQty', 0)) / float(order_info['executedQty']) if float(order_info['executedQty']) > 0 else 0,
                created_time=datetime.fromtimestamp(order_info['time'] / 1000)
            )
            
            return order
            
        except BinanceAPIException as e:
            self.logger.error(f"주문 상태 조회 실패: {e}")
            return None


if __name__ == "__main__":
    # 사용 예시
    order_manager = SpotOrderManager(symbol="BTCUSDT")
    
    # 잔액 확인
    balance = order_manager.get_account_balance()
    print(f"계정 잔액: {balance}")
    
    # 현재 가격 확인
    price = order_manager.get_current_price()
    print(f"현재 가격: {price} USDT")
    
    # 테스트 주문 (실제로는 주석 처리)
    # order = order_manager.place_market_buy_order(quote_amount=10.0)  # 10 USDT로 매수
    # if order:
    #     print(f"주문 완료: {order}")

