"""
손절/익절 관리 모듈

이 모듈은 Spot과 Futures 거래 모두를 지원하는 통합 손절/익절 관리 시스템을 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

from src.utils.logger import get_logger
from src.config.settings import STOP_LOSS_PCT, TAKE_PROFIT_PCT


class StopLossType(Enum):
    """손절 타입"""
    FIXED = "FIXED"  # 고정 손절
    TRAILING = "TRAILING"  # 트레일링 스톱
    PERCENTAGE = "PERCENTAGE"  # 퍼센트 기반


class TakeProfitType(Enum):
    """익절 타입"""
    FIXED = "FIXED"  # 고정 익절
    PERCENTAGE = "PERCENTAGE"  # 퍼센트 기반
    MULTI_LEVEL = "MULTI_LEVEL"  # 다단계 익절


class BaseStopLossManager(ABC):
    """손절/익절 관리 기본 클래스 (공통 인터페이스)"""
    
    def __init__(
        self,
        symbol: str,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_type: StopLossType = StopLossType.PERCENTAGE,
        take_profit_type: TakeProfitType = TakeProfitType.PERCENTAGE
    ):
        """
        Args:
            symbol: 거래 심볼
            stop_loss_pct: 손절 비율 (기본값: 설정값)
            take_profit_pct: 익절 비율 (기본값: 설정값)
            stop_loss_type: 손절 타입
            take_profit_type: 익절 타입
        """
        self.logger = get_logger(__name__)
        self.symbol = symbol
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss_type = stop_loss_type
        self.take_profit_type = take_profit_type
        
        # 손절/익절 가격
        self.stop_loss_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        
        # 트레일링 스톱 관련
        self.trailing_stop_activated: bool = False
        self.highest_price: Optional[float] = None
        self.trailing_distance_pct: float = 0.01  # 1%
    
    @abstractmethod
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        current_price: Optional[float] = None
    ) -> float:
        """
        손절가 계산
        
        Args:
            entry_price: 진입 가격
            side: 포지션 방향 ('LONG' 또는 'SHORT')
            current_price: 현재 가격 (트레일링 스톱용)
            
        Returns:
            손절가
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str
    ) -> float:
        """
        익절가 계산
        
        Args:
            entry_price: 진입 가격
            side: 포지션 방향 ('LONG' 또는 'SHORT')
            
        Returns:
            익절가
        """
        pass
    
    @abstractmethod
    def check_stop_loss(
        self,
        current_price: float,
        side: str,
        entry_price: float
    ) -> bool:
        """
        손절 조건 확인
        
        Args:
            current_price: 현재 가격
            side: 포지션 방향
            entry_price: 진입 가격
            
        Returns:
            손절 조건 만족 여부
        """
        pass
    
    @abstractmethod
    def check_take_profit(
        self,
        current_price: float,
        side: str,
        entry_price: float
    ) -> bool:
        """
        익절 조건 확인
        
        Args:
            current_price: 현재 가격
            side: 포지션 방향
            entry_price: 진입 가격
            
        Returns:
            익절 조건 만족 여부
        """
        pass
    
    def update_trailing_stop(self, current_price: float, side: str, entry_price: float):
        """
        트레일링 스톱 업데이트
        
        Args:
            current_price: 현재 가격
            side: 포지션 방향
            entry_price: 진입 가격
        """
        if self.stop_loss_type != StopLossType.TRAILING:
            return
        
        if side == 'LONG':
            # LONG 포지션: 최고가 추적
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.trailing_stop_activated = True
            
            # 트레일링 스톱가 계산
            if self.trailing_stop_activated and self.highest_price:
                new_stop_loss = self.highest_price * (1 - self.trailing_distance_pct)
                if new_stop_loss > self.stop_loss_price:
                    self.stop_loss_price = new_stop_loss
                    self.logger.info(
                        f"트레일링 스톱 업데이트: {self.stop_loss_price:.2f} "
                        f"(최고가: {self.highest_price:.2f})"
                    )
        else:  # SHORT
            # SHORT 포지션: 최저가 추적
            if self.highest_price is None or current_price < self.highest_price:
                self.highest_price = current_price
                self.trailing_stop_activated = True
            
            # 트레일링 스톱가 계산
            if self.trailing_stop_activated and self.highest_price:
                new_stop_loss = self.highest_price * (1 + self.trailing_distance_pct)
                if self.stop_loss_price is None or new_stop_loss < self.stop_loss_price:
                    self.stop_loss_price = new_stop_loss
                    self.logger.info(
                        f"트레일링 스톱 업데이트: {self.stop_loss_price:.2f} "
                        f"(최저가: {self.highest_price:.2f})"
                    )
    
    def reset(self):
        """손절/익절 설정 초기화"""
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trailing_stop_activated = False
        self.highest_price = None


class SpotStopLossManager(BaseStopLossManager):
    """Spot 거래용 손절/익절 관리자"""
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        current_price: Optional[float] = None
    ) -> float:
        """Spot 거래 손절가 계산"""
        if self.stop_loss_type == StopLossType.FIXED and self.stop_loss_price:
            return self.stop_loss_price
        
        if side == 'LONG':
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        else:  # SHORT (Spot에서는 매도만 가능)
            stop_loss = entry_price * (1 + self.stop_loss_pct)
        
        self.stop_loss_price = stop_loss
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str
    ) -> float:
        """Spot 거래 익절가 계산"""
        if self.take_profit_type == TakeProfitType.FIXED and self.take_profit_price:
            return self.take_profit_price
        
        if side == 'LONG':
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        self.take_profit_price = take_profit
        return take_profit
    
    def check_stop_loss(
        self,
        current_price: float,
        side: str,
        entry_price: float
    ) -> bool:
        """Spot 거래 손절 조건 확인"""
        if self.stop_loss_price is None:
            self.calculate_stop_loss(entry_price, side)
        
        if side == 'LONG':
            return current_price <= self.stop_loss_price
        else:  # SHORT
            return current_price >= self.stop_loss_price
    
    def check_take_profit(
        self,
        current_price: float,
        side: str,
        entry_price: float
    ) -> bool:
        """Spot 거래 익절 조건 확인"""
        if self.take_profit_price is None:
            self.calculate_take_profit(entry_price, side)
        
        if side == 'LONG':
            return current_price >= self.take_profit_price
        else:  # SHORT
            return current_price <= self.take_profit_price


class FuturesStopLossManager(BaseStopLossManager):
    """Futures 거래용 손절/익절 관리자 (레버리지 고려)"""
    
    def __init__(
        self,
        symbol: str,
        leverage: float = 1.0,
        stop_loss_pct: float = STOP_LOSS_PCT,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_type: StopLossType = StopLossType.PERCENTAGE,
        take_profit_type: TakeProfitType = TakeProfitType.PERCENTAGE,
        liquidation_buffer_pct: float = 0.05  # 청산 방지 버퍼 5%
    ):
        """
        Args:
            symbol: 거래 심볼
            leverage: 레버리지
            stop_loss_pct: 손절 비율
            take_profit_pct: 익절 비율
            stop_loss_type: 손절 타입
            take_profit_type: 익절 타입
            liquidation_buffer_pct: 청산 방지 버퍼 비율
        """
        super().__init__(
            symbol,
            stop_loss_pct,
            take_profit_pct,
            stop_loss_type,
            take_profit_type
        )
        self.leverage = leverage
        self.liquidation_buffer_pct = liquidation_buffer_pct
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        current_price: Optional[float] = None
    ) -> float:
        """Futures 거래 손절가 계산 (레버리지 고려)"""
        if self.stop_loss_type == StopLossType.FIXED and self.stop_loss_price:
            return self.stop_loss_price
        
        # 레버리지를 고려한 손절 비율 조정
        # 레버리지가 높을수록 손절가를 더 가깝게 설정
        adjusted_stop_loss_pct = self.stop_loss_pct / self.leverage
        
        if side == 'LONG':
            stop_loss = entry_price * (1 - adjusted_stop_loss_pct)
            # 청산 방지: 손절가가 청산가보다 충분히 위에 있어야 함
            liquidation_price = entry_price * (1 - (1 / self.leverage) + self.liquidation_buffer_pct)
            stop_loss = max(stop_loss, liquidation_price)
        else:  # SHORT
            stop_loss = entry_price * (1 + adjusted_stop_loss_pct)
            # 청산 방지
            liquidation_price = entry_price * (1 + (1 / self.leverage) - self.liquidation_buffer_pct)
            stop_loss = min(stop_loss, liquidation_price)
        
        self.stop_loss_price = stop_loss
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str
    ) -> float:
        """Futures 거래 익절가 계산"""
        if self.take_profit_type == TakeProfitType.FIXED and self.take_profit_price:
            return self.take_profit_price
        
        if side == 'LONG':
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - self.take_profit_pct)
        
        self.take_profit_price = take_profit
        return take_profit
    
    def check_stop_loss(
        self,
        current_price: float,
        side: str,
        entry_price: float
    ) -> bool:
        """Futures 거래 손절 조건 확인"""
        if self.stop_loss_price is None:
            self.calculate_stop_loss(entry_price, side)
        
        # 트레일링 스톱 업데이트
        if self.stop_loss_type == StopLossType.TRAILING:
            self.update_trailing_stop(current_price, side, entry_price)
        
        if side == 'LONG':
            return current_price <= self.stop_loss_price
        else:  # SHORT
            return current_price >= self.stop_loss_price
    
    def check_take_profit(
        self,
        current_price: float,
        side: str,
        entry_price: float
    ) -> bool:
        """Futures 거래 익절 조건 확인"""
        if self.take_profit_price is None:
            self.calculate_take_profit(entry_price, side)
        
        if side == 'LONG':
            return current_price >= self.take_profit_price
        else:  # SHORT
            return current_price <= self.take_profit_price
    
    def check_liquidation_risk(
        self,
        current_price: float,
        side: str,
        entry_price: float,
        margin_used: float,
        margin_balance: float
    ) -> Dict[str, Any]:
        """
        청산 리스크 확인
        
        Args:
            current_price: 현재 가격
            side: 포지션 방향
            entry_price: 진입 가격
            margin_used: 사용 중인 마진
            margin_balance: 마진 잔액
            
        Returns:
            청산 리스크 정보 딕셔너리
        """
        # 마진 비율 계산
        margin_ratio = (margin_used / margin_balance) * 100 if margin_balance > 0 else 0
        
        # 청산가 계산
        if side == 'LONG':
            liquidation_price = entry_price * (1 - (1 / self.leverage))
            distance_to_liquidation = ((current_price - liquidation_price) / current_price) * 100
        else:  # SHORT
            liquidation_price = entry_price * (1 + (1 / self.leverage))
            distance_to_liquidation = ((liquidation_price - current_price) / current_price) * 100
        
        # 리스크 레벨 판단
        risk_level = "LOW"
        if distance_to_liquidation < 2.0:  # 청산가로부터 2% 이내
            risk_level = "CRITICAL"
        elif distance_to_liquidation < 5.0:  # 5% 이내
            risk_level = "HIGH"
        elif distance_to_liquidation < 10.0:  # 10% 이내
            risk_level = "MEDIUM"
        
        return {
            'liquidation_price': liquidation_price,
            'distance_to_liquidation_pct': distance_to_liquidation,
            'margin_ratio': margin_ratio,
            'risk_level': risk_level,
            'is_critical': risk_level == "CRITICAL"
        }


def create_stop_loss_manager(
    trading_type: str,
    symbol: str,
    **kwargs
) -> BaseStopLossManager:
    """
    거래 타입에 맞는 손절/익절 관리자 생성
    
    Args:
        trading_type: 거래 타입 ('spot' 또는 'futures')
        symbol: 거래 심볼
        **kwargs: 추가 파라미터 (leverage 등)
        
    Returns:
        손절/익절 관리자 인스턴스
    """
    if trading_type == 'futures':
        leverage = kwargs.get('leverage', 1.0)
        return FuturesStopLossManager(symbol=symbol, leverage=leverage, **kwargs)
    else:
        return SpotStopLossManager(symbol=symbol, **kwargs)


if __name__ == "__main__":
    # 테스트 코드
    logger = get_logger(__name__)
    
    # Spot 테스트
    logger.info("=== Spot 손절/익절 테스트 ===")
    spot_manager = SpotStopLossManager(symbol="BTCUSDT")
    entry_price = 50000.0
    
    stop_loss = spot_manager.calculate_stop_loss(entry_price, 'LONG')
    take_profit = spot_manager.calculate_take_profit(entry_price, 'LONG')
    
    logger.info(f"진입가: {entry_price}")
    logger.info(f"손절가: {stop_loss:.2f}")
    logger.info(f"익절가: {take_profit:.2f}")
    
    # Futures 테스트
    logger.info("\n=== Futures 손절/익절 테스트 ===")
    futures_manager = FuturesStopLossManager(symbol="BTCUSDT", leverage=5.0)
    
    stop_loss_f = futures_manager.calculate_stop_loss(entry_price, 'LONG')
    take_profit_f = futures_manager.calculate_take_profit(entry_price, 'LONG')
    
    logger.info(f"진입가: {entry_price}")
    logger.info(f"손절가 (레버리지 5배): {stop_loss_f:.2f}")
    logger.info(f"익절가: {take_profit_f:.2f}")
    
    # 청산 리스크 테스트
    risk_info = futures_manager.check_liquidation_risk(
        current_price=49000.0,
        side='LONG',
        entry_price=entry_price,
        margin_used=1000.0,
        margin_balance=5000.0
    )
    logger.info(f"청산 리스크: {risk_info}")

