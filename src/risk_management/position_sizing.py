"""
포지션 사이징 모듈

이 모듈은 Spot과 Futures 거래 모두를 지원하는 통합 포지션 사이징 시스템을 제공합니다.
레버리지, 마진 요구사항, 최대 노출 제한 등을 고려합니다.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from enum import Enum

from src.utils.logger import get_logger
from src.config.settings import MAX_POSITION_SIZE_PCT


class PositionSizingMethod(Enum):
    """포지션 사이징 방법"""
    FIXED = "FIXED"  # 고정 비율
    KELLY = "KELLY"  # 켈리 공식
    RISK_BASED = "RISK_BASED"  # 리스크 기반
    VOLATILITY_BASED = "VOLATILITY_BASED"  # 변동성 기반


class BasePositionSizer(ABC):
    """포지션 사이징 기본 클래스 (공통 인터페이스)"""
    
    def __init__(
        self,
        symbol: str,
        max_position_size_pct: float = MAX_POSITION_SIZE_PCT,
        max_exposure_pct: float = 0.3,  # 최대 노출 30%
        sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_BASED
    ):
        """
        Args:
            symbol: 거래 심볼
            max_position_size_pct: 최대 포지션 크기 비율 (총 자본 대비)
            max_exposure_pct: 최대 노출 비율 (총 자본 대비)
            sizing_method: 포지션 사이징 방법
        """
        self.logger = get_logger(__name__)
        self.symbol = symbol
        self.max_position_size_pct = max_position_size_pct
        self.max_exposure_pct = max_exposure_pct
        self.sizing_method = sizing_method
        
        # 현재 노출 추적
        self.current_exposure: float = 0.0
        self.active_positions: Dict[str, float] = {}  # 심볼별 포지션 크기
    
    @abstractmethod
    def calculate_position_size(
        self,
        total_capital: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: float = 0.02,  # 거래당 리스크 2%
        signal_strength: float = 1.0,
        confidence: float = 0.5
    ) -> float:
        """
        포지션 크기 계산
        
        Args:
            total_capital: 총 자본
            entry_price: 진입 가격
            stop_loss_price: 손절가
            risk_per_trade_pct: 거래당 리스크 비율
            signal_strength: 신호 강도 (0.0 ~ 1.0)
            confidence: 신뢰도 (0.0 ~ 1.0)
            
        Returns:
            포지션 크기 (수량)
        """
        pass
    
    @abstractmethod
    def check_max_exposure(
        self,
        new_position_size: float,
        total_capital: float
    ) -> bool:
        """
        최대 노출 제한 확인
        
        Args:
            new_position_size: 새로운 포지션 크기
            total_capital: 총 자본
            
        Returns:
            최대 노출 제한 내 여부
        """
        pass
    
    def update_exposure(self, symbol: str, position_size: float):
        """
        노출 업데이트
        
        Args:
            symbol: 심볼
            position_size: 포지션 크기
        """
        if position_size == 0:
            # 포지션 청산 시 제거
            if symbol in self.active_positions:
                del self.active_positions[symbol]
        else:
            self.active_positions[symbol] = position_size
        
        # 총 노출 계산
        self.current_exposure = sum(self.active_positions.values())
    
    def get_current_exposure_pct(self, total_capital: float) -> float:
        """
        현재 노출 비율 조회
        
        Args:
            total_capital: 총 자본
            
        Returns:
            현재 노출 비율
        """
        if total_capital <= 0:
            return 0.0
        return (self.current_exposure / total_capital) * 100
    
    def can_open_new_position(
        self,
        new_position_size: float,
        total_capital: float
    ) -> bool:
        """
        새 포지션 개설 가능 여부 확인
        
        Args:
            new_position_size: 새로운 포지션 크기
            total_capital: 총 자본
            
        Returns:
            새 포지션 개설 가능 여부
        """
        if not self.check_max_exposure(new_position_size, total_capital):
            return False
        
        # 현재 노출 + 새 포지션 크기가 최대 노출을 초과하는지 확인
        new_exposure = self.current_exposure + new_position_size
        max_exposure = total_capital * self.max_exposure_pct
        
        if new_exposure > max_exposure:
            self.logger.warning(
                f"최대 노출 초과: 현재 {self.current_exposure:.2f} + "
                f"신규 {new_position_size:.2f} > 최대 {max_exposure:.2f}"
            )
            return False
        
        return True


class SpotPositionSizer(BasePositionSizer):
    """Spot 거래용 포지션 사이징"""
    
    def calculate_position_size(
        self,
        total_capital: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: float = 0.02,
        signal_strength: float = 1.0,
        confidence: float = 0.5
    ) -> float:
        """Spot 거래 포지션 크기 계산"""
        if self.sizing_method == PositionSizingMethod.FIXED:
            # 고정 비율 방식
            position_value = total_capital * self.max_position_size_pct
            position_size = position_value / entry_price
            
        elif self.sizing_method == PositionSizingMethod.RISK_BASED:
            # 리스크 기반 방식
            risk_amount = total_capital * risk_per_trade_pct
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit <= 0:
                self.logger.warning("손절가가 진입가와 같거나 더 나쁩니다. 포지션 크기 0 반환")
                return 0.0
            
            # 기본 포지션 크기
            base_position_size = risk_amount / risk_per_unit
            
            # 신호 강도와 신뢰도 조정
            adjustment_factor = signal_strength * confidence
            position_size = base_position_size * adjustment_factor
            
            # 최대 포지션 크기 제한
            max_position_value = total_capital * self.max_position_size_pct
            max_position_size = max_position_value / entry_price
            position_size = min(position_size, max_position_size)
            
        elif self.sizing_method == PositionSizingMethod.VOLATILITY_BASED:
            # 변동성 기반 (간단한 구현)
            # 실제로는 ATR 등을 사용해야 함
            volatility_factor = abs(entry_price - stop_loss_price) / entry_price
            risk_amount = total_capital * risk_per_trade_pct
            
            if volatility_factor <= 0:
                return 0.0
            
            # 변동성이 클수록 포지션 크기 감소
            position_value = (risk_amount / volatility_factor) * signal_strength
            position_size = position_value / entry_price
            
            # 최대 제한
            max_position_value = total_capital * self.max_position_size_pct
            max_position_size = max_position_value / entry_price
            position_size = min(position_size, max_position_size)
            
        else:  # KELLY (간단한 구현)
            # 켈리 공식: f = (p * b - q) / b
            # 여기서는 단순화된 버전 사용
            win_rate = confidence  # 신뢰도를 승률로 가정
            avg_win = self.take_profit_pct if hasattr(self, 'take_profit_pct') else 0.05
            avg_loss = risk_per_trade_pct
            
            if avg_loss <= 0:
                return 0.0
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # 최대 25%로 제한
            
            position_value = total_capital * kelly_fraction * signal_strength
            position_size = position_value / entry_price
        
        # 최소 포지션 크기 확인 (거래소 최소 주문량)
        min_position_size = 0.0001  # 예시값, 실제로는 거래소 정보에서 가져와야 함
        if position_size < min_position_size:
            self.logger.warning(f"계산된 포지션 크기 {position_size}가 최소값 {min_position_size}보다 작습니다")
            return 0.0
        
        return position_size
    
    def check_max_exposure(
        self,
        new_position_size: float,
        total_capital: float
    ) -> bool:
        """Spot 거래 최대 노출 제한 확인"""
        if total_capital <= 0:
            return False
        
        position_value = new_position_size * self._get_current_price()
        exposure_pct = (position_value / total_capital) * 100
        
        if exposure_pct > self.max_exposure_pct * 100:
            self.logger.warning(
                f"최대 노출 초과: {exposure_pct:.2f}% > {self.max_exposure_pct * 100:.2f}%"
            )
            return False
        
        return True
    
    def _get_current_price(self) -> float:
        """현재 가격 조회 (간단한 구현)"""
        # 실제로는 API에서 가져와야 함
        return 50000.0  # 예시값


class FuturesPositionSizer(BasePositionSizer):
    """Futures 거래용 포지션 사이징 (레버리지, 마진 고려)"""
    
    def __init__(
        self,
        symbol: str,
        leverage: float = 1.0,
        max_position_size_pct: float = MAX_POSITION_SIZE_PCT,
        max_exposure_pct: float = 0.3,
        sizing_method: PositionSizingMethod = PositionSizingMethod.RISK_BASED,
        margin_maintenance_rate: float = 0.01  # 유지 마진률 1%
    ):
        """
        Args:
            symbol: 거래 심볼
            leverage: 레버리지
            max_position_size_pct: 최대 포지션 크기 비율
            max_exposure_pct: 최대 노출 비율
            sizing_method: 포지션 사이징 방법
            margin_maintenance_rate: 유지 마진률
        """
        super().__init__(symbol, max_position_size_pct, max_exposure_pct, sizing_method)
        self.leverage = leverage
        self.margin_maintenance_rate = margin_maintenance_rate
    
    def calculate_position_size(
        self,
        total_capital: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: float = 0.02,
        signal_strength: float = 1.0,
        confidence: float = 0.5,
        margin_balance: Optional[float] = None
    ) -> float:
        """Futures 거래 포지션 크기 계산 (레버리지, 마진 고려)"""
        # 마진 잔액이 제공되지 않으면 총 자본 사용
        available_margin = margin_balance if margin_balance is not None else total_capital
        
        if self.sizing_method == PositionSizingMethod.FIXED:
            # 고정 비율 방식
            position_value = available_margin * self.max_position_size_pct * self.leverage
            position_size = position_value / entry_price
            
        elif self.sizing_method == PositionSizingMethod.RISK_BASED:
            # 리스크 기반 방식 (레버리지 고려)
            risk_amount = available_margin * risk_per_trade_pct
            risk_per_unit = abs(entry_price - stop_loss_price)
            
            if risk_per_unit <= 0:
                self.logger.warning("손절가가 진입가와 같거나 더 나쁩니다. 포지션 크기 0 반환")
                return 0.0
            
            # 기본 포지션 크기
            base_position_size = risk_amount / risk_per_unit
            
            # 신호 강도와 신뢰도 조정
            adjustment_factor = signal_strength * confidence
            position_size = base_position_size * adjustment_factor
            
            # 레버리지 적용
            # 레버리지가 높을수록 마진 효율이 높아지지만, 리스크도 증가
            # 여기서는 레버리지를 고려하여 포지션 크기 조정
            leverage_adjustment = min(self.leverage / 5.0, 1.0)  # 레버리지 5배를 기준으로 조정
            position_size = position_size * leverage_adjustment
            
            # 마진 요구사항 확인
            required_margin = self._calculate_required_margin(position_size, entry_price)
            if required_margin > available_margin * 0.8:  # 마진의 80% 이상 사용 시 제한
                self.logger.warning(
                    f"마진 부족: 필요 {required_margin:.2f} > 사용가능 {available_margin * 0.8:.2f}"
                )
                # 사용 가능한 마진의 80%로 제한
                max_position_value = available_margin * 0.8 * self.leverage
                position_size = max_position_value / entry_price
            
            # 최대 포지션 크기 제한
            max_position_value = available_margin * self.max_position_size_pct * self.leverage
            max_position_size = max_position_value / entry_price
            position_size = min(position_size, max_position_size)
            
        elif self.sizing_method == PositionSizingMethod.VOLATILITY_BASED:
            # 변동성 기반
            volatility_factor = abs(entry_price - stop_loss_price) / entry_price
            risk_amount = available_margin * risk_per_trade_pct
            
            if volatility_factor <= 0:
                return 0.0
            
            position_value = (risk_amount / volatility_factor) * signal_strength
            position_size = position_value / entry_price
            
            # 마진 확인
            required_margin = self._calculate_required_margin(position_size, entry_price)
            if required_margin > available_margin * 0.8:
                max_position_value = available_margin * 0.8 * self.leverage
                position_size = max_position_value / entry_price
            
        else:  # KELLY
            win_rate = confidence
            avg_win = 0.05  # 5% 익절 가정
            avg_loss = risk_per_trade_pct
            
            if avg_loss <= 0:
                return 0.0
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0.0, min(kelly_fraction, 0.25))
            
            position_value = available_margin * kelly_fraction * signal_strength * self.leverage
            position_size = position_value / entry_price
        
        # 최소 포지션 크기 확인
        min_position_size = 0.0001
        if position_size < min_position_size:
            self.logger.warning(f"계산된 포지션 크기 {position_size}가 최소값 {min_position_size}보다 작습니다")
            return 0.0
        
        return position_size
    
    def _calculate_required_margin(self, position_size: float, entry_price: float) -> float:
        """
        필요 마진 계산
        
        Args:
            position_size: 포지션 크기
            entry_price: 진입 가격
            
        Returns:
            필요 마진
        """
        position_value = position_size * entry_price
        initial_margin = position_value / self.leverage
        maintenance_margin = position_value * self.margin_maintenance_rate
        
        # 초기 마진과 유지 마진 중 큰 값 사용
        return max(initial_margin, maintenance_margin)
    
    def check_max_exposure(
        self,
        new_position_size: float,
        total_capital: float,
        entry_price: Optional[float] = None
    ) -> bool:
        """Futures 거래 최대 노출 제한 확인 (레버리지 고려)"""
        if total_capital <= 0:
            return False
        
        if entry_price is None:
            entry_price = self._get_current_price()
        
        # 레버리지를 고려한 노출 계산
        position_value = new_position_size * entry_price
        notional_exposure = position_value  # 명목 노출
        actual_exposure = position_value / self.leverage  # 실제 마진 사용
        
        # 실제 마진 사용량 기준으로 노출 확인
        exposure_pct = (actual_exposure / total_capital) * 100
        
        if exposure_pct > self.max_exposure_pct * 100:
            self.logger.warning(
                f"최대 노출 초과: {exposure_pct:.2f}% > {self.max_exposure_pct * 100:.2f}% "
                f"(명목 노출: {notional_exposure:.2f}, 레버리지: {self.leverage}x)"
            )
            return False
        
        return True
    
    def check_margin_call_risk(
        self,
        position_size: float,
        entry_price: float,
        current_price: float,
        side: str,
        margin_balance: float,
        margin_used: float
    ) -> Dict[str, Any]:
        """
        마진 콜 리스크 확인
        
        Args:
            position_size: 포지션 크기
            entry_price: 진입 가격
            current_price: 현재 가격
            side: 포지션 방향
            margin_balance: 마진 잔액
            margin_used: 사용 중인 마진
            
        Returns:
            마진 콜 리스크 정보
        """
        position_value = position_size * current_price
        required_margin = self._calculate_required_margin(position_size, current_price)
        
        # 미실현 손익 계산
        if side == 'LONG':
            unrealized_pnl = (current_price - entry_price) * position_size
        else:  # SHORT
            unrealized_pnl = (entry_price - current_price) * position_size
        
        # 마진 비율 계산
        margin_ratio = (margin_used / margin_balance) * 100 if margin_balance > 0 else 0
        
        # 마진 콜 임계값 (일반적으로 80% 이상 시 경고)
        margin_call_threshold = 80.0
        liquidation_threshold = 100.0
        
        risk_level = "LOW"
        if margin_ratio >= liquidation_threshold:
            risk_level = "CRITICAL"
        elif margin_ratio >= margin_call_threshold:
            risk_level = "HIGH"
        elif margin_ratio >= 60.0:
            risk_level = "MEDIUM"
        
        return {
            'margin_ratio': margin_ratio,
            'required_margin': required_margin,
            'margin_used': margin_used,
            'margin_balance': margin_balance,
            'unrealized_pnl': unrealized_pnl,
            'risk_level': risk_level,
            'is_margin_call_risk': margin_ratio >= margin_call_threshold,
            'is_liquidation_risk': margin_ratio >= liquidation_threshold
        }
    
    def _get_current_price(self) -> float:
        """현재 가격 조회 (간단한 구현)"""
        return 50000.0  # 예시값


def create_position_sizer(
    trading_type: str,
    symbol: str,
    **kwargs
) -> BasePositionSizer:
    """
    거래 타입에 맞는 포지션 사이징 관리자 생성
    
    Args:
        trading_type: 거래 타입 ('spot' 또는 'futures')
        symbol: 거래 심볼
        **kwargs: 추가 파라미터 (leverage 등)
        
    Returns:
        포지션 사이징 관리자 인스턴스
    """
    if trading_type == 'futures':
        leverage = kwargs.get('leverage', 1.0)
        return FuturesPositionSizer(symbol=symbol, leverage=leverage, **kwargs)
    else:
        return SpotPositionSizer(symbol=symbol, **kwargs)


if __name__ == "__main__":
    # 테스트 코드
    logger = get_logger(__name__)
    
    # Spot 테스트
    logger.info("=== Spot 포지션 사이징 테스트 ===")
    spot_sizer = SpotPositionSizer(symbol="BTCUSDT")
    
    total_capital = 10000.0
    entry_price = 50000.0
    stop_loss_price = 49000.0
    
    position_size = spot_sizer.calculate_position_size(
        total_capital=total_capital,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        risk_per_trade_pct=0.02,
        signal_strength=1.0,
        confidence=0.7
    )
    
    logger.info(f"총 자본: {total_capital}")
    logger.info(f"진입가: {entry_price}")
    logger.info(f"손절가: {stop_loss_price}")
    logger.info(f"계산된 포지션 크기: {position_size:.6f} BTC")
    logger.info(f"포지션 가치: {position_size * entry_price:.2f} USDT")
    
    # Futures 테스트
    logger.info("\n=== Futures 포지션 사이징 테스트 ===")
    futures_sizer = FuturesPositionSizer(symbol="BTCUSDT", leverage=5.0)
    
    position_size_f = futures_sizer.calculate_position_size(
        total_capital=total_capital,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        risk_per_trade_pct=0.02,
        signal_strength=1.0,
        confidence=0.7,
        margin_balance=total_capital
    )
    
    logger.info(f"총 자본: {total_capital}")
    logger.info(f"레버리지: 5x")
    logger.info(f"계산된 포지션 크기: {position_size_f:.6f} BTC")
    logger.info(f"포지션 가치: {position_size_f * entry_price:.2f} USDT")
    
    required_margin = futures_sizer._calculate_required_margin(position_size_f, entry_price)
    logger.info(f"필요 마진: {required_margin:.2f} USDT")
    
    # 마진 콜 리스크 테스트
    margin_risk = futures_sizer.check_margin_call_risk(
        position_size=position_size_f,
        entry_price=entry_price,
        current_price=49000.0,
        side='LONG',
        margin_balance=total_capital,
        margin_used=required_margin
    )
    logger.info(f"마진 콜 리스크: {margin_risk}")

