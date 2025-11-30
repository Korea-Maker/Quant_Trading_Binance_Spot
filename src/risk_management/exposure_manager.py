"""
최대 노출 제한 관리 모듈

이 모듈은 Spot과 Futures 거래의 통합 노출 제한을 관리합니다.
여러 심볼, 여러 포지션에 걸친 총 노출을 추적하고 제한합니다.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.config.settings import MAX_POSITION_SIZE_PCT


@dataclass
class PositionExposure:
    """포지션 노출 정보"""
    symbol: str
    trading_type: str  # 'spot' or 'futures'
    position_size: float
    entry_price: float
    current_price: float
    notional_value: float  # 명목 가치
    actual_exposure: float  # 실제 노출 (Futures는 마진 기준)
    leverage: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class ExposureManager:
    """통합 노출 관리자"""
    
    def __init__(
        self,
        max_total_exposure_pct: float = 0.3,  # 최대 총 노출 30%
        max_per_symbol_exposure_pct: float = 0.1,  # 심볼당 최대 노출 10%
        max_concurrent_positions: int = 5  # 최대 동시 포지션 수
    ):
        """
        Args:
            max_total_exposure_pct: 최대 총 노출 비율
            max_per_symbol_exposure_pct: 심볼당 최대 노출 비율
            max_concurrent_positions: 최대 동시 포지션 수
        """
        self.logger = get_logger(__name__)
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_per_symbol_exposure_pct = max_per_symbol_exposure_pct
        self.max_concurrent_positions = max_concurrent_positions
        
        # 포지션 추적
        self.positions: Dict[str, PositionExposure] = {}  # key: f"{trading_type}_{symbol}"
        self.total_capital: float = 0.0
    
    def set_total_capital(self, total_capital: float):
        """총 자본 설정"""
        self.total_capital = total_capital
        self.logger.info(f"총 자본 설정: {total_capital:.2f}")
    
    def add_position(
        self,
        symbol: str,
        trading_type: str,
        position_size: float,
        entry_price: float,
        current_price: float,
        leverage: float = 1.0
    ) -> bool:
        """
        포지션 추가
        
        Args:
            symbol: 심볼
            trading_type: 거래 타입
            position_size: 포지션 크기
            entry_price: 진입 가격
            current_price: 현재 가격
            leverage: 레버리지 (Futures만)
            
        Returns:
            추가 성공 여부
        """
        position_key = f"{trading_type}_{symbol}"
        
        # 최대 동시 포지션 수 확인
        if len(self.positions) >= self.max_concurrent_positions:
            self.logger.warning(
                f"최대 동시 포지션 수 초과: {len(self.positions)} >= {self.max_concurrent_positions}"
            )
            return False
        
        # 명목 가치 계산
        notional_value = position_size * current_price
        
        # 실제 노출 계산
        if trading_type == 'futures':
            # Futures: 마진 기준 (명목 가치 / 레버리지)
            actual_exposure = notional_value / leverage
        else:
            # Spot: 명목 가치와 동일
            actual_exposure = notional_value
        
        # 노출 제한 확인
        if not self._check_exposure_limits(symbol, actual_exposure):
            return False
        
        # 포지션 추가
        position = PositionExposure(
            symbol=symbol,
            trading_type=trading_type,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            notional_value=notional_value,
            actual_exposure=actual_exposure,
            leverage=leverage
        )
        
        self.positions[position_key] = position
        self.logger.info(
            f"포지션 추가: {symbol} ({trading_type}) "
            f"크기={position_size:.6f}, 노출={actual_exposure:.2f}"
        )
        
        return True
    
    def remove_position(self, symbol: str, trading_type: str):
        """
        포지션 제거
        
        Args:
            symbol: 심볼
            trading_type: 거래 타입
        """
        position_key = f"{trading_type}_{symbol}"
        if position_key in self.positions:
            position = self.positions.pop(position_key)
            self.logger.info(f"포지션 제거: {symbol} ({trading_type}), 노출={position.actual_exposure:.2f}")
    
    def update_position_price(self, symbol: str, trading_type: str, current_price: float):
        """
        포지션 가격 업데이트
        
        Args:
            symbol: 심볼
            trading_type: 거래 타입
            current_price: 현재 가격
        """
        position_key = f"{trading_type}_{symbol}"
        if position_key in self.positions:
            position = self.positions[position_key]
            position.current_price = current_price
            position.notional_value = position.position_size * current_price
            
            # 실제 노출 재계산
            if trading_type == 'futures':
                position.actual_exposure = position.notional_value / position.leverage
            else:
                position.actual_exposure = position.notional_value
    
    def _check_exposure_limits(self, symbol: str, new_exposure: float) -> bool:
        """
        노출 제한 확인
        
        Args:
            symbol: 심볼
            new_exposure: 새로운 노출
            
        Returns:
            제한 내 여부
        """
        if self.total_capital <= 0:
            self.logger.warning("총 자본이 설정되지 않았습니다.")
            return False
        
        # 총 노출 확인
        current_total_exposure = self.get_total_exposure()
        new_total_exposure = current_total_exposure + new_exposure
        total_exposure_pct = (new_total_exposure / self.total_capital) * 100
        
        if total_exposure_pct > self.max_total_exposure_pct * 100:
            self.logger.warning(
                f"최대 총 노출 초과: {total_exposure_pct:.2f}% > {self.max_total_exposure_pct * 100:.2f}%"
            )
            return False
        
        # 심볼당 노출 확인
        symbol_exposure = self.get_symbol_exposure(symbol) + new_exposure
        symbol_exposure_pct = (symbol_exposure / self.total_capital) * 100
        
        if symbol_exposure_pct > self.max_per_symbol_exposure_pct * 100:
            self.logger.warning(
                f"{symbol} 최대 노출 초과: {symbol_exposure_pct:.2f}% > "
                f"{self.max_per_symbol_exposure_pct * 100:.2f}%"
            )
            return False
        
        return True
    
    def get_total_exposure(self) -> float:
        """총 노출 조회"""
        return sum(pos.actual_exposure for pos in self.positions.values())
    
    def get_total_exposure_pct(self) -> float:
        """총 노출 비율 조회"""
        if self.total_capital <= 0:
            return 0.0
        return (self.get_total_exposure() / self.total_capital) * 100
    
    def get_symbol_exposure(self, symbol: str) -> float:
        """심볼별 노출 조회"""
        return sum(
            pos.actual_exposure
            for pos in self.positions.values()
            if pos.symbol == symbol
        )
    
    def get_trading_type_exposure(self, trading_type: str) -> float:
        """거래 타입별 노출 조회"""
        return sum(
            pos.actual_exposure
            for pos in self.positions.values()
            if pos.trading_type == trading_type
        )
    
    def get_exposure_summary(self) -> Dict[str, any]:
        """
        노출 요약 조회
        
        Returns:
            노출 요약 딕셔너리
        """
        total_exposure = self.get_total_exposure()
        spot_exposure = self.get_trading_type_exposure('spot')
        futures_exposure = self.get_trading_type_exposure('futures')
        
        return {
            'total_capital': self.total_capital,
            'total_exposure': total_exposure,
            'total_exposure_pct': self.get_total_exposure_pct(),
            'spot_exposure': spot_exposure,
            'futures_exposure': futures_exposure,
            'active_positions': len(self.positions),
            'max_concurrent_positions': self.max_concurrent_positions,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'trading_type': pos.trading_type,
                    'position_size': pos.position_size,
                    'notional_value': pos.notional_value,
                    'actual_exposure': pos.actual_exposure,
                    'leverage': pos.leverage
                }
                for pos in self.positions.values()
            ]
        }
    
    def can_open_new_position(
        self,
        symbol: str,
        trading_type: str,
        position_size: float,
        entry_price: float,
        leverage: float = 1.0
    ) -> Tuple[bool, str]:
        """
        새 포지션 개설 가능 여부 확인
        
        Args:
            symbol: 심볼
            trading_type: 거래 타입
            position_size: 포지션 크기
            entry_price: 진입 가격
            leverage: 레버리지
            
        Returns:
            (가능 여부, 이유)
        """
        # 최대 동시 포지션 수 확인
        if len(self.positions) >= self.max_concurrent_positions:
            return False, f"최대 동시 포지션 수 초과: {len(self.positions)}/{self.max_concurrent_positions}"
        
        # 노출 계산
        notional_value = position_size * entry_price
        if trading_type == 'futures':
            actual_exposure = notional_value / leverage
        else:
            actual_exposure = notional_value
        
        # 노출 제한 확인
        if not self._check_exposure_limits(symbol, actual_exposure):
            return False, "노출 제한 초과"
        
        return True, "OK"
    
    def reset(self):
        """모든 포지션 초기화"""
        self.positions.clear()
        self.logger.info("모든 포지션 초기화 완료")


if __name__ == "__main__":
    # 테스트 코드
    logger = get_logger(__name__)
    
    logger.info("=== 노출 관리 테스트 ===")
    manager = ExposureManager(
        max_total_exposure_pct=0.3,
        max_per_symbol_exposure_pct=0.1,
        max_concurrent_positions=5
    )
    
    manager.set_total_capital(10000.0)
    
    # Spot 포지션 추가
    success = manager.add_position(
        symbol="BTCUSDT",
        trading_type="spot",
        position_size=0.01,
        entry_price=50000.0,
        current_price=50000.0,
        leverage=1.0
    )
    logger.info(f"Spot 포지션 추가: {success}")
    
    # Futures 포지션 추가
    success = manager.add_position(
        symbol="BTCUSDT",
        trading_type="futures",
        position_size=0.05,
        entry_price=50000.0,
        current_price=50000.0,
        leverage=5.0
    )
    logger.info(f"Futures 포지션 추가: {success}")
    
    # 요약 조회
    summary = manager.get_exposure_summary()
    logger.info(f"노출 요약: {summary}")

