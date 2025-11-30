"""
리스크 관리 모듈

이 모듈은 Spot과 Futures 거래 모두를 지원하는 통합 리스크 관리 시스템을 제공합니다.
"""

from src.risk_management.stop_loss import (
    BaseStopLossManager,
    SpotStopLossManager,
    FuturesStopLossManager,
    StopLossType,
    TakeProfitType,
    create_stop_loss_manager
)

from src.risk_management.position_sizing import (
    BasePositionSizer,
    SpotPositionSizer,
    FuturesPositionSizer,
    PositionSizingMethod,
    create_position_sizer
)

from src.risk_management.exposure_manager import (
    ExposureManager,
    PositionExposure
)

__all__ = [
    # Stop Loss
    'BaseStopLossManager',
    'SpotStopLossManager',
    'FuturesStopLossManager',
    'StopLossType',
    'TakeProfitType',
    'create_stop_loss_manager',
    # Position Sizing
    'BasePositionSizer',
    'SpotPositionSizer',
    'FuturesPositionSizer',
    'PositionSizingMethod',
    'create_position_sizer',
    # Exposure Management
    'ExposureManager',
    'PositionExposure',
]

