"""
전략 기본 클래스 및 인터페이스

이 모듈은 모든 트레이딩 전략의 기본 인터페이스와 공통 기능을 제공합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd

from src.utils.logger import get_logger


class BaseStrategy(ABC):
    """전략 기본 클래스 (추상 클래스)"""
    
    def __init__(self, name: str = "BaseStrategy", **kwargs):
        """
        Args:
            name: 전략 이름
            **kwargs: 추가 설정
        """
        self.logger = get_logger(f"{__name__}.{name}")
        self.name = name
        self.config = kwargs
        self.is_active = False
        self.trade_history = []
        
    @abstractmethod
    def generate_signal(self, data: Dict[str, Any]) -> str:
        """
        거래 신호 생성 (추상 메서드)
        
        Args:
            data: 시장 데이터 및 신호 정보
                - 'primary_signal': 'BUY', 'SELL', 'HOLD'
                - 'signal_strength': 신호 강도
                - 'confidence': 신뢰도 (0-100)
                - 'indicators': 지표 딕셔너리
                - 'patterns': 패턴 딕셔너리
                - 'timestamp': 타임스탬프
                
        Returns:
            'buy', 'sell', 또는 'hold'
        """
        pass
    
    def validate_signal(self, signal: str, data: Dict[str, Any]) -> bool:
        """
        신호 유효성 검증
        
        Args:
            signal: 생성된 신호
            data: 시장 데이터
            
        Returns:
            신호가 유효한지 여부
        """
        # 기본 검증: 신뢰도 확인
        confidence = data.get('confidence', 0)
        if confidence < 50:
            self.logger.debug(f"신뢰도가 낮아 신호 무시: {confidence}%")
            return False
        
        return True
    
    def calculate_position_size(self, 
                               signal: str, 
                               data: Dict[str, Any],
                               total_capital: float) -> float:
        """
        포지션 크기 계산
        
        Args:
            signal: 거래 신호
            data: 시장 데이터
            total_capital: 총 자본
            
        Returns:
            포지션 크기 (0.0 ~ 1.0)
        """
        if signal == 'hold':
            return 0.0
        
        # 기본 포지션 크기 (총 자본의 10%)
        base_size = 0.1
        
        # 신뢰도에 따른 조정
        confidence = data.get('confidence', 50) / 100.0
        adjusted_size = base_size * confidence
        
        # 최대 30%로 제한
        return min(adjusted_size, 0.3)
    
    def log_trade(self, signal: str, data: Dict[str, Any], executed: bool = False):
        """
        거래 로깅
        
        Args:
            signal: 거래 신호
            data: 시장 데이터
            executed: 실제 실행 여부
        """
        trade_record = {
            'timestamp': data.get('timestamp', datetime.now()),
            'strategy': self.name,
            'signal': signal,
            'confidence': data.get('confidence', 0),
            'signal_strength': data.get('signal_strength', 0),
            'executed': executed
        }
        
        self.trade_history.append(trade_record)
        
        if executed:
            self.logger.info(
                f"거래 실행: {signal.upper()} "
                f"(신뢰도: {data.get('confidence', 0):.1f}%, "
                f"강도: {data.get('signal_strength', 0):.2f})"
            )
        else:
            self.logger.debug(
                f"신호 생성: {signal.upper()} "
                f"(신뢰도: {data.get('confidence', 0):.1f}%)"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        전략 통계 조회
        
        Returns:
            전략 통계 딕셔너리
        """
        if not self.trade_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'executed_trades': 0
            }
        
        df = pd.DataFrame(self.trade_history)
        
        return {
            'total_signals': len(df),
            'buy_signals': len(df[df['signal'] == 'buy']),
            'sell_signals': len(df[df['signal'] == 'sell']),
            'hold_signals': len(df[df['signal'] == 'hold']),
            'executed_trades': len(df[df['executed'] == True]),
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
            'avg_signal_strength': df['signal_strength'].mean() if 'signal_strength' in df.columns else 0
        }
    
    def reset(self):
        """전략 상태 초기화"""
        self.trade_history = []
        self.is_active = False
        self.logger.info(f"전략 '{self.name}' 초기화 완료")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class SimpleStrategy(BaseStrategy):
    """간단한 전략 구현 예제"""
    
    def __init__(self, min_confidence: float = 60.0, **kwargs):
        """
        Args:
            min_confidence: 최소 신뢰도 (0-100)
            **kwargs: 부모 클래스 설정
        """
        super().__init__(name="SimpleStrategy", **kwargs)
        self.min_confidence = min_confidence
    
    def generate_signal(self, data: Dict[str, Any]) -> str:
        """간단한 신호 생성 로직"""
        primary_signal = data.get('primary_signal', 'HOLD')
        confidence = data.get('confidence', 0)
        
        # 신뢰도가 최소값 미만이면 홀드
        if confidence < self.min_confidence:
            return 'hold'
        
        # 신호 변환
        if primary_signal == 'BUY':
            return 'buy'
        elif primary_signal == 'SELL':
            return 'sell'
        else:
            return 'hold'

