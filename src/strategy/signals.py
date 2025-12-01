"""
신호 기반 트레이딩 전략

이 모듈은 TradingSignalProcessor에서 생성된 신호를 기반으로 거래 결정을 내리는 전략들을 구현합니다.
"""

from typing import Dict, Any, Optional
from datetime import datetime

from src.strategy.base import BaseStrategy
from src.data_processing.features import TradingSignalProcessor
from src.data_processing.pattern_recognition import PatternRecognition
from src.utils.logger import get_logger


class SignalBasedStrategy(BaseStrategy):
    """신호 기반 전략"""
    
    def __init__(self, 
                 min_confidence: float = 60.0,
                 min_signal_strength: float = 1.0,
                 use_pattern_confirmation: bool = True,
                 **kwargs):
        """
        Args:
            min_confidence: 최소 신뢰도 (0-100)
            min_signal_strength: 최소 신호 강도
            use_pattern_confirmation: 패턴 확인 사용 여부
            **kwargs: 부모 클래스 설정
        """
        super().__init__(name="SignalBasedStrategy", **kwargs)
        self.min_confidence = min_confidence
        self.min_signal_strength = min_signal_strength
        self.use_pattern_confirmation = use_pattern_confirmation
        
        # 신호 프로세서 초기화
        pattern_recognizer = PatternRecognition()
        self.signal_processor = TradingSignalProcessor(pattern_recognizer)
        
        self.logger.info(
            f"SignalBasedStrategy 초기화 완료 "
            f"(최소 신뢰도: {min_confidence}%, 최소 강도: {min_signal_strength})"
        )
    
    def generate_signal(self, data: Dict[str, Any]) -> str:
        """
        신호 기반 거래 결정
        
        Args:
            data: UnifiedDataProcessor에서 생성된 신호 딕셔너리
                - 'primary_signal': 'BUY', 'SELL', 'HOLD'
                - 'signal_strength': 신호 강도
                - 'confidence': 신뢰도 (0-100)
                - 'indicators': 지표 딕셔너리
                - 'patterns': 패턴 딕셔너리
                - 'timestamp': 타임스탬프
                
        Returns:
            'buy', 'sell', 또는 'hold'
        """
        try:
            primary_signal = data.get('primary_signal', 'HOLD')
            confidence = data.get('confidence', 0)
            signal_strength = abs(data.get('signal_strength', 0))
            
            # 기본 검증
            if confidence < self.min_confidence:
                self.logger.debug(f"신뢰도 부족: {confidence}% < {self.min_confidence}%")
                return 'hold'
            
            if signal_strength < self.min_signal_strength:
                self.logger.debug(f"신호 강도 부족: {signal_strength} < {self.min_signal_strength}")
                return 'hold'
            
            # 패턴 확인 (선택적)
            if self.use_pattern_confirmation:
                if not self._check_pattern_confirmation(data):
                    self.logger.debug("패턴 확인 실패")
                    return 'hold'
            
            # 신호 변환
            if primary_signal == 'BUY':
                return 'buy'
            elif primary_signal == 'SELL':
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            self.logger.error(f"신호 생성 중 오류: {e}", exc_info=True)
            return 'hold'
    
    def _check_pattern_confirmation(self, data: Dict[str, Any]) -> bool:
        """
        패턴 확인
        
        Args:
            data: 시장 데이터
            
        Returns:
            패턴이 신호를 확인하는지 여부
        """
        patterns = data.get('patterns', {})
        primary_signal = data.get('primary_signal', 'HOLD')
        
        if not patterns:
            # 패턴이 없으면 확인 불가 (통과)
            return True
        
        # 매수 신호인 경우 불리시 패턴 확인
        if primary_signal == 'BUY':
            bullish_patterns = [
                'PATTERN_BULLISH_ENGULFING',
                'PATTERN_MORNING_STAR',
                'PATTERN_HAMMER',
                'DOUBLE_BOTTOM',
                'INVERSE_HEAD_AND_SHOULDERS',
                'GOLDEN_CROSS'
            ]
            for pattern in bullish_patterns:
                if pattern in patterns and patterns[pattern] != 0:
                    return True
        
        # 매도 신호인 경우 베어리시 패턴 확인
        elif primary_signal == 'SELL':
            bearish_patterns = [
                'PATTERN_BEARISH_ENGULFING',
                'PATTERN_EVENING_STAR',
                'PATTERN_HANGING_MAN',
                'DOUBLE_TOP',
                'HEAD_AND_SHOULDERS',
                'DEATH_CROSS'
            ]
            for pattern in bearish_patterns:
                if pattern in patterns and patterns[pattern] != 0:
                    return True
        
        # 홀드 신호는 항상 통과
        return True
    
    def calculate_position_size(self, 
                               signal: str, 
                               data: Dict[str, Any],
                               total_capital: float) -> float:
        """
        신호 강도와 신뢰도 기반 포지션 크기 계산
        
        Args:
            signal: 거래 신호
            data: 시장 데이터
            total_capital: 총 자본
            
        Returns:
            포지션 크기 (0.0 ~ 1.0)
        """
        if signal == 'hold':
            return 0.0
        
        # 기본 포지션 크기
        base_size = 0.1  # 10%
        
        # 신뢰도 조정 (50-100% -> 0.5-1.0 배수)
        confidence = data.get('confidence', 50) / 100.0
        confidence_multiplier = 0.5 + (confidence - 0.5) * 1.0  # 50%일 때 0.5배, 100%일 때 1.0배
        
        # 신호 강도 조정
        signal_strength = abs(data.get('signal_strength', 0))
        strength_multiplier = min(signal_strength / 3.0, 1.0)  # 최대 3.0일 때 1.0배
        
        # 최종 포지션 크기
        position_size = base_size * confidence_multiplier * strength_multiplier
        
        # 최대 30%로 제한
        return min(position_size, 0.3)
    
    def adjust_parameters(self, performance_feedback: Dict[str, Any]) -> None:
        """
        성과 피드백에 따라 전략 파라미터 조정
        
        Args:
            performance_feedback: 성과 피드백 딕셔너리
                - 'win_rate': 승률 (0-100)
                - 'recent_performance': 최근 성과 정보
                - 'risk_level': 리스크 레벨 ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
                - 'trading_type': 거래 타입 ('spot' 또는 'futures')
        """
        try:
            win_rate = performance_feedback.get('win_rate', 50.0)
            risk_level = performance_feedback.get('risk_level', 'MEDIUM')
            trading_type = performance_feedback.get('trading_type', 'spot')
            
            # 승률이 낮으면 신뢰도 요구사항 증가 (더 보수적으로)
            if win_rate < 50.0:
                # 승률이 낮을수록 더 보수적으로
                adjustment_factor = (50.0 - win_rate) / 50.0  # 0% 승률일 때 1.0, 50%일 때 0.0
                min_confidence_increase = adjustment_factor * 15.0  # 최대 15% 증가
                self.min_confidence = min(self.min_confidence + min_confidence_increase, 90.0)
                self.min_signal_strength = min(self.min_signal_strength + adjustment_factor * 0.5, 3.0)
                
                self.logger.info(
                    f"성과 저조로 인한 파라미터 조정: "
                    f"min_confidence={self.min_confidence:.1f}%, "
                    f"min_signal_strength={self.min_signal_strength:.2f}"
                )
            
            # 승률이 높으면 신뢰도 요구사항 완화 (선택적, 더 공격적으로)
            elif win_rate > 70.0:
                # 승률이 높을수록 더 공격적으로 (하지만 제한적으로)
                adjustment_factor = (win_rate - 70.0) / 30.0  # 70%일 때 0.0, 100%일 때 1.0
                min_confidence_decrease = adjustment_factor * 5.0  # 최대 5% 감소
                self.min_confidence = max(self.min_confidence - min_confidence_decrease, 50.0)
                self.min_signal_strength = max(self.min_signal_strength - adjustment_factor * 0.2, 0.5)
                
                self.logger.info(
                    f"성과 우수로 인한 파라미터 조정: "
                    f"min_confidence={self.min_confidence:.1f}%, "
                    f"min_signal_strength={self.min_signal_strength:.2f}"
                )
            
            # 리스크 레벨에 따른 조정
            if risk_level == 'HIGH' or risk_level == 'CRITICAL':
                # 리스크가 높으면 더 보수적으로
                self.min_confidence = min(self.min_confidence + 5.0, 90.0)
                self.min_signal_strength = min(self.min_signal_strength + 0.3, 3.0)
                self.logger.info(
                    f"높은 리스크로 인한 파라미터 조정: "
                    f"min_confidence={self.min_confidence:.1f}%, "
                    f"min_signal_strength={self.min_signal_strength:.2f}"
                )
            elif risk_level == 'LOW':
                # 리스크가 낮으면 약간 더 공격적으로
                self.min_confidence = max(self.min_confidence - 2.0, 50.0)
                self.min_signal_strength = max(self.min_signal_strength - 0.1, 0.5)
                self.logger.info(
                    f"낮은 리스크로 인한 파라미터 조정: "
                    f"min_confidence={self.min_confidence:.1f}%, "
                    f"min_signal_strength={self.min_signal_strength:.2f}"
                )
                
        except Exception as e:
            self.logger.error(f"파라미터 조정 중 오류: {e}", exc_info=True)


class ConservativeSignalStrategy(SignalBasedStrategy):
    """보수적인 신호 기반 전략 (높은 신뢰도 요구)"""
    
    def __init__(self, **kwargs):
        super().__init__(
            min_confidence=75.0,
            min_signal_strength=2.0,
            use_pattern_confirmation=True,
            name="ConservativeSignalStrategy",
            **kwargs
        )


class AggressiveSignalStrategy(SignalBasedStrategy):
    """공격적인 신호 기반 전략 (낮은 신뢰도로도 거래)"""
    
    def __init__(self, **kwargs):
        super().__init__(
            min_confidence=50.0,
            min_signal_strength=0.5,
            use_pattern_confirmation=False,
            name="AggressiveSignalStrategy",
            **kwargs
        )
    
    def calculate_position_size(self, 
                               signal: str, 
                               data: Dict[str, Any],
                               total_capital: float) -> float:
        """공격적 전략은 더 큰 포지션 사용"""
        if signal == 'hold':
            return 0.0
        
        # 기본 포지션 크기가 더 큼
        base_size = 0.15  # 15%
        
        confidence = data.get('confidence', 50) / 100.0
        signal_strength = abs(data.get('signal_strength', 0))
        
        position_size = base_size * confidence * min(signal_strength / 2.0, 1.0)
        
        # 최대 40%로 제한 (공격적)
        return min(position_size, 0.4)


class PatternConfirmationStrategy(SignalBasedStrategy):
    """패턴 확인을 강조하는 전략"""
    
    def __init__(self, **kwargs):
        super().__init__(
            min_confidence=65.0,
            min_signal_strength=1.5,
            use_pattern_confirmation=True,
            name="PatternConfirmationStrategy",
            **kwargs
        )
    
    def _check_pattern_confirmation(self, data: Dict[str, Any]) -> bool:
        """더 엄격한 패턴 확인"""
        patterns = data.get('patterns', {})
        primary_signal = data.get('primary_signal', 'HOLD')
        
        if not patterns:
            # 패턴이 없으면 거래하지 않음
            return False
        
        # 최소 2개 이상의 확인 패턴 필요
        confirmation_count = 0
        
        if primary_signal == 'BUY':
            bullish_patterns = [
                'PATTERN_BULLISH_ENGULFING',
                'PATTERN_MORNING_STAR',
                'DOUBLE_BOTTOM',
                'INVERSE_HEAD_AND_SHOULDERS',
                'GOLDEN_CROSS'
            ]
            for pattern in bullish_patterns:
                if pattern in patterns and patterns[pattern] != 0:
                    confirmation_count += 1
        
        elif primary_signal == 'SELL':
            bearish_patterns = [
                'PATTERN_BEARISH_ENGULFING',
                'PATTERN_EVENING_STAR',
                'DOUBLE_TOP',
                'HEAD_AND_SHOULDERS',
                'DEATH_CROSS'
            ]
            for pattern in bearish_patterns:
                if pattern in patterns and patterns[pattern] != 0:
                    confirmation_count += 1
        
        return confirmation_count >= 2

