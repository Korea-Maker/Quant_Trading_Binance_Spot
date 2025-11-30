"""
단계별 리스크 체크 시스템

이 모듈은 각 프로세스 단계별로 리스크를 체크하는 시스템을 제공합니다.
이미지 2의 리스크체크1-5에 해당하는 기능을 구현합니다.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime

from src.utils.logger import get_logger
from src.config.settings import TRADING_TYPE


class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskCheckResult:
    """리스크 체크 결과"""
    
    def __init__(
        self,
        passed: bool,
        risk_level: RiskLevel,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            passed: 체크 통과 여부
            risk_level: 리스크 레벨
            message: 메시지
            details: 상세 정보
        """
        self.passed = passed
        self.risk_level = risk_level
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def __bool__(self):
        return self.passed
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.risk_level.value}: {self.message}"


class BaseRiskChecker:
    """리스크 체크 기본 클래스"""
    
    def __init__(self, trading_type: str = TRADING_TYPE):
        """
        Args:
            trading_type: 거래 타입 ('spot' 또는 'futures')
        """
        self.logger = get_logger(__name__)
        self.trading_type = trading_type
    
    def check(self, data: Dict[str, Any]) -> RiskCheckResult:
        """
        리스크 체크 수행
        
        Args:
            data: 체크할 데이터
            
        Returns:
            리스크 체크 결과
        """
        raise NotImplementedError


class DataCollectionRiskChecker(BaseRiskChecker):
    """리스크체크1: 데이터 수집 단계 리스크 체크"""
    
    def check(self, data: Dict[str, Any]) -> RiskCheckResult:
        """데이터 수집 단계 리스크 체크"""
        issues = []
        risk_level = RiskLevel.LOW
        
        # 데이터 품질 체크
        if 'price' in data:
            price = data.get('price')
            if price is None or price <= 0:
                issues.append("가격 데이터가 유효하지 않음")
                risk_level = RiskLevel.CRITICAL
        
        # 연결 상태 체크
        connection_status = data.get('connection_status', 'unknown')
        if connection_status != 'connected':
            issues.append(f"연결 상태 불량: {connection_status}")
            risk_level = RiskLevel.HIGH
        
        # 데이터 지연 체크
        data_delay = data.get('data_delay_ms', 0)
        if data_delay > 1000:  # 1초 이상 지연
            issues.append(f"데이터 지연: {data_delay}ms")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 오더북 데이터 체크
        if 'orderbook' in data:
            orderbook = data.get('orderbook')
            if not orderbook or len(orderbook.get('bids', [])) == 0:
                issues.append("오더북 데이터 부족")
                risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        passed = len(issues) == 0
        message = "데이터 수집 정상" if passed else "; ".join(issues)
        
        return RiskCheckResult(
            passed=passed,
            risk_level=risk_level,
            message=message,
            details={
                'issues': issues,
                'data_delay_ms': data_delay,
                'connection_status': connection_status
            }
        )


class PreprocessingRiskChecker(BaseRiskChecker):
    """리스크체크2: 데이터 전처리 단계 리스크 체크"""
    
    def check(self, data: Dict[str, Any]) -> RiskCheckResult:
        """데이터 전처리 단계 리스크 체크"""
        issues = []
        risk_level = RiskLevel.LOW
        
        # 결측치 체크
        missing_data = data.get('missing_data_count', 0)
        total_data = data.get('total_data_count', 0)
        if total_data > 0:
            missing_ratio = missing_data / total_data
            if missing_ratio > 0.1:  # 10% 이상 결측
                issues.append(f"결측치 비율 높음: {missing_ratio:.2%}")
                risk_level = RiskLevel.HIGH
            elif missing_ratio > 0.05:  # 5% 이상 결측
                issues.append(f"결측치 비율: {missing_ratio:.2%}")
                risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 이상치 체크
        outlier_count = data.get('outlier_count', 0)
        if outlier_count > 0:
            outlier_ratio = outlier_count / total_data if total_data > 0 else 0
            if outlier_ratio > 0.05:  # 5% 이상 이상치
                issues.append(f"이상치 비율 높음: {outlier_ratio:.2%}")
                risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 데이터 일관성 체크
        data_consistency = data.get('data_consistency', True)
        if not data_consistency:
            issues.append("데이터 일관성 문제")
            risk_level = RiskLevel.HIGH
        
        passed = len(issues) == 0
        message = "데이터 전처리 정상" if passed else "; ".join(issues)
        
        return RiskCheckResult(
            passed=passed,
            risk_level=risk_level,
            message=message,
            details={
                'issues': issues,
                'missing_data_count': missing_data,
                'outlier_count': outlier_count,
                'data_consistency': data_consistency
            }
        )


class IndicatorRiskChecker(BaseRiskChecker):
    """리스크체크3: 기술지표 계산 단계 리스크 체크"""
    
    def check(self, data: Dict[str, Any]) -> RiskCheckResult:
        """기술지표 계산 단계 리스크 체크"""
        issues = []
        risk_level = RiskLevel.LOW
        
        # 지표 계산 성공 여부
        indicators = data.get('indicators', {})
        if not indicators:
            issues.append("지표 데이터 없음")
            risk_level = RiskLevel.CRITICAL
        
        # 지표 신뢰도 체크
        indicator_reliability = data.get('indicator_reliability', 1.0)
        if indicator_reliability < 0.5:
            issues.append(f"지표 신뢰도 낮음: {indicator_reliability:.2f}")
            risk_level = RiskLevel.HIGH
        elif indicator_reliability < 0.7:
            issues.append(f"지표 신뢰도: {indicator_reliability:.2f}")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 필수 지표 존재 여부
        required_indicators = ['RSI', 'MACD', 'MA']
        missing_indicators = [
            ind for ind in required_indicators
            if ind not in indicators
        ]
        if missing_indicators:
            issues.append(f"필수 지표 누락: {', '.join(missing_indicators)}")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 지표 값 유효성 체크
        for ind_name, ind_value in indicators.items():
            if ind_value is None or (isinstance(ind_value, float) and (ind_value < -1000 or ind_value > 1000)):
                issues.append(f"지표 {ind_name} 값 이상: {ind_value}")
                risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        passed = len(issues) == 0
        message = "기술지표 계산 정상" if passed else "; ".join(issues)
        
        return RiskCheckResult(
            passed=passed,
            risk_level=risk_level,
            message=message,
            details={
                'issues': issues,
                'indicator_reliability': indicator_reliability,
                'indicators_count': len(indicators)
            }
        )


class PatternRecognitionRiskChecker(BaseRiskChecker):
    """리스크체크4: 패턴 인식 단계 리스크 체크"""
    
    def check(self, data: Dict[str, Any]) -> RiskCheckResult:
        """패턴 인식 단계 리스크 체크"""
        issues = []
        risk_level = RiskLevel.LOW
        
        # 패턴 데이터 존재 여부
        patterns = data.get('patterns', {})
        if not patterns:
            issues.append("패턴 데이터 없음")
            risk_level = RiskLevel.MEDIUM  # 패턴이 없어도 거래 가능
        
        # 패턴 신뢰도 체크
        pattern_confidence = data.get('pattern_confidence', 0.0)
        if pattern_confidence < 0.3:
            issues.append(f"패턴 신뢰도 낮음: {pattern_confidence:.2f}")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 패턴 충돌 체크
        bullish_patterns = sum(1 for p in patterns.values() if isinstance(p, (int, float)) and p > 0)
        bearish_patterns = sum(1 for p in patterns.values() if isinstance(p, (int, float)) and p < 0)
        
        if bullish_patterns > 0 and bearish_patterns > 0:
            issues.append("패턴 충돌: 상승/하락 패턴 동시 존재")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        passed = len(issues) == 0 or risk_level == RiskLevel.LOW
        message = "패턴 인식 정상" if passed else "; ".join(issues)
        
        return RiskCheckResult(
            passed=passed,
            risk_level=risk_level,
            message=message,
            details={
                'issues': issues,
                'pattern_confidence': pattern_confidence,
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns
            }
        )


class SignalGenerationRiskChecker(BaseRiskChecker):
    """리스크체크5: 신호 생성 단계 리스크 체크"""
    
    def check(self, data: Dict[str, Any]) -> RiskCheckResult:
        """신호 생성 단계 리스크 체크"""
        issues = []
        risk_level = RiskLevel.LOW
        
        # 신호 존재 여부
        signal = data.get('primary_signal', 'HOLD')
        if signal == 'HOLD':
            # HOLD 신호는 리스크가 낮음
            return RiskCheckResult(
                passed=True,
                risk_level=RiskLevel.LOW,
                message="HOLD 신호 - 거래 없음",
                details={'signal': signal}
            )
        
        # 신호 강도 체크
        signal_strength = abs(data.get('signal_strength', 0))
        if signal_strength < 0.5:
            issues.append(f"신호 강도 낮음: {signal_strength:.2f}")
            risk_level = RiskLevel.HIGH
        
        # 신뢰도 체크
        confidence = data.get('confidence', 0)
        if confidence < 50:
            issues.append(f"신호 신뢰도 낮음: {confidence}%")
            risk_level = RiskLevel.HIGH
        elif confidence < 70:
            issues.append(f"신호 신뢰도: {confidence}%")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        
        # 시장 상태 체크
        market_condition = data.get('market_condition', 'normal')
        if market_condition == 'volatile':
            issues.append("시장 변동성 높음")
            risk_level = RiskLevel.MEDIUM if risk_level == RiskLevel.LOW else risk_level
        elif market_condition == 'extreme':
            issues.append("시장 극단적 상태")
            risk_level = RiskLevel.CRITICAL
        
        # Futures 특화 체크
        if self.trading_type == 'futures':
            # 레버리지 리스크 체크
            leverage = data.get('leverage', 1.0)
            if leverage > 10:
                issues.append(f"레버리지 높음: {leverage}x")
                risk_level = RiskLevel.HIGH if risk_level == RiskLevel.LOW else risk_level
            
            # 마진 상태 체크
            margin_ratio = data.get('margin_ratio', 0)
            if margin_ratio > 80:
                issues.append(f"마진 비율 높음: {margin_ratio:.2f}%")
                risk_level = RiskLevel.CRITICAL
            elif margin_ratio > 60:
                issues.append(f"마진 비율: {margin_ratio:.2f}%")
                risk_level = RiskLevel.HIGH if risk_level == RiskLevel.LOW else risk_level
        
        passed = len(issues) == 0 or risk_level == RiskLevel.LOW
        message = "신호 생성 정상" if passed else "; ".join(issues)
        
        return RiskCheckResult(
            passed=passed,
            risk_level=risk_level,
            message=message,
            details={
                'issues': issues,
                'signal': signal,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'market_condition': market_condition
            }
        )


class IntegratedRiskChecker:
    """통합 리스크 체커 (모든 단계 체크)"""
    
    def __init__(self, trading_type: str = TRADING_TYPE):
        """
        Args:
            trading_type: 거래 타입
        """
        self.logger = get_logger(__name__)
        self.trading_type = trading_type
        
        # 각 단계별 체커 초기화
        self.checkers = {
            'data_collection': DataCollectionRiskChecker(trading_type),
            'preprocessing': PreprocessingRiskChecker(trading_type),
            'indicators': IndicatorRiskChecker(trading_type),
            'pattern_recognition': PatternRecognitionRiskChecker(trading_type),
            'signal_generation': SignalGenerationRiskChecker(trading_type)
        }
    
    def check_all(self, data: Dict[str, Any]) -> Dict[str, RiskCheckResult]:
        """
        모든 단계 리스크 체크
        
        Args:
            data: 체크할 데이터 (단계별 데이터 포함)
            
        Returns:
            단계별 체크 결과 딕셔너리
        """
        results = {}
        
        # 각 단계별 체크
        for stage, checker in self.checkers.items():
            stage_data = data.get(stage, {})
            result = checker.check(stage_data)
            results[stage] = result
            
            if not result.passed:
                self.logger.warning(f"[{stage}] {result}")
            else:
                self.logger.debug(f"[{stage}] {result}")
        
        return results
    
    def check_stage(self, stage: str, data: Dict[str, Any]) -> RiskCheckResult:
        """
        특정 단계 리스크 체크
        
        Args:
            stage: 단계명
            data: 체크할 데이터
            
        Returns:
            체크 결과
        """
        if stage not in self.checkers:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                message=f"알 수 없는 단계: {stage}"
            )
        
        checker = self.checkers[stage]
        return checker.check(data)
    
    def should_proceed(self, results: Dict[str, RiskCheckResult]) -> bool:
        """
        다음 단계 진행 가능 여부 판단
        
        Args:
            results: 체크 결과 딕셔너리
            
        Returns:
            진행 가능 여부
        """
        # CRITICAL 리스크가 있으면 중단
        for result in results.values():
            if result.risk_level == RiskLevel.CRITICAL and not result.passed:
                return False
        
        # HIGH 리스크가 2개 이상이면 중단
        high_risk_count = sum(
            1 for result in results.values()
            if result.risk_level == RiskLevel.HIGH and not result.passed
        )
        if high_risk_count >= 2:
            return False
        
        return True


if __name__ == "__main__":
    # 테스트 코드
    logger = get_logger(__name__)
    
    logger.info("=== 리스크 체크 테스트 ===")
    checker = IntegratedRiskChecker(trading_type='futures')
    
    # 테스트 데이터
    test_data = {
        'data_collection': {
            'price': 50000.0,
            'connection_status': 'connected',
            'data_delay_ms': 100,
            'orderbook': {'bids': [[50000, 1.0]], 'asks': [[50001, 1.0]]}
        },
        'preprocessing': {
            'missing_data_count': 2,
            'total_data_count': 100,
            'outlier_count': 1,
            'data_consistency': True
        },
        'indicators': {
            'RSI': 50.0,
            'MACD': 0.5,
            'MA': 50000.0
        },
        'pattern_recognition': {
            'patterns': {'PATTERN_BULLISH_ENGULFING': 1},
            'pattern_confidence': 0.7
        },
        'signal_generation': {
            'primary_signal': 'BUY',
            'signal_strength': 1.5,
            'confidence': 75,
            'market_condition': 'normal',
            'leverage': 5.0,
            'margin_ratio': 30.0
        }
    }
    
    # 전체 체크
    results = checker.check_all(test_data)
    
    for stage, result in results.items():
        logger.info(f"{stage}: {result}")
    
    # 진행 가능 여부
    can_proceed = checker.should_proceed(results)
    logger.info(f"진행 가능: {can_proceed}")

