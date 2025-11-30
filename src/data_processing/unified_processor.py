# src/data_processing/unified_processor.py

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
import asyncio
from collections import deque
from src.utils.logger import get_logger
from src.data_processing.indicators import TechnicalIndicators
from src.data_processing.pattern_recognition import PatternRecognition
from src.data_processing.features import TradingSignalProcessor
from src.data_processing.preprocessor import DataPreprocessor
from src.risk_management.risk_checker import IntegratedRiskChecker, RiskCheckResult, RiskLevel
from src.config.settings import TRADING_TYPE


class DataProcessor(ABC):
    """백테스팅과 실시간 처리를 위한 통합 데이터 처리 인터페이스"""

    @abstractmethod
    def process_data(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """데이터 처리 메인 메서드"""
        pass

    @abstractmethod
    def generate_signals(self, processed_data: pd.DataFrame) -> Dict:
        """처리된 데이터에서 거래 신호 생성"""
        pass


class UnifiedDataProcessor(DataProcessor):
    """백테스팅과 실시간 처리를 통합한 데이터 프로세서"""

    def __init__(self,
                 buffer_size: int = 1000,
                 min_data_points: int = 200,
                 enable_ml_features: bool = False):
        """
        Args:
            buffer_size: 실시간 데이터 버퍼 크기
            min_data_points: 지표 계산을 위한 최소 데이터 포인트
            enable_ml_features: ML 특징 생성 여부
        """
        self.logger = get_logger(__name__)
        self.buffer_size = buffer_size
        self.min_data_points = min_data_points
        self.enable_ml_features = enable_ml_features

        # 지표 및 패턴 인식 모듈
        self.indicators = TechnicalIndicators()
        self.pattern_recognition = PatternRecognition()
        
        # 데이터 전처리 모듈
        self.preprocessor = DataPreprocessor(
            outlier_std_threshold=3.0,
            missing_data_threshold=0.1,
            enable_normalization=False
        )

        # TradingSignalProcessor 초기화 - pattern_recognizer 인자 전달
        self.signal_processor = TradingSignalProcessor(self.pattern_recognition)

        # 리스크 체커 초기화
        self.risk_checker = IntegratedRiskChecker(trading_type=TRADING_TYPE)
        
        # 리스크 체크 결과 저장 (모니터링용)
        self.risk_check_results: Dict[str, List[RiskCheckResult]] = {
            'data_collection': [],
            'preprocessing': [],
            'indicators': [],
            'pattern_recognition': [],
            'signal_generation': []
        }

        # 실시간 데이터 버퍼
        self.data_buffer = deque(maxlen=buffer_size)
        self.processed_buffer = deque(maxlen=buffer_size)

        # 처리 상태
        self.last_processed_time = None
        self.processing_mode = None  # 'backtest' or 'realtime'

        # 백테스팅 결과를 저장할 피드백 데이터
        self.feedback_data = {
            'signal_accuracy': {},
            'pattern_success_rate': {},
            'indicator_thresholds': {}
        }

    def process_data(self, data: Union[pd.DataFrame, Dict], mode: str = 'auto') -> pd.DataFrame:
        """데이터 처리 메인 인터페이스

        Args:
            data: 입력 데이터 (DataFrame 또는 Dict)
            mode: 처리 모드 ('backtest', 'realtime', 'auto')

        Returns:
            처리된 DataFrame
        """
        try:
            # Dict 입력 처리 - 실시간 데이터
            if isinstance(data, dict):
                # 실시간 데이터는 스트리밍 처리 메서드 사용
                processed = self._process_streaming_data(data)
                if processed is None:
                    # 버퍼가 충분하지 않으면 빈 DataFrame 반환
                    return pd.DataFrame()
                # 처리된 데이터를 DataFrame으로 변환
                return pd.DataFrame([processed])

            # DataFrame 입력 처리
            elif isinstance(data, pd.DataFrame):
                if data.empty:
                    self.logger.warning("빈 데이터프레임이 입력되었습니다.")
                    return pd.DataFrame()

                # 모드 자동 결정
                if mode == 'auto':
                    mode = 'backtest' if len(data) > 100 else 'realtime'

                self.processing_mode = mode

                if mode == 'backtest':
                    return self._process_batch_data(data)
                else:
                    # 실시간 모드에서도 배치 처리 사용
                    return self._process_batch_data(data)

            else:
                raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")

        except Exception as e:
            self.logger.error(f"데이터 처리 중 오류 발생: {e}")
            raise

    def _process_batch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """백테스팅용 배치 데이터 처리

        Args:
            df: OHLCV DataFrame

        Returns:
            처리된 DataFrame
        """
        self.logger.info(f"배치 데이터 처리 시작: {len(df)} 행")

        # 1. 데이터 수집 단계 리스크 체크
        if len(df) > 0:
            data_collection_risk_data = self._prepare_risk_check_data(df, 'data_collection')
            data_collection_result = self.risk_checker.check_stage('data_collection', data_collection_risk_data)
            if not self._handle_risk_check_result('data_collection', data_collection_result):
                self.logger.error("데이터 수집 단계 리스크 체크 실패로 처리 중단")
                return pd.DataFrame()

        # 2. 기본 전처리
        df = self._preprocess_data(df)
        
        if df.empty:
            self.logger.warning("전처리 후 데이터가 비어있습니다.")
            return pd.DataFrame()

        # 3. 전처리 단계 리스크 체크
        preprocessing_risk_data = self._prepare_risk_check_data(df, 'preprocessing')
        preprocessing_result = self.risk_checker.check_stage('preprocessing', preprocessing_risk_data)
        if not self._handle_risk_check_result('preprocessing', preprocessing_result):
            self.logger.error("전처리 단계 리스크 체크 실패로 처리 중단")
            return pd.DataFrame()

        # 4. TradingSignalProcessor를 사용한 전체 처리
        # generate_indicators=False로 설정하여 중복 지표 계산 방지
        df = self.signal_processor.process_data(df, symbol="BTCUSDT", generate_indicators=False)

        # 5. 기술적 지표 계산 (signal_processor에서 안했다면)
        if 'MA_20' not in df.columns:
            df = self.indicators.add_all_indicators(df)

        # 6. 기술지표 계산 단계 리스크 체크
        if len(df) > 0:
            indicators_risk_data = self._prepare_risk_check_data(df, 'indicators')
            indicators_result = self.risk_checker.check_stage('indicators', indicators_risk_data)
            if not self._handle_risk_check_result('indicators', indicators_result):
                self.logger.error("기술지표 계산 단계 리스크 체크 실패로 처리 중단")
                return pd.DataFrame()

        # 7. 패턴 인식 (signal_processor에서 안했다면)
        if 'PATTERN_BULLISH_ENGULFING' not in df.columns:
            df = self.pattern_recognition.detect_all_patterns(df)
            df = self.pattern_recognition.find_chart_patterns(df)
            df = self.pattern_recognition.detect_advanced_patterns(df)

        # 8. 패턴 인식 단계 리스크 체크
        if len(df) > 0:
            pattern_risk_data = self._prepare_risk_check_data(df, 'pattern_recognition')
            pattern_result = self.risk_checker.check_stage('pattern_recognition', pattern_risk_data)
            if not self._handle_risk_check_result('pattern_recognition', pattern_result):
                self.logger.warning("패턴 인식 단계 리스크 체크 실패 (경고 후 계속)")

        # 9. ML 특징 생성 (선택사항)
        if self.enable_ml_features:
            df = self._generate_ml_features(df)

        # 10. 피드백 데이터 적용
        df = self._apply_feedback(df)

        # 11. 신호 생성 단계 리스크 체크 (신호 생성 후)
        if len(df) > 0:
            # 신호 정보 추출
            signals = self.generate_signals(df)
            signal_risk_data = self._prepare_risk_check_data(df, 'signal_generation', additional_data=signals)
            signal_result = self.risk_checker.check_stage('signal_generation', signal_risk_data)
            if not self._handle_risk_check_result('signal_generation', signal_result):
                self.logger.warning("신호 생성 단계 리스크 체크 실패 (경고 후 계속)")

        self.logger.info("배치 데이터 처리 완료")
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리

        Args:
            df: 원본 DataFrame

        Returns:
            전처리된 DataFrame
        """
        # DataPreprocessor 사용
        processed_df, stats = self.preprocessor.preprocess(df, validate=True)
        
        # 전처리 통계 로깅
        if stats:
            self.logger.debug(f"전처리 통계: {stats}")
        
        return processed_df
    
    def _prepare_risk_check_data(self, df: pd.DataFrame, stage: str, 
                                  additional_data: Optional[Dict] = None) -> Dict:
        """각 단계별 리스크 체크 데이터 준비
        
        Args:
            df: 현재 단계의 DataFrame
            stage: 단계명 ('data_collection', 'preprocessing', 'indicators', 
                          'pattern_recognition', 'signal_generation')
            additional_data: 추가 데이터 (신호 정보 등)
            
        Returns:
            리스크 체크용 데이터 딕셔너리
        """
        risk_data = {}
        
        if stage == 'data_collection':
            # 데이터 수집 단계 체크
            latest_price = df['close'].iloc[-1] if 'close' in df.columns and len(df) > 0 else None
            risk_data = {
                'price': float(latest_price) if latest_price is not None else None,
                'connection_status': 'connected',  # 실제로는 연결 상태 확인 필요
                'data_delay_ms': 0,  # 실제로는 계산 필요
                'orderbook': {}  # 실제로는 수집 필요
            }
            
        elif stage == 'preprocessing':
            # 전처리 단계 체크
            stats = self.preprocessor.get_preprocessing_stats()
            missing_data = stats.get('missing_data', {})
            outlier_data = stats.get('outliers', {})
            consistency_data = stats.get('data_consistency', {})
            
            risk_data = {
                'missing_data_count': missing_data.get('missing_before', 0),
                'total_data_count': len(df),
                'outlier_count': outlier_data.get('removed_count', 0),
                'data_consistency': consistency_data.get('is_consistent', True)
            }
            
        elif stage == 'indicators':
            # 기술지표 계산 단계 체크
            indicator_cols = [col for col in df.columns 
                             if col in ['RSI_14', 'MACD', 'MA_20', 'MA_50', 
                                       'BB_upper_20', 'BB_lower_20']]
            indicators = {}
            for col in indicator_cols:
                if col in df.columns and len(df) > 0:
                    val = df[col].iloc[-1]
                    if pd.notna(val):
                        # 지표 이름 정규화 (RSI_14 -> RSI, MA_20 -> MA 등)
                        if 'RSI' in col:
                            indicators['RSI'] = float(val)
                        elif 'MACD' in col:
                            indicators['MACD'] = float(val)
                        elif 'MA' in col:
                            indicators['MA'] = float(val)
            
            # 지표 신뢰도 계산 (NaN 비율 기반)
            indicator_reliability = 1.0
            if indicator_cols:
                nan_ratio = df[indicator_cols].isna().sum().sum() / (len(df) * len(indicator_cols))
                indicator_reliability = 1.0 - min(nan_ratio, 1.0)
            
            risk_data = {
                'indicators': indicators,
                'indicator_reliability': indicator_reliability
            }
            
        elif stage == 'pattern_recognition':
            # 패턴 인식 단계 체크
            pattern_cols = [col for col in df.columns if col.startswith('PATTERN_')]
            patterns = {}
            for col in pattern_cols:
                if col in df.columns and len(df) > 0:
                    val = df[col].iloc[-1]
                    if pd.notna(val) and val != 0:
                        patterns[col] = float(val)
            
            # 패턴 신뢰도 계산
            pattern_confidence = 0.0
            if patterns:
                # 활성 패턴 수 기반 신뢰도
                active_patterns = sum(1 for v in patterns.values() if v != 0)
                pattern_confidence = min(active_patterns / 5.0, 1.0)  # 최대 5개 패턴 기준
            
            risk_data = {
                'patterns': patterns,
                'pattern_confidence': pattern_confidence
            }
            
        elif stage == 'signal_generation':
            # 신호 생성 단계 체크
            if additional_data and isinstance(additional_data, dict):
                signals = additional_data
            else:
                # DataFrame에서 신호 정보 추출
                signals = {}
                if 'BUY_RECOMMENDATION' in df.columns and len(df) > 0:
                    signals['primary_signal'] = 'BUY' if df['BUY_RECOMMENDATION'].iloc[-1] > 0 else 'HOLD'
                elif 'SELL_RECOMMENDATION' in df.columns and len(df) > 0:
                    signals['primary_signal'] = 'SELL' if df['SELL_RECOMMENDATION'].iloc[-1] > 0 else 'HOLD'
                else:
                    signals['primary_signal'] = 'HOLD'
                
                if 'SIGNAL_CONFIDENCE' in df.columns and len(df) > 0:
                    signals['confidence'] = float(df['SIGNAL_CONFIDENCE'].iloc[-1])
                else:
                    signals['confidence'] = 50.0
                
                if 'COMBINED_SIGNAL' in df.columns and len(df) > 0:
                    signals['signal_strength'] = abs(float(df['COMBINED_SIGNAL'].iloc[-1]))
                else:
                    signals['signal_strength'] = 0.0
            
            # 시장 상태 추정 (변동성 기반)
            market_condition = 'normal'
            if 'ATR_percent_14' in df.columns and len(df) > 0:
                atr_pct = df['ATR_percent_14'].iloc[-1]
                if pd.notna(atr_pct):
                    if atr_pct > 5.0:
                        market_condition = 'extreme'
                    elif atr_pct > 3.0:
                        market_condition = 'volatile'
            
            risk_data = {
                'primary_signal': signals.get('primary_signal', 'HOLD'),
                'signal_strength': signals.get('signal_strength', 0.0),
                'confidence': signals.get('confidence', 0),
                'market_condition': market_condition
            }
            
            # Futures 특화 정보 추가
            if TRADING_TYPE == 'futures':
                risk_data['leverage'] = 1.0  # 실제로는 설정에서 가져와야 함
                risk_data['margin_ratio'] = 0.0  # 실제로는 계산 필요
        
        return risk_data
    
    def _handle_risk_check_result(self, stage: str, result: RiskCheckResult) -> bool:
        """리스크 체크 결과 처리 및 다음 단계 진행 여부 결정
        
        Args:
            stage: 단계명
            result: 리스크 체크 결과
            
        Returns:
            다음 단계 진행 가능 여부
        """
        # 결과 저장 (모니터링용)
        if stage in self.risk_check_results:
            self.risk_check_results[stage].append(result)
            # 최근 100개만 유지
            if len(self.risk_check_results[stage]) > 100:
                self.risk_check_results[stage] = self.risk_check_results[stage][-100:]
        
        # 로깅
        if not result.passed:
            if result.risk_level == RiskLevel.CRITICAL:
                self.logger.error(f"[{stage}] 리스크 체크 실패 (CRITICAL): {result.message}")
            elif result.risk_level == RiskLevel.HIGH:
                self.logger.warning(f"[{stage}] 리스크 체크 실패 (HIGH): {result.message}")
            else:
                self.logger.warning(f"[{stage}] 리스크 체크 실패: {result.message}")
        else:
            self.logger.debug(f"[{stage}] 리스크 체크 통과: {result.message}")
        
        # 진행 가능 여부 결정
        # CRITICAL 리스크는 항상 중단
        if result.risk_level == RiskLevel.CRITICAL and not result.passed:
            return False
        
        # HIGH 리스크는 경고 후 계속 (선택적)
        # 여기서는 HIGH 리스크도 계속 진행하도록 설정
        # 필요시 False로 변경하여 중단 가능
        
        return True

    def _update_buffer(self, data: Dict):
        """실시간 데이터 버퍼 업데이트

        Args:
            data: 새로운 데이터 포인트
        """
        # 타임스탬프 확인 및 추가
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()

        self.data_buffer.append(data)
        self.last_processed_time = data['timestamp']

    def _remove_outliers(self, df: pd.DataFrame, std_threshold: float = 3.0) -> pd.DataFrame:
        """이상치 제거

        Args:
            df: 입력 DataFrame
            std_threshold: 표준편차 임계값

        Returns:
            이상치가 제거된 DataFrame
        """
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']

        for col in numeric_columns:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - std_threshold * std
                upper_bound = mean + std_threshold * std

                # 이상치를 경계값으로 대체
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound

        return df

    def _generate_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """머신러닝용 추가 특징 생성

        Args:
            df: 기본 지표가 계산된 DataFrame

        Returns:
            ML 특징이 추가된 DataFrame
        """
        # 가격 변화율 특징
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)

        # 기술적 지표 간 상호작용 특징
        if 'RSI_14' in df.columns and 'MA_20' in df.columns:
            df['rsi_ma_ratio'] = df['RSI_14'] / 50  # RSI normalized
            df['price_ma_ratio'] = df['close'] / df['MA_20']

        # 시간 기반 특징
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        # 변동성 특징
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        return df

    def _apply_feedback(self, df: pd.DataFrame) -> pd.DataFrame:
        """백테스팅 피드백을 실시간 처리에 적용

        Args:
            df: 처리된 DataFrame

        Returns:
            피드백이 적용된 DataFrame
        """
        # 신호 정확도 기반 필터링
        if 'signal_accuracy' in self.feedback_data:
            for signal_type, accuracy in self.feedback_data['signal_accuracy'].items():
                if accuracy < 0.5 and signal_type in df.columns:
                    # 정확도가 낮은 신호는 신뢰도를 낮춤
                    df[f'{signal_type}_confidence'] = df[signal_type] * accuracy

        # 패턴 성공률 기반 가중치 적용
        if 'pattern_success_rate' in self.feedback_data:
            for pattern, success_rate in self.feedback_data['pattern_success_rate'].items():
                if pattern in df.columns:
                    df[f'{pattern}_weighted'] = df[pattern] * success_rate

        # 동적 임계값 적용
        if 'indicator_thresholds' in self.feedback_data:
            for indicator, thresholds in self.feedback_data['indicator_thresholds'].items():
                if indicator in df.columns:
                    # 동적 임계값을 사용한 신호 재생성
                    if 'upper' in thresholds:
                        df[f'{indicator}_signal_upper'] = (df[indicator] > thresholds['upper']).astype(int)
                    if 'lower' in thresholds:
                        df[f'{indicator}_signal_lower'] = (df[indicator] < thresholds['lower']).astype(int)

        return df

    def generate_signals(self, processed_data: pd.DataFrame) -> Dict:
        """처리된 데이터에서 신호 생성

        Args:
            processed_data: 처리된 DataFrame

        Returns:
            신호 딕셔너리
        """
        try:
            latest_data = processed_data.iloc[-1]

            # TradingSignalProcessor의 get_signal_stats 사용
            stats = self.signal_processor.get_signal_stats(processed_data)

            signals = {
                'timestamp': latest_data.name if hasattr(latest_data, 'name') else pd.Timestamp.now(),
                'primary_signal': 'HOLD',
                'signal_strength': 0,
                'confidence': 50,
                'indicators': {},
                'patterns': {},
                'stats': stats
            }

            # 신호 생성 로직 개선
            # 1. BUY_RECOMMENDATION/SELL_RECOMMENDATION 확인
            buy_rec = latest_data.get('BUY_RECOMMENDATION', 0) if 'BUY_RECOMMENDATION' in processed_data.columns else 0
            sell_rec = latest_data.get('SELL_RECOMMENDATION', 0) if 'SELL_RECOMMENDATION' in processed_data.columns else 0
            
            # 2. COMBINED_SIGNAL 확인 (없으면 다른 신호 컬럼 확인)
            combined_signal = latest_data.get('COMBINED_SIGNAL', 0)
            if combined_signal == 0:
                # 대체 신호 컬럼 확인
                if 'CONFIRMED_BUY_SIGNAL' in processed_data.columns:
                    combined_signal = latest_data.get('CONFIRMED_BUY_SIGNAL', 0) * 2.0
                elif 'CONFIRMED_SELL_SIGNAL' in processed_data.columns:
                    combined_signal = latest_data.get('CONFIRMED_SELL_SIGNAL', 0) * -2.0
                elif 'PATTERN_BUY_SIGNAL' in processed_data.columns:
                    combined_signal = latest_data.get('PATTERN_BUY_SIGNAL', 0) * 1.5
                elif 'PATTERN_SELL_SIGNAL' in processed_data.columns:
                    combined_signal = latest_data.get('PATTERN_SELL_SIGNAL', 0) * -1.5
            
            # 3. 신뢰도 계산
            confidence = latest_data.get('SIGNAL_CONFIDENCE', 50)
            if confidence == 50:  # 기본값이면 재계산 시도
                # 패턴 기반 신뢰도
                if 'BULLISH_PATTERNS' in processed_data.columns:
                    bullish_count = latest_data.get('BULLISH_PATTERNS', 0)
                    bearish_count = latest_data.get('BEARISH_PATTERNS', 0)
                    if bullish_count > 0 or bearish_count > 0:
                        confidence = 50 + min((bullish_count + bearish_count) * 10, 30)
                # 신호 강도 기반 신뢰도
                if abs(combined_signal) > 0:
                    confidence = max(confidence, 50 + min(abs(combined_signal) * 10, 30))
            
            # 4. 신호 결정
            if buy_rec == 1 or (combined_signal > 0 and confidence >= 50):
                signals['primary_signal'] = 'BUY'
                signals['signal_strength'] = abs(combined_signal) if combined_signal > 0 else 1.0
                signals['confidence'] = confidence
            elif sell_rec == 1 or (combined_signal < 0 and confidence >= 50):
                signals['primary_signal'] = 'SELL'
                signals['signal_strength'] = abs(combined_signal) if combined_signal < 0 else 1.0
                signals['confidence'] = confidence
            else:
                # HOLD이지만 신호 강도와 신뢰도는 설정
                signals['signal_strength'] = abs(combined_signal)
                signals['confidence'] = confidence

            # 디버깅: 신호 생성 정보 로깅
            self.logger.debug(
                f"신호 생성 디버깅: "
                f"BUY_REC={buy_rec}, SELL_REC={sell_rec}, "
                f"COMBINED_SIGNAL={combined_signal:.2f}, "
                f"CONFIDENCE={confidence:.1f}%, "
                f"최종신호={signals['primary_signal']}"
            )
            
            # 지표 정보
            for col in processed_data.columns:
                if col in ['RSI_14', 'MACD', 'MA_20', 'MA_50', 'BB_upper_20', 'BB_lower_20', 'close']:
                    signals['indicators'][col] = float(latest_data.get(col, 0))

            # 패턴 정보
            pattern_cols = [col for col in processed_data.columns if col.startswith('PATTERN_')]
            for col in pattern_cols:
                if latest_data.get(col, 0) != 0:
                    signals['patterns'][col] = latest_data[col]

            return signals

        except Exception as e:
            self.logger.error(f"신호 생성 중 오류 발생: {e}")
            return {
                'timestamp': pd.Timestamp.now(),
                'primary_signal': 'ERROR',
                'signal_strength': 0,
                'confidence': 0,
                'indicators': {},
                'patterns': {},
                'error': str(e)
            }

    def _get_primary_signal(self, data: pd.Series) -> str:
        """주요 신호 추출"""
        if data.get('CONFIRMED_BUY_SIGNAL', 0) > 0:
            return 'BUY'
        elif data.get('CONFIRMED_SELL_SIGNAL', 0) > 0:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_confidence(self, data: pd.Series) -> float:
        """신호 신뢰도 계산"""
        confidence_factors = []

        # 패턴 강도
        if 'PATTERN_STRENGTH' in data:
            confidence_factors.append(min(abs(data['PATTERN_STRENGTH']) / 100, 1.0))

        # 지표 일치도
        bullish_indicators = 0
        bearish_indicators = 0

        # RSI
        if 'RSI_14' in data:
            if data['RSI_14'] < 30:
                bullish_indicators += 1
            elif data['RSI_14'] > 70:
                bearish_indicators += 1

        # MACD
        if 'MACD_hist' in data and data['MACD_hist'] != 0:
            if data['MACD_hist'] > 0:
                bullish_indicators += 1
            else:
                bearish_indicators += 1

        total_indicators = bullish_indicators + bearish_indicators
        if total_indicators > 0:
            alignment = max(bullish_indicators, bearish_indicators) / total_indicators
            confidence_factors.append(alignment)

        # 평균 신뢰도 계산
        return np.mean(confidence_factors) if confidence_factors else 0.5

    def _assess_risk(self, data: pd.Series) -> str:
        """리스크 수준 평가"""
        risk_score = 0

        # 변동성 기반 리스크
        if 'ATR_percent_14' in data:
            if data['ATR_percent_14'] > 5:
                risk_score += 2
            elif data['ATR_percent_14'] > 3:
                risk_score += 1

        # 볼린저 밴드 위치
        if 'close' in data and 'BB_upper_20' in data and 'BB_lower_20' in data:
            bb_position = (data['close'] - data['BB_lower_20']) / (data['BB_upper_20'] - data['BB_lower_20'])
            if bb_position > 0.9 or bb_position < 0.1:
                risk_score += 1

        # 리스크 레벨 결정
        if risk_score >= 3:
            return 'HIGH'
        elif risk_score >= 1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _extract_key_indicators(self, data: pd.Series) -> Dict:
        """주요 지표 값 추출"""
        indicators = {}
        key_indicators = ['RSI_14', 'MACD', 'ATR_14', 'close', 'volume']

        for indicator in key_indicators:
            if indicator in data:
                indicators[indicator] = float(data[indicator])

        return indicators

    def _extract_active_patterns(self, data: pd.Series) -> List[str]:
        """활성 패턴 추출"""
        patterns = []
        pattern_columns = [col for col in data.index if col.startswith('PATTERN_') and data[col] != 0]

        for col in pattern_columns:
            if data[col] != 0:
                patterns.append(col)

        return patterns

    def _generate_recommendation(self, data: pd.Series) -> str:
        """거래 추천 생성"""
        signal = self._get_primary_signal(data)
        confidence = self._calculate_confidence(data)
        risk = self._assess_risk(data)

        if signal == 'BUY':
            if confidence > 0.7 and risk != 'HIGH':
                return "강력 매수 추천"
            elif confidence > 0.5:
                return "매수 고려"
            else:
                return "매수 신호이나 확신도 낮음"
        elif signal == 'SELL':
            if confidence > 0.7 and risk != 'HIGH':
                return "강력 매도 추천"
            elif confidence > 0.5:
                return "매도 고려"
            else:
                return "매도 신호이나 확신도 낮음"
        else:
            return "관망 추천"

    def update_feedback(self, feedback_data: Dict):
        """백테스팅 결과를 피드백 데이터로 업데이트

        Args:
            feedback_data: 백테스팅에서 얻은 피드백 데이터
        """
        self.feedback_data.update(feedback_data)
        self.logger.info("피드백 데이터 업데이트 완료")

    async def process_realtime_stream(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        """실시간 데이터 스트림 처리 (비동기)

        Args:
            input_queue: 입력 데이터 큐
            output_queue: 출력 신호 큐
        """
        self.logger.info("실시간 스트림 처리 시작")
        data_count = 0

        while True:
            try:
                # 입력 큐에서 데이터 가져오기 (타임아웃 없이 대기)
                self.logger.debug("큐에서 데이터 대기 중...")
                data = await input_queue.get()
                data_count += 1
                self.logger.info(f"큐에서 데이터 수신 (#{data_count}): {type(data)}")

                # None이면 종료 신호
                if data is None:
                    self.logger.info("종료 신호 수신")
                    break

                # 데이터 처리
                self.logger.debug(f"데이터 처리 시작: {data.get('close', 'N/A') if isinstance(data, dict) else 'N/A'}")
                processed = self._process_streaming_data(data)

                # 처리된 데이터가 있으면 출력 큐에 추가
                if processed:
                    if 'signals' in processed:
                        signals = processed['signals']
                        # 신호 딕셔너리에 전체 processed 데이터도 포함
                        if isinstance(signals, dict):
                            signals['processed_data'] = processed
                        await output_queue.put(signals)
                        self.logger.debug(f"신호 큐에 추가: {signals.get('primary_signal', 'HOLD')}")
                    else:
                        self.logger.warning("처리된 데이터에 신호가 없습니다.")
                else:
                    self.logger.debug("데이터 처리 결과가 None입니다.")

            except Exception as e:
                self.logger.error(f"스트림 처리 중 오류: {e}")
                continue

        self.logger.info("실시간 스트림 처리 종료")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표를 기반으로 추가 특징 생성

        Args:
            df: 기술적 지표가 계산된 DataFrame

        Returns:
            특징이 추가된 DataFrame
        """
        result = df.copy()

        try:
            # 가격 기반 특징
            result['price_range'] = result['high'] - result['low']
            result['price_change'] = result['close'] - result['open']
            result['price_change_pct'] = (result['close'] - result['open']) / result['open'] * 100

            # 이동평균 기반 특징
            if 'MA_20' in result.columns and 'MA_50' in result.columns:
                result['ma_cross'] = result['MA_20'] - result['MA_50']
                result['price_to_ma20'] = result['close'] / result['MA_20']
                result['price_to_ma50'] = result['close'] / result['MA_50']

            # RSI 기반 특징
            if 'RSI_14' in result.columns:
                result['rsi_overbought'] = (result['RSI_14'] > 70).astype(int)
                result['rsi_oversold'] = (result['RSI_14'] < 30).astype(int)
                result['rsi_neutral'] = ((result['RSI_14'] >= 30) & (result['RSI_14'] <= 70)).astype(int)

            # MACD 기반 특징
            if 'MACD' in result.columns and 'MACD_signal' in result.columns:
                result['macd_divergence'] = result['MACD'] - result['MACD_signal']
                result['macd_cross_up'] = ((result['MACD'] > result['MACD_signal']) &
                                           (result['MACD'].shift(1) <= result['MACD_signal'].shift(1))).astype(int)
                result['macd_cross_down'] = ((result['MACD'] < result['MACD_signal']) &
                                             (result['MACD'].shift(1) >= result['MACD_signal'].shift(1))).astype(int)

            # 볼린저 밴드 기반 특징
            if 'BB_upper_20' in result.columns and 'BB_lower_20' in result.columns:
                bb_width = result['BB_upper_20'] - result['BB_lower_20']
                bb_middle = (result['BB_upper_20'] + result['BB_lower_20']) / 2
                result['bb_position'] = (result['close'] - result['BB_lower_20']) / bb_width
                result['bb_squeeze'] = bb_width / bb_middle  # 낮을수록 squeeze

            # 거래량 기반 특징
            if 'volume_MA_20' in result.columns:
                result['volume_ratio'] = result['volume'] / result['volume_MA_20']
                result['high_volume'] = (result['volume'] > result['volume_MA_20'] * 1.5).astype(int)

            # 변동성 특징
            if 'ATR_14' in result.columns:
                result['volatility_normalized'] = result['ATR_14'] / result['close']
                result['high_volatility'] = (result['ATR_percent_14'] > 2).astype(int)

            self.logger.info("추가 특징 생성 완료")

        except Exception as e:
            self.logger.error(f"특징 생성 중 오류: {e}")

        return result

    def _process_streaming_data(self, data: Dict) -> Optional[Dict]:
        """실시간 스트리밍 데이터 처리

        Args:
            data: 실시간 데이터 딕셔너리

        Returns:
            처리된 데이터 또는 None
        """
        try:
            # 실시간 데이터를 DataFrame으로 변환
            df_row = pd.DataFrame([data])
            if 'timestamp' in df_row.columns:
                df_row['timestamp'] = pd.to_datetime(df_row['timestamp'], unit='ms')

            # 버퍼에 추가
            self.data_buffer.append(data)
            buffer_size = len(self.data_buffer)
            
            self.logger.info(f"버퍼에 데이터 추가됨: 현재 크기 {buffer_size}/{self.min_data_points}")

            # 최소 데이터 포인트 확인
            if buffer_size < self.min_data_points:
                if buffer_size % 10 == 0 or buffer_size <= 5:  # 10개마다 또는 처음 5개는 항상 로그
                    self.logger.info(f"버퍼 수집 중: {buffer_size}/{self.min_data_points} (신호 생성 대기 중...)")
                return None
            
            self.logger.info(f"[OK] 최소 데이터 포인트 충족: {buffer_size}/{self.min_data_points} - 신호 생성 시작")

            # 버퍼를 DataFrame으로 변환
            df = pd.DataFrame(list(self.data_buffer))
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

            # 데이터 수집 단계 리스크 체크 (실시간 데이터)
            if len(df) > 0:
                data_collection_risk_data = self._prepare_risk_check_data(df, 'data_collection')
                # 실시간 데이터의 경우 연결 상태와 지연 시간을 실제로 확인해야 함
                # 여기서는 기본값 사용
                data_collection_result = self.risk_checker.check_stage('data_collection', data_collection_risk_data)
                if not self._handle_risk_check_result('data_collection', data_collection_result):
                    self.logger.error("실시간 데이터 수집 단계 리스크 체크 실패로 처리 중단")
                    return None

            # 배치 처리 로직 재사용 (내부에서 리스크 체크 수행)
            processed_df = self._process_batch_data(df)
            
            if processed_df.empty:
                self.logger.warning("처리된 데이터가 비어있습니다.")
                return None

            # 최신 데이터만 반환
            latest_processed = processed_df.iloc[-1].to_dict()

            # 처리된 데이터를 버퍼에 저장
            self.processed_buffer.append(latest_processed)

            # 신호 생성
            signals = self.generate_signals(processed_df)
            
            # 신호 생성 단계 리스크 체크 (실시간 처리)
            if signals:
                signal_risk_data = self._prepare_risk_check_data(processed_df, 'signal_generation', additional_data=signals)
                signal_result = self.risk_checker.check_stage('signal_generation', signal_risk_data)
                if not self._handle_risk_check_result('signal_generation', signal_result):
                    self.logger.warning("실시간 신호 생성 단계 리스크 체크 실패 (경고 후 계속)")
            
            # 신호 로깅
            if signals:
                primary_signal = signals.get('primary_signal', 'HOLD')
                confidence = signals.get('confidence', 0)
                signal_strength = signals.get('signal_strength', 0)
                self.logger.info(
                    f"신호 생성: {primary_signal} | "
                    f"신뢰도: {confidence:.1f}% | "
                    f"강도: {signal_strength:.2f}"
                )
            else:
                self.logger.debug("신호 생성 실패 또는 HOLD")

            # 실시간 데이터에 신호 추가
            latest_processed['signals'] = signals

            self.logger.debug("실시간 데이터 처리 완료")
            return latest_processed

        except Exception as e:
            self.logger.error(f"실시간 데이터 처리 중 오류: {e}")
            return None

    def process_realtime_tick(self, tick_data: Dict) -> Optional[Dict]:
        """실시간 틱 데이터 처리

        Args:
            tick_data: 틱 데이터

        Returns:
            처리된 결과 또는 None
        """
        # _process_streaming_data와 동일한 로직
        return self._process_streaming_data(tick_data)

    def get_buffer_status(self) -> Dict:
        """버퍼 상태 조회

        Returns:
            버퍼 상태 정보
        """
        return {
            'data_buffer_size': len(self.data_buffer),
            'processed_buffer_size': len(self.processed_buffer),
            'min_data_points': self.min_data_points,
            'buffer_full': len(self.data_buffer) >= self.min_data_points,
            'last_processed_time': self.last_processed_time
        }

    def clear_buffers(self):
        """버퍼 초기화"""
        self.data_buffer.clear()
        self.processed_buffer.clear()
        self.last_processed_time = None
        self.logger.info("버퍼가 초기화되었습니다")
    
    def get_risk_check_statistics(self) -> Dict:
        """리스크 체크 통계 조회
        
        Returns:
            리스크 체크 통계 딕셔너리
        """
        stats = {}
        for stage, results in self.risk_check_results.items():
            if results:
                total = len(results)
                passed = sum(1 for r in results if r.passed)
                failed = total - passed
                
                # 리스크 레벨별 통계
                critical = sum(1 for r in results if r.risk_level == RiskLevel.CRITICAL and not r.passed)
                high = sum(1 for r in results if r.risk_level == RiskLevel.HIGH and not r.passed)
                medium = sum(1 for r in results if r.risk_level == RiskLevel.MEDIUM and not r.passed)
                low = sum(1 for r in results if r.risk_level == RiskLevel.LOW)
                
                stats[stage] = {
                    'total_checks': total,
                    'passed': passed,
                    'failed': failed,
                    'pass_rate': passed / total if total > 0 else 0.0,
                    'risk_levels': {
                        'critical': critical,
                        'high': high,
                        'medium': medium,
                        'low': low
                    },
                    'latest_result': str(results[-1]) if results else None
                }
            else:
                stats[stage] = {
                    'total_checks': 0,
                    'passed': 0,
                    'failed': 0,
                    'pass_rate': 0.0,
                    'risk_levels': {
                        'critical': 0,
                        'high': 0,
                        'medium': 0,
                        'low': 0
                    },
                    'latest_result': None
                }
        
        return stats


# 백테스팅과 실시간 처리를 연결하는 어댑터
class BacktestRealtimeAdapter:
    """백테스팅 결과를 실시간 처리에 활용하는 어댑터"""

    def __init__(self, processor: UnifiedDataProcessor):
        self.processor = processor
        self.logger = get_logger(__name__)

    def extract_feedback_from_backtest(self, backtest_results: Dict) -> Dict:
        """백테스팅 결과에서 피드백 데이터 추출

        Args:
            backtest_results: 백테스팅 결과

        Returns:
            피드백 데이터
        """
        global trades_df
        feedback = {
            'signal_accuracy': {},
            'pattern_success_rate': {},
            'indicator_thresholds': {}
        }

        try:
            # BacktestEngine의 실제 결과 구조에 맞게 수정
            if 'trades' in backtest_results:
                trades_data = backtest_results['trades']

                if isinstance(trades_data, pd.DataFrame) and not trades_data.empty:
                    trades_df = trades_data
                elif isinstance(trades_data, list) and len(trades_data) > 0:
                    # list 처리 로직
                    trades_df = pd.DataFrame(trades_data)

                # 거래 데이터를 전처리

                # 거래 신호 분석
                if not trades_df.empty and 'signal' in trades_df.columns:
                    # 매수 신호 정확도
                    buy_signals = trades_df[trades_df['signal'] > 0]
                    if len(buy_signals) > 0:
                        # 다음 거래에서 수익이 발생했는지 확인
                        profitable_buys = 0
                        for idx in buy_signals.index[:-1]:
                            mask1 = trades_df.index > idx
                            mask2 = trades_df['signal'] < 0
                            next_sell_idx = trades_df.loc[mask1 & mask2].index
                            if len(next_sell_idx) > 0:
                                buy_price = trades_df.loc[idx, 'close']
                                sell_price = trades_df.loc[next_sell_idx[0], 'close']
                                if sell_price > buy_price:
                                    profitable_buys += 1

                        feedback['signal_accuracy']['BUY_SIGNAL'] = profitable_buys / len(buy_signals) if len(
                            buy_signals) > 0 else 0.5

                    # 매도 신호 정확도 (기본값)
                    feedback['signal_accuracy']['SELL_SIGNAL'] = 0.7

            # 성과 지표 기반 조정
            if 'performance' in backtest_results:
                performance = backtest_results['performance']

                # 수익률이 좋으면 현재 설정 유지, 나쁘면 조정
                if performance.get('total_return', 0) < 0:
                    # RSI 임계값 조정
                    feedback['indicator_thresholds']['RSI_14'] = {
                        'upper': 70,  # 더 보수적으로
                        'lower': 30
                    }
                else:
                    feedback['indicator_thresholds']['RSI_14'] = {
                        'upper': 75,
                        'lower': 25
                    }

            # 패턴 성공률 (기본값 사용)
            feedback['pattern_success_rate'] = {
                'PATTERN_BULLISH_ENGULFING': 0.65,
                'PATTERN_BEARISH_HARAMI': 0.55,
                'PATTERN_MORNING_STAR': 0.70,
                'PATTERN_EVENING_STAR': 0.68
            }

        except Exception as e:
            self.logger.error(f"피드백 추출 중 오류: {e}")
            # 오류 시 기본값 반환
            return {
                'signal_accuracy': {'BUY_SIGNAL': 0.5, 'SELL_SIGNAL': 0.5},
                'pattern_success_rate': {'PATTERN_BULLISH_ENGULFING': 0.65},
                'indicator_thresholds': {'RSI_14': {'upper': 75, 'lower': 25}}
            }

        return feedback

# 사용 예시
if __name__ == "__main__":
    import asyncio
    from src.data_collection.collectors import DataCollector
    from src.backtesting.backtest_engine import BacktestEngine


    async def test_unified_processor():
        # 1. 통합 프로세서 생성
        processor = UnifiedDataProcessor(enable_ml_features=True)

        # 2. 백테스팅 데이터로 테스트
        collector = DataCollector()
        historical_data = collector.get_historical_data("BTCUSDT", "1h", "1 month ago UTC")

        print("백테스팅 모드 처리...")
        processed_data = processor.process_data(historical_data)
        print(f"처리된 데이터 크기: {processed_data.shape}")

        # 3. 백테스팅 실행
        engine = BacktestEngine(initial_capital=10000)

        # 간단한 전략 정의
        def simple_strategy(data):
            if data['CONFIRMED_BUY_SIGNAL'] > 0:
                return 'buy'
            elif data['CONFIRMED_SELL_SIGNAL'] > 0:
                return 'sell'
            return 'hold'

        # 백테스팅 실행
        results = engine.run(processed_data, simple_strategy)

        # 4. 피드백 추출 및 적용
        adapter = BacktestRealtimeAdapter(processor)
        feedback = adapter.extract_feedback_from_backtest(results)
        processor.update_feedback(feedback)

        print("\n피드백 데이터:")
        print(feedback)

        # 5. 실시간 데이터 시뮬레이션
        print("\n실시간 모드 시뮬레이션...")

        # 실시간 데이터 큐
        data_queue = asyncio.Queue()
        signal_queue = asyncio.Queue()

        # 실시간 처리 태스크 시작
        process_task = asyncio.create_task(
            processor.process_realtime_stream(data_queue, signal_queue)
        )

        # 시뮬레이션 데이터 생성
        for i in range(5):
            simulated_data = {
                'timestamp': datetime.now(),
                'open': 50000 + np.random.randn() * 100,
                'high': 50100 + np.random.randn() * 100,
                'low': 49900 + np.random.randn() * 100,
                'close': 50050 + np.random.randn() * 100,
                'volume': 1000 + np.random.randn() * 50
            }

            await data_queue.put(simulated_data)

            # 신호 확인
            try:
                signal = await asyncio.wait_for(signal_queue.get(), timeout=1.0)
                print(f"\n생성된 신호: {signal}")
            except asyncio.TimeoutError:
                print(f"데이터 포인트 {i + 1}: 신호 없음 (버퍼 채우는 중)")

        # 종료
        await data_queue.put(None)
        await process_task


    # 실행
    asyncio.run(test_unified_processor())
