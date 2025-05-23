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

        # TradingSignalProcessor 초기화 - pattern_recognizer 인자 전달
        self.signal_processor = TradingSignalProcessor(self.pattern_recognition)

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

        # 1. 기본 전처리
        df = self._preprocess_data(df)

        # 2. TradingSignalProcessor를 사용한 전체 처리
        # generate_indicators=False로 설정하여 중복 지표 계산 방지
        df = self.signal_processor.process_data(df, symbol="BTCUSDT", generate_indicators=False)

        # 3. 기술적 지표 계산 (signal_processor에서 안했다면)
        if 'MA_20' not in df.columns:
            df = self.indicators.add_all_indicators(df)

        # 4. 패턴 인식 (signal_processor에서 안했다면)
        if 'PATTERN_BULLISH_ENGULFING' not in df.columns:
            df = self.pattern_recognition.detect_all_patterns(df)
            df = self.pattern_recognition.find_chart_patterns(df)
            df = self.pattern_recognition.detect_advanced_patterns(df)

        # 5. ML 특징 생성 (선택사항)
        if self.enable_ml_features:
            df = self._generate_ml_features(df)

        # 6. 피드백 데이터 적용
        df = self._apply_feedback(df)

        self.logger.info("배치 데이터 처리 완료")
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리

        Args:
            df: 원본 DataFrame

        Returns:
            전처리된 DataFrame
        """
        # 결측치 처리 - deprecated 메서드 수정
        df = df.ffill().bfill()  # fillna(method='ffill') 대신 ffill() 사용

        # 이상치 제거
        df = self._remove_outliers(df)

        # 타임스탬프 인덱스 설정
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('timestamp', inplace=True)

        return df

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

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 전처리

        Args:
            df: 원본 DataFrame

        Returns:
            전처리된 DataFrame
        """
        # 결측치 처리
        df = df.ffill().bfill()

        # 이상치 제거
        df = self._remove_outliers(df)

        # 타임스탬프 인덱스 설정
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('timestamp', inplace=True)

        return df

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

            # 매수/매도 신호 확인
            if 'BUY_RECOMMENDATION' in processed_data.columns and latest_data.get('BUY_RECOMMENDATION', 0) == 1:
                signals['primary_signal'] = 'BUY'
                signals['signal_strength'] = latest_data.get('COMBINED_SIGNAL', 0)
                signals['confidence'] = latest_data.get('SIGNAL_CONFIDENCE', 50)

            elif 'SELL_RECOMMENDATION' in processed_data.columns and latest_data.get('SELL_RECOMMENDATION', 0) == 1:
                signals['primary_signal'] = 'SELL'
                signals['signal_strength'] = latest_data.get('COMBINED_SIGNAL', 0)
                signals['confidence'] = latest_data.get('SIGNAL_CONFIDENCE', 50)

            # 지표 정보
            for col in processed_data.columns:
                if col in ['RSI_14', 'MACD', 'MA_20', 'MA_50', 'BB_upper_20', 'BB_lower_20']:
                    signals['indicators'][col] = latest_data.get(col, 0)

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

        while True:
            try:
                # 입력 큐에서 데이터 가져오기
                data = await input_queue.get()

                # None이면 종료 신호
                if data is None:
                    break

                # 데이터 처리
                processed = self._process_streaming_data(data)

                # 처리된 데이터가 있으면 출력 큐에 추가
                if processed and 'signals' in processed:
                    await output_queue.put(processed['signals'])

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

            # 최소 데이터 포인트 확인
            if len(self.data_buffer) < self.min_data_points:
                self.logger.debug(f"버퍼 크기 부족: {len(self.data_buffer)}/{self.min_data_points}")
                return None

            # 버퍼를 DataFrame으로 변환
            df = pd.DataFrame(list(self.data_buffer))
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

            # 배치 처리 로직 재사용
            processed_df = self._process_batch_data(df)

            # 최신 데이터만 반환
            latest_processed = processed_df.iloc[-1].to_dict()

            # 처리된 데이터를 버퍼에 저장
            self.processed_buffer.append(latest_processed)

            # 신호 생성
            signals = self.generate_signals(processed_df)

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
/* <<<<<<<<<<<<<<  ✨ Windsurf Command ⭐ >>>>>>>>>>>>>>>> */
/* <<<<<<<<<<  6e898b31-46ec-4ed4-a478-3a4c9b23ad38  >>>>>>>>>>> */
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
