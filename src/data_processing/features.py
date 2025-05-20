import pandas as pd
import numpy as np
from src.utils.logger import get_logger


class SignalGenerator:
    """패턴 인식 결과를 기반으로 매매 신호 생성"""

    def __init__(self, pattern_recognizer=None):
        """초기화"""
        self.logger = get_logger(__name__)
        self.pattern_recognizer = pattern_recognizer

    def generate_signals(self, df, pattern_weight=1.0, indicator_weight=0.5,
                         volume_weight=0.3, volatility_weight=0.2,
                         buy_threshold=2.0, sell_threshold=-2.0):
        """
        여러 요소를 고려하여 종합적인 매매 신호 생성

        Args:
            df: 패턴 감지 결과와 기술적 지표가 포함된 데이터프레임
            pattern_weight: 패턴 감지 결과의 가중치
            indicator_weight: 기술적 지표의 가중치
            volume_weight: 거래량 지표의 가중치
            volatility_weight: 변동성 지표의 가중치
            buy_threshold: 매수 신호 임계값
            sell_threshold: 매도 신호 임계값

        Returns:
            매매 신호가 추가된 데이터프레임
        """
        # 입력 데이터 검증
        if df is None or df.empty:
            self.logger.error("유효한 데이터프레임이 필요합니다.")
            return pd.DataFrame()

        # 데이터프레임 복사
        result = df.copy()

        try:
            # 1. 패턴 기반 신호 계산 (이미 계산되었으면 사용, 아니면 계산)
            if 'SIGNAL_STRENGTH' not in result.columns and self.pattern_recognizer:
                result = self.pattern_recognizer.get_all_signals(result)

            if 'SIGNAL_STRENGTH' not in result.columns:
                # 패턴 신호가 없으면 기본값 0 설정
                result['SIGNAL_STRENGTH'] = 0

            # 2. 기술적 지표 기반 신호 계산
            indicator_signal = self._calculate_indicator_signals(result)

            # 3. 거래량 분석
            volume_signal = self._analyze_volume(result)

            # 4. 변동성 분석
            volatility_signal = self._analyze_volatility(result)

            # 5. 종합 신호 계산 (가중 평균)
            result['COMBINED_SIGNAL'] = (
                    pattern_weight * result['SIGNAL_STRENGTH'] +
                    indicator_weight * indicator_signal +
                    volume_weight * volume_signal +
                    volatility_weight * volatility_signal
            )

            # 6. 매매 신호 생성
            result['BUY_SIGNAL'] = (result['COMBINED_SIGNAL'] >= buy_threshold).astype(int)
            result['SELL_SIGNAL'] = (result['COMBINED_SIGNAL'] <= sell_threshold).astype(int)

            # 7. 신호 신뢰도 계산 (0-100%)
            result['SIGNAL_CONFIDENCE'] = self._calculate_confidence(result)

            # 8. 포지션 크기 추천
            result['POSITION_SIZE'] = self._recommend_position_size(result)

            self.logger.info("매매 신호 생성 완료")
            return result

        except Exception as e:
            self.logger.error(f"신호 생성 중 오류 발생: {e}")
            return df

    def _calculate_indicator_signals(self, df):
        """기술적 지표를 기반으로 신호 계산"""
        signal = pd.Series(0, index=df.index)

        try:
            # 이동평균 기반 신호
            if all(col in df.columns for col in ['MA_20', 'MA_50']):
                # 골든 크로스 (단기 > 장기)
                signal += ((df['MA_20'] > df['MA_50']) &
                           (df['MA_20'].shift(1) <= df['MA_50'].shift(1))).astype(int) * 2

                # 데드 크로스 (단기 < 장기)
                signal -= ((df['MA_20'] < df['MA_50']) &
                           (df['MA_20'].shift(1) >= df['MA_50'].shift(1))).astype(int) * 2

                # 추세 방향
                signal += np.sign(df['MA_20'] - df['MA_50'])

            # RSI 기반 신호
            if 'RSI_14' in df.columns:
                # 과매수/과매도 조건
                signal += (df['RSI_14'] < 30).astype(int)  # 과매도 (매수 신호)
                signal -= (df['RSI_14'] > 70).astype(int)  # 과매수 (매도 신호)

            # MACD 기반 신호
            if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL']):
                # MACD 크로스오버
                signal += ((df['MACD'] > df['MACD_SIGNAL']) &
                           (df['MACD'].shift(1) <= df['MACD_SIGNAL'].shift(1))).astype(int) * 1.5

                # MACD 크로스언더
                signal -= ((df['MACD'] < df['MACD_SIGNAL']) &
                           (df['MACD'].shift(1) >= df['MACD_SIGNAL'].shift(1))).astype(int) * 1.5

            # 볼린저 밴드 기반 신호
            if all(col in df.columns for col in ['BB_lower_20', 'BB_upper_20']):
                # 하단 밴드 터치 (매수 신호)
                signal += (df['close'] <= df['BB_lower_20']).astype(int)

                # 상단 밴드 터치 (매도 신호)
                signal -= (df['close'] >= df['BB_upper_20']).astype(int)

        except Exception as e:
            self.logger.debug(f"지표 신호 계산 중 오류: {e}")

        return signal

    def _analyze_volume(self, df):
        """거래량 분석 기반 신호 계산"""
        signal = pd.Series(0, index=df.index)

        try:
            if 'volume' in df.columns:
                # 거래량 이동평균
                vol_ma = df['volume'].rolling(20).mean()

                # 거래량 급증
                volume_surge = (df['volume'] > 2 * vol_ma).astype(int)

                # 거래량 방향성 (가격 상승과 함께 거래량 증가 = 강한 신호)
                price_up = (df['close'] > df['close'].shift(1)).astype(int)
                price_down = (df['close'] < df['close'].shift(1)).astype(int)

                # 거래량과 가격 방향 결합
                signal += volume_surge * price_up
                signal -= volume_surge * price_down

        except Exception as e:
            self.logger.debug(f"거래량 분석 중 오류: {e}")

        return signal

    def _analyze_volatility(self, df):
        """변동성 분석 기반 신호 계산"""
        signal = pd.Series(0, index=df.index)

        try:
            if all(col in df.columns for col in ['high', 'low', 'close']):
                # ATR 계산 (Approximate)
                range_values = (df['high'] - df['low']).abs()
                atr = range_values.rolling(14).mean()

                # 변동성 수축 (낮은 ATR) - 돌파 준비
                volatility_squeeze = (atr < atr.rolling(20).quantile(0.2)).astype(int)

                # 변동성 확장 (높은 ATR) - 추세 형성 가능성
                volatility_expansion = (atr > atr.rolling(20).quantile(0.8)).astype(int)

                # 방향성 있는 확장만 신호로 간주
                close_diff = df['close'].diff()
                signal += volatility_expansion * np.sign(close_diff)

                # 변동성 수축 후 돌파 신호
                signal += volatility_squeeze.shift(1) * np.sign(close_diff) * 0.5

        except Exception as e:
            self.logger.debug(f"변동성 분석 중 오류: {e}")

        return signal

    def _calculate_confidence(self, df):
        """신호 신뢰도 계산 (0-100%)"""
        confidence = pd.Series(50, index=df.index)  # 기본값 50%

        try:
            # 신호 강도 기반 신뢰도
            abs_signal = df['COMBINED_SIGNAL'].abs()
            max_signal = abs_signal.rolling(100).max().fillna(abs_signal.max())
            normalized_strength = (abs_signal / max_signal * 40).clip(0, 40)
            confidence += normalized_strength

            # 패턴 확인 빈도 기반 추가 신뢰도
            if 'BULLISH_PATTERNS' in df.columns and 'BEARISH_PATTERNS' in df.columns:
                pattern_count = df['BULLISH_PATTERNS'] + df['BEARISH_PATTERNS']
                confidence += pattern_count.clip(0, 3) * 5

            # 저변동성 환경에서 신뢰도 감소
            if 'daily_volatility' in df.columns:
                low_vol = (df['daily_volatility'] < df['daily_volatility'].rolling(20).mean() * 0.5)
                confidence = confidence.where(~low_vol, confidence * 0.8)

            # 결과 클리핑
            confidence = confidence.clip(0, 100)

        except Exception as e:
            self.logger.debug(f"신뢰도 계산 중 오류: {e}")
            confidence = pd.Series(50, index=df.index)

        return confidence

    def _recommend_position_size(self, df):
        """신호 강도와 신뢰도 기반 포지션 크기 추천 (0-100%)"""
        position_size = pd.Series(0, index=df.index)

        try:
            # 기본 크기 설정 (매수: 양수, 매도: 음수)
            buy_size = df['BUY_SIGNAL'] * df['SIGNAL_CONFIDENCE'] / 100
            sell_size = -df['SELL_SIGNAL'] * df['SIGNAL_CONFIDENCE'] / 100
            position_size = buy_size + sell_size

            # 일관성 확인 - 같은 방향 신호가 지속되면 크기 증가
            for i in range(1, len(df)):
                if position_size.iloc[i] > 0 and position_size.iloc[i - 1] > 0:
                    position_size.iloc[i] = min(position_size.iloc[i] * 1.2, 1.0)
                elif position_size.iloc[i] < 0 and position_size.iloc[i - 1] < 0:
                    position_size.iloc[i] = max(position_size.iloc[i] * 1.2, -1.0)

        except Exception as e:
            self.logger.debug(f"포지션 크기 계산 중 오류: {e}")

        return position_size * 100  # 0-100% 스케일로 변환


class SignalFilter:
    """매매 신호 필터링 및 확인"""

    def __init__(self):
        """초기화"""
        self.logger = get_logger(__name__)

    def filter_signals(self, df, min_confidence=60, trend_filter=True):
        """
        생성된 신호 필터링

        Args:
            df: 신호가 있는 데이터프레임
            min_confidence: 최소 신뢰도 (0-100)
            trend_filter: 추세 필터 사용 여부

        Returns:
            필터링된 신호를 포함한 데이터프레임
        """
        result = df.copy()

        try:
            # 1. 신뢰도 필터
            result['FILTERED_BUY'] = ((result['BUY_SIGNAL'] == 1) &
                                      (result['SIGNAL_CONFIDENCE'] >= min_confidence)).astype(int)

            result['FILTERED_SELL'] = ((result['SELL_SIGNAL'] == 1) &
                                       (result['SIGNAL_CONFIDENCE'] >= min_confidence)).astype(int)

            # 2. 추세 필터 (선택적)
            if trend_filter:
                if all(col in result.columns for col in ['MA_50', 'MA_200']):
                    # 상승 추세에서만 매수 (단기이평선 > 장기이평선)
                    uptrend = result['MA_50'] > result['MA_200']
                    result['FILTERED_BUY'] = result['FILTERED_BUY'] & uptrend

                    # 하락 추세에서만 매도 (단기이평선 < 장기이평선)
                    downtrend = result['MA_50'] < result['MA_200']
                    result['FILTERED_SELL'] = result['FILTERED_SELL'] & downtrend
                elif 'MA_50' in result.columns:
                    # MA_200이 없는 경우 추세 방향 사용
                    uptrend = result['MA_50'] > result['MA_50'].shift(5)
                    result['FILTERED_BUY'] = result['FILTERED_BUY'] & uptrend

                    downtrend = result['MA_50'] < result['MA_50'].shift(5)
                    result['FILTERED_SELL'] = result['FILTERED_SELL'] & downtrend

            # 3. 거래량 확인
            if 'volume' in result.columns:
                # 거래량 이동평균 계산
                vol_ma = result['volume'].rolling(20).mean()

                # 거래량 필터 적용 (평균 거래량 이상)
                result['FILTERED_BUY'] = result['FILTERED_BUY'] & (result['volume'] >= vol_ma)
                result['FILTERED_SELL'] = result['FILTERED_SELL'] & (result['volume'] >= vol_ma)

            # 4. 최종 시그널 (필터링된 신호를 최종 신호로 결정)
            result['FINAL_SIGNAL'] = result['FILTERED_BUY'].astype(int) - result['FILTERED_SELL'].astype(int)

            self.logger.info("신호 필터링 완료")
        except Exception as e:
            self.logger.error(f"신호 필터링 중 오류 발생: {e}")

        return result

    def validate_with_volatility(self, df, volatility_factor=1.5):
        """변동성 기반 신호 검증"""
        result = df.copy()

        try:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                # 평균 실제 범위 (ATR) 계산
                high_low = result['high'] - result['low']
                high_close = (result['high'] - result['close'].shift()).abs()
                low_close = (result['low'] - result['close'].shift()).abs()

                ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = ranges.rolling(14).mean()

                # 매수/매도 목표가 계산
                buy_target = result['close'] + atr * volatility_factor
                sell_target = result['close'] - atr * volatility_factor

                # 목표가 컬럼 추가
                result['BUY_TARGET'] = buy_target.where(result['FILTERED_BUY'] == 1, np.nan)
                result['SELL_TARGET'] = sell_target.where(result['FILTERED_SELL'] == 1, np.nan)

                # 손절 수준 계산 (ATR의 1배)
                result['BUY_STOP'] = (result['close'] - atr).where(result['FILTERED_BUY'] == 1, np.nan)
                result['SELL_STOP'] = (result['close'] + atr).where(result['FILTERED_SELL'] == 1, np.nan)

                # 위험/보상 비율 계산
                buy_reward = (buy_target - result['close']).abs()
                buy_risk = (result['BUY_STOP'] - result['close']).abs()
                buy_rr = buy_reward / buy_risk

                sell_reward = (sell_target - result['close']).abs()
                sell_risk = (result['SELL_STOP'] - result['close']).abs()
                sell_rr = sell_reward / sell_risk

                # 위험/보상 비율이 좋은 신호만 유효화
                result['VALID_BUY'] = ((result['FILTERED_BUY'] == 1) & (buy_rr >= 2.0)).astype(int)
                result['VALID_SELL'] = ((result['FILTERED_SELL'] == 1) & (sell_rr >= 2.0)).astype(int)

                # 거래 이익 예상치
                result['EXPECTED_PROFIT'] = result['VALID_BUY'] * buy_reward - result['VALID_SELL'] * sell_reward

            self.logger.info("변동성 기반 신호 검증 완료")
        except Exception as e:
            self.logger.error(f"변동성 기반 검증 중 오류 발생: {e}")

        return result

    def validate_with_patterns(self, df, pattern_recognizer=None):
        """패턴 인식 결과로 신호 검증"""
        result = df.copy()

        try:
            # 패턴 감지 실행 (아직 안 되어 있다면)
            if pattern_recognizer and 'PATTERN_DESCRIPTIONS' not in result.columns:
                result = pattern_recognizer.get_pattern_descriptions(result)

            # 패턴 불일치 체크
            if all(col in result.columns for col in
                   ['FILTERED_BUY', 'FILTERED_SELL', 'BULLISH_PATTERNS', 'BEARISH_PATTERNS']):
                # 매수 신호와 베어리시 패턴이 동시에 있는 경우 경고
                conflicting_buy = (result['FILTERED_BUY'] == 1) & (result['BEARISH_PATTERNS'] > 0)

                # 매도 신호와 불리시 패턴이 동시에 있는 경우 경고
                conflicting_sell = (result['FILTERED_SELL'] == 1) & (result['BULLISH_PATTERNS'] > 0)

                # 결과 표시
                result['PATTERN_CONFLICT'] = ((conflicting_buy) | (conflicting_sell)).astype(int)

                # 신뢰도 조정
                result.loc[conflicting_buy, 'SIGNAL_CONFIDENCE'] *= 0.7
                result.loc[conflicting_sell, 'SIGNAL_CONFIDENCE'] *= 0.7

                # 패턴으로 신호 강화
                confirming_buy = (result['FILTERED_BUY'] == 1) & (result['BULLISH_PATTERNS'] > 0)
                confirming_sell = (result['FILTERED_SELL'] == 1) & (result['BEARISH_PATTERNS'] > 0)

                result.loc[confirming_buy, 'SIGNAL_CONFIDENCE'] *= 1.3
                result.loc[confirming_sell, 'SIGNAL_CONFIDENCE'] *= 1.3

                # 신뢰도 클리핑
                result['SIGNAL_CONFIDENCE'] = result['SIGNAL_CONFIDENCE'].clip(0, 100)

            self.logger.info("패턴 기반 신호 검증 완료")
        except Exception as e:
            self.logger.error(f"패턴 기반 검증 중 오류 발생: {e}")

        return result

    def summarize_signals(self, df):
        """신호 요약 및 트레이딩 추천 생성"""
        result = df.copy()

        try:
            # 최종 매매 추천
            result['BUY_RECOMMENDATION'] = ((result['FILTERED_BUY'] == 1) &
                                            (result['SIGNAL_CONFIDENCE'] >= 70)).astype(int)

            result['SELL_RECOMMENDATION'] = ((result['FILTERED_SELL'] == 1) &
                                             (result['SIGNAL_CONFIDENCE'] >= 70)).astype(int)

            # 포지션 크기 조절
            result['RECOMMENDED_SIZE'] = result['POSITION_SIZE'] * result['SIGNAL_CONFIDENCE'] / 100

            # 거래 설명 생성
            def create_trade_description(row):
                if row['BUY_RECOMMENDATION'] == 1:
                    desc = f"매수: 신뢰도 {row['SIGNAL_CONFIDENCE']:.1f}%, "
                    if 'PATTERN_DESCRIPTIONS' in result.columns and row['PATTERN_DESCRIPTIONS'] != "감지된 패턴 없음":
                        desc += f"패턴: {row['PATTERN_DESCRIPTIONS']}, "
                    desc += f"추천 비중: {abs(row['RECOMMENDED_SIZE']):.1f}%"
                    if 'BUY_TARGET' in result.columns:
                        desc += f", 목표가: {row['BUY_TARGET']:.2f}, 손절가: {row['BUY_STOP']:.2f}"
                    return desc

                elif row['SELL_RECOMMENDATION'] == 1:
                    desc = f"매도: 신뢰도 {row['SIGNAL_CONFIDENCE']:.1f}%, "
                    if 'PATTERN_DESCRIPTIONS' in result.columns and row['PATTERN_DESCRIPTIONS'] != "감지된 패턴 없음":
                        desc += f"패턴: {row['PATTERN_DESCRIPTIONS']}, "
                    desc += f"추천 비중: {abs(row['RECOMMENDED_SIZE']):.1f}%"
                    if 'SELL_TARGET' in result.columns:
                        desc += f", 목표가: {row['SELL_TARGET']:.2f}, 손절가: {row['SELL_STOP']:.2f}"
                    return desc

                return ""

            result['TRADE_DESCRIPTION'] = result.apply(create_trade_description, axis=1)

            self.logger.info("신호 요약 및 추천 생성 완료")
        except Exception as e:
            self.logger.error(f"신호 요약 중 오류 발생: {e}")

        return result


class TradingSignalProcessor:
    """트레이딩 신호 처리 및 관리"""

    def __init__(self, pattern_recognizer=None):
        """초기화"""
        self.logger = get_logger(__name__)
        self.pattern_recognizer = pattern_recognizer
        self.signal_generator = SignalGenerator(pattern_recognizer)
        self.signal_filter = SignalFilter()
        self.signals_history = {}

    def process_data(self, df, symbol, generate_indicators=True):
        """
        데이터 처리 및 트레이딩 신호 생성

        Args:
            df: OHLCV 데이터프레임
            symbol: 심볼 (티커)
            generate_indicators: 기술적 지표 생성 여부

        Returns:
            신호가 포함된 처리된 데이터프레임
        """
        if df is None or df.empty:
            self.logger.error("유효한 데이터프레임이 필요합니다.")
            return pd.DataFrame()

        result = df.copy()

        try:
            # 1. 기술적 지표 생성 (필요시)
            if generate_indicators:
                result = self._generate_technical_indicators(result)

                # 2. 패턴 인식 수행
            if self.pattern_recognizer:
                result = self.pattern_recognizer.get_all_signals(result)

                # 3. 트레이딩 신호 생성
            result = self.signal_generator.generate_signals(result)

            # 4. 신호 필터링
            result = self.signal_filter.filter_signals(result)

            # 5. 변동성 기반 검증
            result = self.signal_filter.validate_with_volatility(result)

            # 6. 패턴 기반 검증
            result = self.signal_filter.validate_with_patterns(result, self.pattern_recognizer)

            # 7. 신호 요약 및 추천
            result = self.signal_filter.summarize_signals(result)

            # 8. 최근 신호 저장
            self.signals_history[symbol] = result.iloc[-10:].copy()

            self.logger.info(f"{symbol} 트레이딩 신호 처리 완료")
            return result

        except Exception as e:
            self.logger.error(f"신호 처리 중 오류 발생: {e}")
            return df

    def _generate_technical_indicators(self, df):
        """기술적 지표 생성"""
        import talib

        result = df.copy()

        try:
            # 이동평균 (MA)
            result['MA_5'] = talib.SMA(result['close'], timeperiod=5)
            result['MA_10'] = talib.SMA(result['close'], timeperiod=10)
            result['MA_20'] = talib.SMA(result['close'], timeperiod=20)
            result['MA_50'] = talib.SMA(result['close'], timeperiod=50)
            result['MA_100'] = talib.SMA(result['close'], timeperiod=100)
            result['MA_200'] = talib.SMA(result['close'], timeperiod=200)

            # 지수 이동평균 (EMA)
            result['EMA_9'] = talib.EMA(result['close'], timeperiod=9)
            result['EMA_21'] = talib.EMA(result['close'], timeperiod=21)

            # MACD (Moving Average Convergence Divergence)
            result['MACD'], result['MACD_SIGNAL'], result['MACD_HIST'] = talib.MACD(
                result['close'], fastperiod=12, slowperiod=26, signalperiod=9)

            # RSI (Relative Strength Index)
            result['RSI_14'] = talib.RSI(result['close'], timeperiod=14)

            # 볼린저 밴드 (Bollinger Bands)
            result['BB_upper_20'], result['BB_middle_20'], result['BB_lower_20'] = talib.BBANDS(
                result['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

            # ATR (Average True Range)
            result['ATR_14'] = talib.ATR(result['high'], result['low'], result['close'], timeperiod=14)

            # OBV (On-Balance Volume)
            result['OBV'] = talib.OBV(result['close'], result['volume'])

            # CCI (Commodity Channel Index)
            result['CCI_14'] = talib.CCI(result['high'], result['low'], result['close'], timeperiod=14)

            # 스토캐스틱 (Stochastic)
            result['STOCH_K'], result['STOCH_D'] = talib.STOCH(
                result['high'], result['low'], result['close'],
                fastk_period=14, slowk_period=3, slowd_period=3)

            # ADX (Average Directional Index)
            result['ADX_14'] = talib.ADX(result['high'], result['low'], result['close'], timeperiod=14)

            # 파라볼릭 SAR
            result['SAR'] = talib.SAR(result['high'], result['low'], acceleration=0.02, maximum=0.2)

            # Ichimoku Cloud 컴포넌트
            high_9 = result['high'].rolling(window=9).max()
            low_9 = result['low'].rolling(window=9).min()
            result['TENKAN_SEN'] = (high_9 + low_9) / 2  # 전환선

            high_26 = result['high'].rolling(window=26).max()
            low_26 = result['low'].rolling(window=26).min()
            result['KIJUN_SEN'] = (high_26 + low_26) / 2  # 기준선

            # 선행 스팬 1 & 2
            result['SENKOU_SPAN_A'] = ((result['TENKAN_SEN'] + result['KIJUN_SEN']) / 2).shift(26)
            result['SENKOU_SPAN_B'] = ((result['high'].rolling(window=52).max() +
                                        result['low'].rolling(window=52).min()) / 2).shift(26)

            # 후행 스팬
            result['CHIKOU_SPAN'] = result['close'].shift(-26)

            # 추세 방향 지표
            result['TREND_5_20'] = np.where(result['MA_5'] > result['MA_20'], 1,
                                            np.where(result['MA_5'] < result['MA_20'], -1, 0))

            result['TREND_20_50'] = np.where(result['MA_20'] > result['MA_50'], 1,
                                             np.where(result['MA_20'] < result['MA_50'], -1, 0))

            result['TREND_50_200'] = np.where(result['MA_50'] > result['MA_200'], 1,
                                              np.where(result['MA_50'] < result['MA_200'], -1, 0))

            # 이격도 계산 (Price Rate of Change)
            result['DEVIATION_20'] = (result['close'] / result['MA_20'] - 1) * 100
            result['DEVIATION_50'] = (result['close'] / result['MA_50'] - 1) * 100

            # 변동성 지표
            result['DAILY_VOLATILITY'] = (result['high'] - result['low']) / result['close'] * 100
            result['VOL_MA_20'] = result['DAILY_VOLATILITY'].rolling(20).mean()

            # 거래량 지표
            result['VOLUME_MA_20'] = result['volume'].rolling(20).mean()
            result['VOLUME_RATIO'] = result['volume'] / result['VOLUME_MA_20']

            self.logger.info("기술적 지표 생성 완료")
        except Exception as e:
            self.logger.error(f"기술적 지표 생성 중 오류 발생: {e}")

        return result

    def get_latest_signals(self, symbol=None):
        """최근 신호 조회"""
        if symbol:
            if symbol in self.signals_history:
                return self.signals_history[symbol]
            else:
                self.logger.warning(f"{symbol}에 대한 신호 기록이 없습니다.")
                return pd.DataFrame()
        else:
            # 모든 심볼의 최신 신호 조회
            latest_signals = {}
            for sym, signals in self.signals_history.items():
                latest_signals[sym] = signals.iloc[-1].copy()

            return pd.DataFrame.from_dict(latest_signals, orient='index')

    def get_signal_stats(self, df):
        """신호 통계 계산"""
        stats = {}

        try:
            if 'BUY_RECOMMENDATION' in df.columns and 'SELL_RECOMMENDATION' in df.columns:
                # 매수/매도 신호 개수
                stats['buy_signals'] = df['BUY_RECOMMENDATION'].sum()
                stats['sell_signals'] = df['SELL_RECOMMENDATION'].sum()

                # 신호 신뢰도 평균
                buy_confidence = df.loc[df['BUY_RECOMMENDATION'] == 1, 'SIGNAL_CONFIDENCE']
                sell_confidence = df.loc[df['SELL_RECOMMENDATION'] == 1, 'SIGNAL_CONFIDENCE']

                stats['avg_buy_confidence'] = buy_confidence.mean() if len(buy_confidence) > 0 else 0
                stats['avg_sell_confidence'] = sell_confidence.mean() if len(sell_confidence) > 0 else 0

                # 추세 방향
                if 'TREND_50_200' in df.columns:
                    stats['trend_direction'] = df['TREND_50_200'].iloc[-1]
                    stats['trend_strength'] = abs(df['TREND_50_200'].rolling(50).mean().iloc[-1])

                # 변동성
                if 'DAILY_VOLATILITY' in df.columns:
                    stats['current_volatility'] = df['DAILY_VOLATILITY'].iloc[-1]
                    stats['volatility_trend'] = (df['DAILY_VOLATILITY'].iloc[-1] /
                                                 df['DAILY_VOLATILITY'].rolling(20).mean().iloc[-1])

                # 거래량 트렌드
                if 'VOLUME_RATIO' in df.columns:
                    stats['volume_trend'] = df['VOLUME_RATIO'].iloc[-1]

            self.logger.info("신호 통계 계산 완료")
        except Exception as e:
            self.logger.error(f"신호 통계 계산 중 오류 발생: {e}")

        return stats

    def generate_trading_plan(self, df, symbol, risk_per_trade=0.02):
        """트레이딩 계획 생성"""
        plan = {}

        try:
            latest = df.iloc[-1]

            if latest['BUY_RECOMMENDATION'] == 1:
                plan['action'] = 'BUY'
                plan['symbol'] = symbol
                plan['confidence'] = latest['SIGNAL_CONFIDENCE']
                plan['entry_price'] = latest['close']

                if 'BUY_TARGET' in latest and not pd.isna(latest['BUY_TARGET']):
                    plan['target_price'] = latest['BUY_TARGET']
                else:
                    # 목표가가 없으면 ATR의 2배로 설정
                    atr = latest.get('ATR_14', latest['close'] * 0.02)  # 기본값은 가격의 2%
                    plan['target_price'] = latest['close'] * (1 + (atr / latest['close']) * 2)

                if 'BUY_STOP' in latest and not pd.isna(latest['BUY_STOP']):
                    plan['stop_loss'] = latest['BUY_STOP']
                else:
                    # 손절가가 없으면 ATR의 1배로 설정
                    atr = latest.get('ATR_14', latest['close'] * 0.02)
                    plan['stop_loss'] = latest['close'] * (1 - (atr / latest['close']))

                # 위험/보상 비율
                risk = abs(plan['entry_price'] - plan['stop_loss'])
                reward = abs(plan['target_price'] - plan['entry_price'])
                plan['risk_reward_ratio'] = reward / risk if risk > 0 else 0

                # 포지션 크기 추천
                position_pct = min(risk_per_trade / (risk / plan['entry_price']), 1.0)
                plan['position_size_pct'] = position_pct * 100

                # 패턴 설명
                if 'PATTERN_DESCRIPTIONS' in latest:
                    plan['patterns'] = latest['PATTERN_DESCRIPTIONS']

                # 추가 지표 정보
                if 'RSI_14' in latest:
                    plan['rsi'] = latest['RSI_14']
                if 'TREND_50_200' in latest:
                    plan['trend'] = 'BULLISH' if latest['TREND_50_200'] > 0 else 'BEARISH'

            elif latest['SELL_RECOMMENDATION'] == 1:
                plan['action'] = 'SELL'
                plan['symbol'] = symbol
                plan['confidence'] = latest['SIGNAL_CONFIDENCE']
                plan['entry_price'] = latest['close']

                if 'SELL_TARGET' in latest and not pd.isna(latest['SELL_TARGET']):
                    plan['target_price'] = latest['SELL_TARGET']
                else:
                    # 목표가가 없으면 ATR의 2배로 설정
                    atr = latest.get('ATR_14', latest['close'] * 0.02)
                    plan['target_price'] = latest['close'] * (1 - (atr / latest['close']) * 2)

                if 'SELL_STOP' in latest and not pd.isna(latest['SELL_STOP']):
                    plan['stop_loss'] = latest['SELL_STOP']
                else:
                    # 손절가가 없으면 ATR의 1배로 설정
                    atr = latest.get('ATR_14', latest['close'] * 0.02)
                    plan['stop_loss'] = latest['close'] * (1 + (atr / latest['close']))

                # 위험/보상 비율
                risk = abs(plan['entry_price'] - plan['stop_loss'])
                reward = abs(plan['target_price'] - plan['entry_price'])
                plan['risk_reward_ratio'] = reward / risk if risk > 0 else 0

                # 포지션 크기 추천
                position_pct = min(risk_per_trade / (risk / plan['entry_price']), 1.0)
                plan['position_size_pct'] = position_pct * 100

                # 패턴 설명
                if 'PATTERN_DESCRIPTIONS' in latest:
                    plan['patterns'] = latest['PATTERN_DESCRIPTIONS']

                # 추가 지표 정보
                if 'RSI_14' in latest:
                    plan['rsi'] = latest['RSI_14']
                if 'TREND_50_200' in latest:
                    plan['trend'] = 'BULLISH' if latest['TREND_50_200'] > 0 else 'BEARISH'
            else:
                plan['action'] = 'HOLD'
                plan['symbol'] = symbol
                plan['close_price'] = latest['close']

                # 현재 시장 상태 정보
                if 'RSI_14' in latest:
                    plan['rsi'] = latest['RSI_14']
                if 'TREND_50_200' in latest:
                    plan['trend'] = 'BULLISH' if latest['TREND_50_200'] > 0 else 'BEARISH'
                if 'DAILY_VOLATILITY' in latest:
                    plan['volatility'] = latest['DAILY_VOLATILITY']

            self.logger.info(f"{symbol} 트레이딩 계획 생성 완료")
        except Exception as e:
            self.logger.error(f"트레이딩 계획 생성 중 오류 발생: {e}")
            plan['action'] = 'ERROR'
            plan['error'] = str(e)

        return plan


def main():
    # 1. 데이터 수집
    from src.data_collection.collectors import DataCollector
    from src.data_processing.pattern_recognition import PatternRecognition
    from src.utils.logger import get_logger

    logger = get_logger("TradingSignalExample")
    collector = DataCollector()

    # 비트코인 4시간 데이터 가져오기
    symbol = "BTCUSDT"
    interval = "4h"
    df = collector.get_historical_data(symbol, interval, "3 months ago UTC")

    if df is None or df.empty:
        logger.error("데이터 수집 실패")
        return

    logger.info(f"{symbol} {interval} 데이터 수집 완료: {len(df)} 행")

    # 2. 패턴 인식기 초기화
    pattern_recognizer = PatternRecognition()

    # 3. 시그널 처리기 초기화
    signal_processor = TradingSignalProcessor(pattern_recognizer)

    # 4. 데이터 처리 및 신호 생성
    processed_data = signal_processor.process_data(df, symbol)

    # 5. 결과 출력
    if processed_data is not None and not processed_data.empty:
        # 최근 10개 데이터의 패턴 및 신호 출력
        recent_data = processed_data.iloc[-10:].copy()

        logger.info("\n패턴 감지 결과 (마지막 10행):")
        selected_columns = ['close']

        if 'PATTERN_DESCRIPTIONS' in recent_data.columns:
            selected_columns.append('PATTERN_DESCRIPTIONS')

        print(recent_data[selected_columns])

        # 가장 최근 날짜의 신호 분석
        last_date = recent_data.index[-1]
        last_row = recent_data.iloc[-1]

        print(f"\n흥미로운 패턴 발견 날짜: {last_date}")
        print("활성화된 패턴:")

        # 패턴 정보 출력
        if 'DOJI' in recent_data.columns and last_row['DOJI'] == 1:
            print(f"- DOJI: 불리시")
        if 'GRAVESTONE_DOJI' in recent_data.columns and last_row['GRAVESTONE_DOJI'] == 1:
            print(f"- GRAVESTONE_DOJI: 불리시")
        if 'ENGULFING' in recent_data.columns and last_row['ENGULFING'] != 0:
            direction = "불리시" if last_row['ENGULFING'] > 0 else "베어리시"
            print(f"- ENGULFING: {direction}")
        if 'BUY_SIGNAL' in recent_data.columns and last_row['BUY_SIGNAL'] == 1:
            print(f"- BUY_SIGNAL: 매수")
        elif 'BUY_SIGNAL' in recent_data.columns and last_row['BUY_SIGNAL'] == -1:
            print(f"- BUY_SIGNAL: 중립")
        if 'TRIANGLE_PATTERN' in recent_data.columns and last_row['TRIANGLE_PATTERN'] != 0:
            print(f"- TRIANGLE_PATTERN: {last_row['TRIANGLE_PATTERN']}")
        if 'GOLDEN_CROSS' in recent_data.columns and last_row['GOLDEN_CROSS'] == 1:
            print(f"- GOLDEN_CROSS: {last_row['GOLDEN_CROSS']}")

        # 패턴 설명
        if 'PATTERN_DESCRIPTIONS' in recent_data.columns:
            print(f"패턴 설명: {last_row['PATTERN_DESCRIPTIONS']}")

        # 신호 강도
        if 'SIGNAL_STRENGTH' in recent_data.columns:
            print(f"신호 강도: {last_row['SIGNAL_STRENGTH']}")

            # 신호 해석
            if last_row['SIGNAL_STRENGTH'] >= 3:
                print("신호 해석: 강한 매수 신호")
            elif last_row['SIGNAL_STRENGTH'] > 0:
                print("신호 해석: 약한 매수 신호")
            elif last_row['SIGNAL_STRENGTH'] == 0:
                print("신호 해석: 중립")
            elif last_row['SIGNAL_STRENGTH'] > -3:
                print("신호 해석: 약한 매도 신호")
            else:
                print("신호 해석: 강한 매도 신호")

        # 최종 추천 출력
        logger.info("\n트레이딩 추천:")
        if 'BUY_RECOMMENDATION' in recent_data.columns and last_row['BUY_RECOMMENDATION'] == 1:
            print(f"매수 추천 (신뢰도: {last_row['SIGNAL_CONFIDENCE']:.1f}%)")
            if 'TRADE_DESCRIPTION' in recent_data.columns:
                print(last_row['TRADE_DESCRIPTION'])
        elif 'SELL_RECOMMENDATION' in recent_data.columns and last_row['SELL_RECOMMENDATION'] == 1:
            print(f"매도 추천 (신뢰도: {last_row['SIGNAL_CONFIDENCE']:.1f}%)")
            if 'TRADE_DESCRIPTION' in recent_data.columns:
                print(last_row['TRADE_DESCRIPTION'])
        else:
            print("관망 추천")

        # 트레이딩 계획 생성
        trading_plan = signal_processor.generate_trading_plan(processed_data, symbol)

        logger.info("\n트레이딩 계획:")
        for key, value in trading_plan.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.2f}")
            else:
                print(f"- {key}: {value}")

        # 차트 시각화 (선택 사항)
        try:
            visualize_signals(processed_data, symbol, interval)
        except Exception as e:
            logger.error(f"차트 시각화 중 오류 발생: {e}")
    else:
        logger.error("신호 처리 실패")


def visualize_signals(df, symbol, interval):
    """신호 시각화"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.font_manager as fm
    import pandas as pd
    import os
    import platform

    logger = get_logger("Visualization")

    try:
        # 운영체제 감지 및 한글 폰트 설정
        system = platform.system()
        logger.info(f"감지된 운영체제: {system}")

        if system == 'Windows':
            # Windows 시스템용 폰트
            if 'Malgun Gothic' in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.family'] = 'Malgun Gothic'
                logger.info("Windows 폰트 'Malgun Gothic'을 설정했습니다.")
            else:
                # 대체 폰트
                plt.rcParams['font.family'] = 'Gulim'
                logger.info("Windows 대체 폰트 'Gulim'을 설정했습니다.")

        elif system == 'Darwin':  # macOS
            # macOS 시스템용 폰트
            if any('AppleGothic' in f.name for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = 'AppleGothic'
                logger.info("macOS 폰트 'AppleGothic'을 설정했습니다.")
            else:
                # 대체 폰트
                plt.rcParams['font.family'] = 'Arial Unicode MS'
                logger.info("macOS 대체 폰트 'Arial Unicode MS'를 설정했습니다.")

        elif system == 'Linux':
            # 리눅스 시스템용 폰트
            linux_fonts = ['NanumGothic', 'NanumBarunGothic', 'UnDotum', 'UnBatang']
            font_found = False

            for font in linux_fonts:
                if any(font in f.name for f in fm.fontManager.ttflist):
                    plt.rcParams['font.family'] = font
                    logger.info(f"Linux 폰트 '{font}'를 설정했습니다.")
                    font_found = True
                    break

            if not font_found:
                # 대체 폰트
                plt.rcParams['font.family'] = 'DejaVu Sans'
                logger.info("Linux 대체 폰트 'DejaVu Sans'를 설정했습니다.")

        else:
            # 기타 시스템
            logger.warning(f"인식되지 않은 운영체제: {system}. 기본 폰트를 사용합니다.")

        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False

        # 폰트 검색 디버깅
        logger.debug(f"현재 설정된 폰트 패밀리: {plt.rcParams['font.family']}")

        # 폰트 사용 가능성 검사 함수
        def check_font_availability():
            from matplotlib.font_manager import findfont, FontProperties
            font = findfont(FontProperties(family=plt.rcParams['font.family']))
            logger.debug(f"실제 사용되는 폰트 경로: {font}")
            return font

        check_font_availability()

        # output 디렉토리가 없으면 생성
        output_dir = './output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"'{output_dir}' 디렉토리를 생성했습니다.")

        # 마지막 100개 데이터 포인트만 시각화
        plot_df = df.iloc[-100:].copy()

        # 그래프 생성
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

        # 메인 가격 차트
        ax1 = axes[0]
        ax1.plot(plot_df.index, plot_df['close'], label='종가', color='black', alpha=0.75)

        # 이동평균선 추가
        if 'MA_20' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['MA_20'], label='MA 20', color='blue', alpha=0.6)
        if 'MA_50' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['MA_50'], label='MA 50', color='red', alpha=0.6)

        # 매수/매도 신호 추가
        if 'BUY_RECOMMENDATION' in plot_df.columns:
            buys = plot_df[plot_df['BUY_RECOMMENDATION'] == 1]
            ax1.scatter(buys.index, buys['close'], color='green', s=100, marker='^', label='매수 신호')

        if 'SELL_RECOMMENDATION' in plot_df.columns:
            sells = plot_df[plot_df['SELL_RECOMMENDATION'] == 1]
            ax1.scatter(sells.index, sells['close'], color='red', s=100, marker='v', label='매도 신호')

        # 볼린저 밴드 추가
        if all(col in plot_df.columns for col in ['BB_upper_20', 'BB_lower_20']):
            ax1.plot(plot_df.index, plot_df['BB_upper_20'], 'k--', alpha=0.3)
            ax1.plot(plot_df.index, plot_df['BB_lower_20'], 'k--', alpha=0.3)
            ax1.fill_between(plot_df.index, plot_df['BB_lower_20'], plot_df['BB_upper_20'], alpha=0.1, color='gray')

        # 목표가 및 손절가 추가
        if 'BUY_TARGET' in plot_df.columns:
            for idx, row in buys.iterrows():
                if not pd.isna(row['BUY_TARGET']):
                    ax1.plot([idx, idx], [row['close'], row['BUY_TARGET']], 'g--', alpha=0.5)
                    ax1.plot([idx, plot_df.index[-1]], [row['BUY_TARGET'], row['BUY_TARGET']], 'g--', alpha=0.5)

                if not pd.isna(row['BUY_STOP']):
                    ax1.plot([idx, idx], [row['close'], row['BUY_STOP']], 'r--', alpha=0.5)
                    ax1.plot([idx, plot_df.index[-1]], [row['BUY_STOP'], row['BUY_STOP']], 'r--', alpha=0.5)

        if 'SELL_TARGET' in plot_df.columns:
            for idx, row in sells.iterrows():
                if not pd.isna(row['SELL_TARGET']):
                    ax1.plot([idx, idx], [row['close'], row['SELL_TARGET']], 'g--', alpha=0.5)
                    ax1.plot([idx, plot_df.index[-1]], [row['SELL_TARGET'], row['SELL_TARGET']], 'g--', alpha=0.5)

                if not pd.isna(row['SELL_STOP']):
                    ax1.plot([idx, idx], [row['close'], row['SELL_STOP']], 'r--', alpha=0.5)
                    ax1.plot([idx, plot_df.index[-1]], [row['SELL_STOP'], row['SELL_STOP']], 'r--', alpha=0.5)

        ax1.set_title(f'{symbol} {interval} 차트')
        ax1.set_ylabel('가격')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        # 거래량 차트
        ax2 = axes[1]
        if 'volume' in plot_df.columns:
            ax2.bar(plot_df.index, plot_df['volume'], color='blue', alpha=0.5, width=0.8)
            if 'VOLUME_MA_20' in plot_df.columns:
                ax2.plot(plot_df.index, plot_df['VOLUME_MA_20'], color='red', alpha=0.7)

            ax2.set_ylabel('거래량')
            ax2.grid(True, alpha=0.3)

        # RSI 차트
        ax3 = axes[2]
        if 'RSI_14' in plot_df.columns:
            ax3.plot(plot_df.index, plot_df['RSI_14'], color='purple', alpha=0.7)
            ax3.axhline(y=70, color='r', linestyle='--', alpha=0.3)
            ax3.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            ax3.fill_between(plot_df.index, 70, plot_df['RSI_14'].where(plot_df['RSI_14'] >= 70), color='r', alpha=0.1)
            ax3.fill_between(plot_df.index, 30, plot_df['RSI_14'].where(plot_df['RSI_14'] <= 30), color='g', alpha=0.1)

            ax3.set_ylabel('RSI')
            ax3.grid(True, alpha=0.3)

        # x축 날짜 포맷 설정
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f'./output/{symbol}_{interval}_signals.png')
        plt.close()

        logger.info(f"차트를 ./output/{symbol}_{interval}_signals.png에 저장했습니다.")
    except Exception as e:
        logger.error(f"차트 시각화 중 오류 발생: {e}")


if __name__ == "__main__":
    main()