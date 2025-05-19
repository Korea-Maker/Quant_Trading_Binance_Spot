# pattern_recognition.py

import pandas as pd
import numpy as np
import talib
from typing import List, Dict, Optional, Union, Tuple
from src.utils.logger import get_logger


class PatternRecognition:
    """캔들스틱 및 차트 패턴 인식 클래스"""

    def __init__(self):
        """초기화"""
        self.logger = get_logger(__name__)
        self.pattern_functions = self._get_pattern_functions()

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """데이터프레임 유효성 검사

        Args:
            df: 검사할 데이터프레임

        Returns:
            bool: 유효성 여부
        """
        if df is None or df.empty:
            self.logger.error("유효한 데이터프레임이 필요합니다.")
            return False

        required_columns = ['open', 'high', 'low', 'close']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"데이터프레임에 참조할 컬럼이 없습니다: {missing}")
            return False

        return True

    def _get_pattern_functions(self) -> Dict[str, Dict]:
        """TA-Lib 패턴 인식 함수와 설명 매핑

        Returns:
            Dict: 패턴 함수 딕셔너리
        """
        return {
            # 망치형(Hammer) 패턴
            'HAMMER': {
                'func': talib.CDLHAMMER,
                'type': 'bullish',
                'desc': '망치형 - 하락 추세에서 반전 신호'
            },
            'HANGING_MAN': {
                'func': talib.CDLHANGINGMAN,
                'type': 'bearish',
                'desc': '교수형 - 상승 추세에서 반전 신호'
            },
            'INVERTED_HAMMER': {
                'func': talib.CDLINVERTEDHAMMER,
                'type': 'bullish',
                'desc': '역망치형 - 하락 추세에서 반전 신호'
            },
            'SHOOTING_STAR': {
                'func': talib.CDLSHOOTINGSTAR,
                'type': 'bearish',
                'desc': '유성형 - 상승 추세에서 반전 신호'
            },

            # 도지(Doji) 패턴
            'DOJI': {
                'func': talib.CDLDOJI,
                'type': 'neutral',
                'desc': '도지 - 시가와 종가가 거의 같은 불확실성 패턴'
            },
            'DRAGONFLY_DOJI': {
                'func': talib.CDLDRAGONFLYDOJI,
                'type': 'bullish',
                'desc': '잠자리형 도지 - 하락 추세에서 반전 신호'
            },
            'GRAVESTONE_DOJI': {
                'func': talib.CDLGRAVESTONEDOJI,
                'type': 'bearish',
                'desc': '묘비형 도지 - 상승 추세에서 반전 신호'
            },

            # 스타(Star) 패턴
            'MORNING_STAR': {
                'func': talib.CDLMORNINGSTAR,
                'type': 'bullish',
                'desc': '샛별형 - 하락 추세에서 강한 반전 신호'
            },
            'EVENING_STAR': {
                'func': talib.CDLEVENINGSTAR,
                'type': 'bearish',
                'desc': '저녁별형 - 상승 추세에서 강한 반전 신호'
            },
            'MORNING_DOJI_STAR': {
                'func': talib.CDLMORNINGDOJISTAR,
                'type': 'bullish',
                'desc': '도지 샛별형 - 하락 추세에서 매우 강한 반전 신호'
            },
            'EVENING_DOJI_STAR': {
                'func': talib.CDLEVENINGDOJISTAR,
                'type': 'bearish',
                'desc': '도지 저녁별형 - 상승 추세에서 매우 강한 반전 신호'
            },

            # 주목할만한 패턴
            'ENGULFING': {
                'func': talib.CDLENGULFING,
                'type': 'dynamic',  # 상승 또는 하락 추세에 따라 다름
                'desc': '감싸는 형태 - 이전 캔들을 완전히 감싸는 강한 반전 신호'
            },
            'HARAMI': {
                'func': talib.CDLHARAMI,
                'type': 'dynamic',
                'desc': '하라미 - 이전 캔들 안에 형성되는 약한 반전 신호'
            },
            'PIERCING': {
                'func': talib.CDLPIERCING,
                'type': 'bullish',
                'desc': '관통형 - 하락 추세에서 반전 신호'
            },
            'DARK_CLOUD_COVER': {
                'func': talib.CDLDARKCLOUDCOVER,
                'type': 'bearish',
                'desc': '먹구름 - 상승 추세에서 반전 신호'
            },

            # 3개 캔들 패턴
            'THREE_WHITE_SOLDIERS': {
                'func': talib.CDL3WHITESOLDIERS,
                'type': 'bullish',
                'desc': '세 개의 백색 병사 - 강한 상승 신호'
            },
            'THREE_BLACK_CROWS': {
                'func': talib.CDL3BLACKCROWS,
                'type': 'bearish',
                'desc': '세 개의 까마귀 - 강한 하락 신호'
            },
            'THREE_INSIDE_UP': {
                'func': talib.CDL3INSIDE,
                'type': 'bullish',
                'desc': '상승 내부 패턴 - 하락 추세에서 반전 신호'
            },
            'THREE_OUTSIDE_UP': {
                'func': talib.CDL3OUTSIDE,
                'type': 'bullish',
                'desc': '상승 외부 패턴 - 하락 추세에서 강한 반전 신호'
            },

            # 갭 패턴
            'BREAKAWAY': {
                'func': talib.CDLBREAKAWAY,
                'type': 'dynamic',
                'desc': '탈출 갭 - 강한 추세 전환 신호'
            },
            'KICKING': {
                'func': talib.CDLKICKING,
                'type': 'dynamic',
                'desc': '발차기 - 매우 강한 반전 신호'
            }
        }

    def detect_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 캔들스틱 패턴 감지

        Args:
            df: OHLCV 데이터프레임

        Returns:
            패턴이 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # 모든 패턴 함수 적용
            for pattern_name, pattern_info in self.pattern_functions.items():
                pattern_func = pattern_info['func']
                result[f'PATTERN_{pattern_name}'] = pattern_func(
                    result['open'], result['high'], result['low'], result['close']
                )

                # 패턴 총합 컬럼 추가 (불리시: 양수, 베어리시: 음수)
            pattern_cols = [col for col in result.columns if col.startswith('PATTERN_')]

            # 각 행에서 불리시 패턴(100)과 베어리시 패턴(-100) 개수 합산
            result['BULLISH_PATTERNS'] = result[pattern_cols].apply(
                lambda x: sum(1 for val in x if val == 100), axis=1
            )
            result['BEARISH_PATTERNS'] = result[pattern_cols].apply(
                lambda x: sum(1 for val in x if val == -100), axis=1
            )

            # 불리시-베어리시 신호의 차이 (양수면 불리시 우세, 음수면 베어리시 우세)
            result['PATTERN_STRENGTH'] = result['BULLISH_PATTERNS'] - result['BEARISH_PATTERNS']

            self.logger.info("모든 캔들스틱 패턴 감지 완료")
        except Exception as e:
            self.logger.error(f"패턴 감지 중 오류 발생: {e}")

        return result

    def detect_pattern_group(self, df: pd.DataFrame, pattern_type: str = 'all') -> pd.DataFrame:
        """특정 유형의 캔들스틱 패턴 감지

        Args:
            df: OHLCV 데이터프레임
            pattern_type: 패턴 유형 ('bullish', 'bearish', 'neutral', 'dynamic', 'all')

        Returns:
            패턴이 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # 패턴 유형에 따라 필터링
            if pattern_type == 'all':
                filtered_patterns = self.pattern_functions
            else:
                filtered_patterns = {
                    name: info for name, info in self.pattern_functions.items()
                    if info['type'] == pattern_type or info['type'] == 'dynamic'
                }

                # 필터링된 패턴 적용
            for pattern_name, pattern_info in filtered_patterns.items():
                pattern_func = pattern_info['func']
                result[f'PATTERN_{pattern_name}'] = pattern_func(
                    result['open'], result['high'], result['low'], result['close']
                )

            self.logger.info(f"{pattern_type} 유형의 캔들스틱 패턴 감지 완료")
        except Exception as e:
            self.logger.error(f"패턴 감지 중 오류 발생: {e}")

        return result

    def get_pattern_signals(self, df: pd.DataFrame, min_strength: int = 1) -> pd.DataFrame:
        """패턴 기반 매매 신호 생성

        Args:
            df: 패턴이 감지된 데이터프레임
            min_strength: 최소 신호 강도 (여러 패턴이 동시에 발생할 경우)

        Returns:
            매매 신호가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

            # 패턴 컬럼이 없으면 패턴 감지 실행
        pattern_cols = [col for col in df.columns if col.startswith('PATTERN_')]
        if not pattern_cols:
            df = self.detect_all_patterns(df)
            pattern_cols = [col for col in df.columns if col.startswith('PATTERN_')]

        result = df.copy()
        try:
            # 불리시/베어리시 신호 컬럼이 없으면 계산
            if 'BULLISH_PATTERNS' not in result.columns or 'BEARISH_PATTERNS' not in result.columns:
                result['BULLISH_PATTERNS'] = result[pattern_cols].apply(
                    lambda x: sum(1 for val in x if val == 100), axis=1
                )
                result['BEARISH_PATTERNS'] = result[pattern_cols].apply(
                    lambda x: sum(1 for val in x if val == -100), axis=1
                )
                result['PATTERN_STRENGTH'] = result['BULLISH_PATTERNS'] - result['BEARISH_PATTERNS']

                # 매수 신호 (불리시 패턴이 min_strength 이상)
            result['PATTERN_BUY_SIGNAL'] = (result['BULLISH_PATTERNS'] >= min_strength).astype(int)

            # 매도 신호 (베어리시 패턴이 min_strength 이상)
            result['PATTERN_SELL_SIGNAL'] = (result['BEARISH_PATTERNS'] >= min_strength).astype(int)

            # 추세 확인 (3일 이동평균 방향)
            if 'MA_3' not in result.columns:
                result['MA_3'] = talib.SMA(result['close'], timeperiod=3)

                # 추세 방향 (1: 상승, -1: 하락, 0: 중립)
            result['TREND_DIRECTION'] = np.sign(result['MA_3'].diff())

            # 추세를 고려한 신호 (추세와 일치하는 패턴만 신호로 인정)
            result['CONFIRMED_BUY_SIGNAL'] = ((result['PATTERN_BUY_SIGNAL'] == 1) &
                                              (result['TREND_DIRECTION'] >= 0)).astype(int)

            result['CONFIRMED_SELL_SIGNAL'] = ((result['PATTERN_SELL_SIGNAL'] == 1) &
                                               (result['TREND_DIRECTION'] <= 0)).astype(int)

            self.logger.info(f"패턴 기반 매매 신호 생성 완료 (최소 강도: {min_strength})")
        except Exception as e:
            self.logger.error(f"매매 신호 생성 중 오류 발생: {e}")

        return result

    def get_pattern_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """감지된 패턴에 대한 설명 생성

        Args:
            df: 패턴이 감지된 데이터프레임

        Returns:
            설명이 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

            # 패턴 컬럼이 없으면 패턴 감지 실행
        pattern_cols = [col for col in df.columns if col.startswith('PATTERN_')]
        if not pattern_cols:
            df = self.detect_all_patterns(df)
            pattern_cols = [col for col in df.columns if col.startswith('PATTERN_')]

        result = df.copy()

        try:
            # 각 행에 대해 감지된 패턴 설명 생성
            def get_row_patterns(row):
                active_patterns = []
                for col in pattern_cols:
                    # 패턴 이름 추출 (PATTERN_ 프리픽스 제거)
                    pattern_name = col[8:]
                    if pattern_name in self.pattern_functions:
                        # 패턴 유형 (불리시/베어리시) 확인
                        if row[col] == 100:  # 불리시 패턴
                            active_patterns.append(f"🔼 {self.pattern_functions[pattern_name]['desc']}")
                        elif row[col] == -100:  # 베어리시 패턴
                            active_patterns.append(f"🔽 {self.pattern_functions[pattern_name]['desc']}")

                if not active_patterns:
                    return "감지된 패턴 없음"

                return " | ".join(active_patterns)

                # 설명 컬럼 추가

            result['PATTERN_DESCRIPTIONS'] = result.apply(get_row_patterns, axis=1)

            self.logger.info("패턴 설명 생성 완료")
        except Exception as e:
            self.logger.error(f"패턴 설명 생성 중 오류 발생: {e}")
            result['PATTERN_DESCRIPTIONS'] = "설명 생성 실패"

        return result

    def find_chart_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """차트 패턴 감지 (삼각형, 쐐기형, 헤드앤숄더 등)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            차트 패턴이 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # 이동평균이 없으면 추가
            if 'MA_10' not in result.columns:
                result['MA_10'] = talib.SMA(result['close'], timeperiod=10)
            if 'MA_20' not in result.columns:
                result['MA_20'] = talib.SMA(result['close'], timeperiod=20)

                # --- 추세선 패턴 감지 ---
            # 이중 바닥 (Double Bottom) 패턴
            result['DOUBLE_BOTTOM'] = self._detect_double_bottom(result)

            # 이중 정점 (Double Top) 패턴
            result['DOUBLE_TOP'] = self._detect_double_top(result)

            # 지지선/저항선 돌파 감지
            result = self._detect_support_resistance_breakout(result)

            # 골든 크로스 / 데드 크로스
            result['GOLDEN_CROSS'] = ((result['MA_10'] > result['MA_20']) &
                                      (result['MA_10'].shift(1) <= result['MA_20'].shift(1))).astype(int)
            result['DEATH_CROSS'] = ((result['MA_10'] < result['MA_20']) &
                                     (result['MA_10'].shift(1) >= result['MA_20'].shift(1))).astype(int)

            self.logger.info("차트 패턴 감지 완료")
        except Exception as e:
            self.logger.error(f"차트 패턴 감지 중 오류 발생: {e}")

        return result

    def _detect_double_bottom(self, df: pd.DataFrame, window: int = 20, threshold_pct: float = 0.03) -> pd.Series:
        """이중 바닥 패턴 감지

        Args:
            df: OHLCV 데이터프레임
            window: 분석 기간
            threshold_pct: 두 바닥 간 최대 허용 가격 차이 (%)

        Returns:
            이중 바닥 패턴 감지 시리즈 (감지: 1, 미감지: 0)
        """
        result = pd.Series(0, index=df.index)

        # 각 행에서 이전 window 기간 동안의 데이터로 이중 바닥 패턴 검사
        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i]

            # 지역 최저점 찾기 (이전 가격보다 낮고, 다음 가격보다 낮은 지점)
            lows = window_data[(window_data['low'] < window_data['low'].shift(1)) &
                               (window_data['low'] < window_data['low'].shift(-1))]['low']

            # 최소 2개의 지역 최저점이 필요
            if len(lows) >= 2:
                # 가장 낮은 두 지점 선택
                two_lowest = lows.nsmallest(2)

                # 두 최저점의 가격 차이 계산 (%)
                price_diff_pct = abs(two_lowest.iloc[0] - two_lowest.iloc[1]) / two_lowest.iloc[0]

                # 두 최저점 사이 거리 계산 (최소 5봉 간격)
                idx_diff = abs(two_lowest.index[0] - two_lowest.index[1])

                # 이중 바닥 조건: 두 최저점의 가격이 유사하고, 일정 거리 이상 떨어져 있음
                if price_diff_pct <= threshold_pct and idx_diff >= 5:
                    # 현재 가격이 두 최저점 사이의 최고점 위에 있으면 이중 바닥 확정
                    middle_high = window_data.loc[two_lowest.index[0]:two_lowest.index[1]]['high'].max()
                    if df.iloc[i]['close'] > middle_high:
                        result.iloc[i] = 1

        return result

    def _detect_double_top(self, df: pd.DataFrame, window: int = 20, threshold_pct: float = 0.03) -> pd.Series:
        """이중 정점 패턴 감지

        Args:
            df: OHLCV 데이터프레임
            window: 분석 기간
            threshold_pct: 두 정점 간 최대 허용 가격 차이 (%)

        Returns:
            이중 정점 패턴 감지 시리즈 (감지: 1, 미감지: 0)
        """
        result = pd.Series(0, index=df.index)

        # 각 행에서 이전 window 기간 동안의 데이터로 이중 정점 패턴 검사
        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i]

            # 지역 최고점 찾기 (이전 가격보다 높고, 다음 가격보다 높은 지점)
            highs = window_data[(window_data['high'] > window_data['high'].shift(1)) &
                                (window_data['high'] > window_data['high'].shift(-1))]['high']

            # 최소 2개의 지역 최고점이 필요
            if len(highs) >= 2:
                # 가장 높은 두 지점 선택
                two_highest = highs.nlargest(2)

                # 두 최고점의 가격 차이 계산 (%)
                price_diff_pct = abs(two_highest.iloc[0] - two_highest.iloc[1]) / two_highest.iloc[0]

                # 두 최고점 사이 거리 계산 (최소 5봉 간격)
                idx_diff = abs(two_highest.index[0] - two_highest.index[1])

                # 이중 정점 조건: 두 최고점의 가격이 유사하고, 일정 거리 이상 떨어져 있음
                if price_diff_pct <= threshold_pct and idx_diff >= 5:
                    # 현재 가격이 두 최고점 사이의 최저점 아래에 있으면 이중 정점 확정
                    middle_low = window_data.loc[two_highest.index[0]:two_highest.index[1]]['low'].min()
                    if df.iloc[i]['close'] < middle_low:
                        result.iloc[i] = 1

        return result

    def _detect_support_resistance_breakout(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """지지선/저항선 돌파 감지

        Args:
            df: OHLCV 데이터프레임
            window: 지지/저항선 판단 기간

        Returns:
            돌파 신호가 추가된 데이터프레임
        """
        result = df.copy()

        # 지지선/저항선 돌파 컬럼 초기화
        result['SUPPORT_BREAKOUT'] = 0
        result['RESISTANCE_BREAKOUT'] = 0

        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i - 1]  # 현재 봉 제외

            # 지지선 (이전 기간의 최저가 중 상위 3개 평균)
            support_levels = window_data['low'].nsmallest(3).mean()

            # 저항선 (이전 기간의 최고가 중 상위 3개 평균)
            resistance_levels = window_data['high'].nlargest(3).mean()

            # 현재 봉
            current_candle = df.iloc[i]

            # 지지선 돌파 (현재 종가가 지지선 아래로 내려감)
            if current_candle['close'] < support_levels and current_candle['open'] > support_levels:
                result.loc[df.index[i], 'SUPPORT_BREAKOUT'] = -1  # 하락 돌파

            # 저항선 돌파 (현재 종가가 저항선 위로 올라감)
            if current_candle['close'] > resistance_levels and current_candle['open'] < resistance_levels:
                result.loc[df.index[i], 'RESISTANCE_BREAKOUT'] = 1  # 상승 돌파

        return result

    def detect_chart_formations(self, df: pd.DataFrame,
                                min_points: int = 5,
                                max_angle: float = 45.0) -> pd.DataFrame:
        """추세선 및 차트 형성 패턴 감지

        Args:
            df: OHLCV 데이터프레임
            min_points: 추세선 형성을 위한 최소 접점 수
            max_angle: 추세선 최대 각도 (도)

        Returns:
            추세선 정보가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()

        try:
            # 각 봉에서의 고점, 저점 찾기
            result['is_pivot_high'] = ((result['high'] > result['high'].shift(1)) &
                                       (result['high'] > result['high'].shift(2)) &
                                       (result['high'] > result['high'].shift(-1)) &
                                       (result['high'] > result['high'].shift(-2))).astype(int)

            result['is_pivot_low'] = ((result['low'] < result['low'].shift(1)) &
                                      (result['low'] < result['low'].shift(2)) &
                                      (result['low'] < result['low'].shift(-1)) &
                                      (result['low'] < result['low'].shift(-2))).astype(int)

            # 상승 추세선 찾기
            result['uptrend_line'] = self._detect_trendline(result, is_uptrend=True,
                                                            min_points=min_points, max_angle=max_angle)

            # 하락 추세선 찾기
            result['downtrend_line'] = self._detect_trendline(result, is_uptrend=False,
                                                              min_points=min_points, max_angle=max_angle)

            # 횡보 구간 (레인지) 감지
            result['range_market'] = ((result['uptrend_line'] == 0) &
                                      (result['downtrend_line'] == 0) &
                                      (result['close'].rolling(14).std() / result['close'].rolling(
                                          14).mean() < 0.03)).astype(int)

            self.logger.info("차트 형성 패턴 감지 완료")
        except Exception as e:
            self.logger.error(f"차트 형성 패턴 감지 중 오류 발생: {e}")

        return result

    def _detect_trendline(self, df: pd.DataFrame, is_uptrend: bool = True,
                          min_points: int = 5, max_angle: float = 45.0) -> pd.Series:
        """추세선 감지

        Args:
            df: OHLCV 데이터프레임
            is_uptrend: 상승 추세선 여부 (False: 하락 추세선)
            min_points: 추세선 형성을 위한 최소 접점 수
            max_angle: 추세선 최대 각도 (도)

        Returns:
            추세선 감지 시리즈 (감지: 1, 미감지: 0)
        """
        result = pd.Series(0, index=df.index)
        window = 30  # 분석 기간

        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i]

            # 상승 추세선은 저점을, 하락 추세선은 고점을 연결
            if is_uptrend:
                # 저점 위치 찾기
                pivot_points = window_data[window_data['is_pivot_low'] == 1]
            else:
                # 고점 위치 찾기
                pivot_points = window_data[window_data['is_pivot_high'] == 1]

                # 피봇 포인트가 최소 점 수 이상인 경우에만 추세선 계산
            if len(pivot_points) >= min_points:
                # 시간축을 숫자로 변환 (인덱스 위치)
                x = np.array(range(len(pivot_points)))

                if is_uptrend:
                    y = pivot_points['low'].values
                else:
                    y = pivot_points['high'].values

                    # 선형 회귀로 추세선 기울기 계산
                if len(x) > 1:  # 최소 2개 점 필요
                    slope, intercept = np.polyfit(x, y, 1)

                    # 기울기의 각도 계산 (도 단위)
                    angle = abs(np.degrees(np.arctan(slope)))

                    # 기울기 방향과 각도 확인
                    valid_slope = (is_uptrend and slope > 0) or (not is_uptrend and slope < 0)

                    if valid_slope and angle < max_angle:
                        result.iloc[i] = 1

        return result

    def detect_advanced_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 차트 패턴 감지 (헤드앤숄더, 플래그, 페넌트 등)

        Args:
            df: OHLCV 데이터프레임

        Returns:
            패턴이 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()

        try:
            # 이동평균 추가
            if 'MA_20' not in result.columns:
                result['MA_20'] = talib.SMA(result['close'], timeperiod=20)
            if 'MA_50' not in result.columns:
                result['MA_50'] = talib.SMA(result['close'], timeperiod=50)

                # 헤드앤숄더 패턴 감지
            result['HEAD_AND_SHOULDERS'] = self._detect_head_and_shoulders(result)

            # 역헤드앤숄더 패턴 감지
            result['INVERSE_HEAD_AND_SHOULDERS'] = self._detect_inverse_head_and_shoulders(result)

            # 삼각형 패턴 감지 (수렴형)
            result['TRIANGLE_PATTERN'] = self._detect_triangle_pattern(result)

            self.logger.info("고급 차트 패턴 감지 완료")
        except Exception as e:
            self.logger.error(f"고급 차트 패턴 감지 중 오류 발생: {e}")

        return result

    def _detect_head_and_shoulders(self, df: pd.DataFrame, window: int = 40) -> pd.Series:
        """헤드앤숄더 패턴 감지 (상승추세 후 반전 신호)"""
        result = pd.Series(0, index=df.index)

        # 최소한 window 크기 이상의 데이터가 필요
        if len(df) < window:
            return result

        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i]

            # 지역 최고점 찾기
            peak_indices = window_data[(window_data['high'] > window_data['high'].shift(1)) &
                                       (window_data['high'] > window_data['high'].shift(-1))].index

            # 최소 3개의 피크가 필요
            if len(peak_indices) >= 3:
                # 가장 높은 3개의 피크 선택
                peaks = window_data.loc[peak_indices]['high'].nlargest(3)
                peak_indices = peaks.index

                # 피크들이 시간순으로 있고, 중간 피크가 가장 높아야 함
                if len(peak_indices) == 3:
                    left, middle, right = peak_indices

                    if left < middle < right:
                        left_peak = window_data.loc[left]['high']
                        middle_peak = window_data.loc[middle]['high']
                        right_peak = window_data.loc[right]['high']

                        # 중간 피크(헤드)가 가장 높고, 좌우 피크(숄더)가 비슷해야 함
                        if middle_peak > left_peak and middle_peak > right_peak:
                            # 좌우 피크 높이 차이가 20% 이내
                            shoulder_diff = abs(left_peak - right_peak) / left_peak
                            if shoulder_diff < 0.2:
                                # 목선(neckline) 확인 - 두 숄더 사이의 저점들의 연결선
                                left_trough_idx = window_data.loc[left:middle]['low'].idxmin()
                                right_trough_idx = window_data.loc[middle:right]['low'].idxmin()

                                if left_trough_idx < right_trough_idx:
                                    left_trough = window_data.loc[left_trough_idx]['low']
                                    right_trough = window_data.loc[right_trough_idx]['low']

                                    # 목선 기울기 - Timedelta를 일수로 변환
                                    try:
                                        # datetime 인덱스인 경우
                                        if pd.api.types.is_datetime64_any_dtype([left_trough_idx, right_trough_idx]):
                                            delta = right_trough_idx - left_trough_idx
                                            days_diff = delta.total_seconds() / (24 * 3600)  # 초를 일로 변환
                                        else:
                                            # 정수 인덱스인 경우
                                            days_diff = right_trough_idx - left_trough_idx

                                        if days_diff > 0:
                                            slope = (right_trough - left_trough) / days_diff

                                            # 현재 가격이 목선 아래로 떨어졌는지 확인 (패턴 완성)
                                            try:
                                                # datetime 인덱스인 경우
                                                if pd.api.types.is_datetime64_any_dtype(
                                                        [df.index[i], right_trough_idx]):
                                                    delta = df.index[i] - right_trough_idx
                                                    days_since_right = delta.total_seconds() / (24 * 3600)
                                                else:
                                                    # 정수 인덱스인 경우
                                                    days_since_right = df.index[i] - right_trough_idx

                                                neckline = right_trough + slope * days_since_right

                                                if df.iloc[i]['close'] < neckline:
                                                    result.iloc[i] = 1
                                            except Exception as e:
                                                self.logger.debug(f"날짜 차이 계산 오류: {e}")
                                    except Exception as e:
                                        self.logger.debug(f"목선 기울기 계산 오류: {e}")

        return result

    def _detect_inverse_head_and_shoulders(self, df: pd.DataFrame, window: int = 40) -> pd.Series:
        """역헤드앤숄더 패턴 감지 (하락추세 후 반전 신호)"""
        result = pd.Series(0, index=df.index)

        # 최소한 window 크기 이상의 데이터가 필요
        if len(df) < window:
            return result

        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i]

            # 지역 최저점 찾기
            trough_indices = window_data[(window_data['low'] < window_data['low'].shift(1)) &
                                         (window_data['low'] < window_data['low'].shift(-1))].index

            # 최소 3개의 저점이 필요
            if len(trough_indices) >= 3:
                # 가장 낮은 3개의 저점 선택
                troughs = window_data.loc[trough_indices]['low'].nsmallest(3)
                trough_indices = troughs.index

                # 저점들이 시간순으로 있고, 중간 저점이 가장 낮아야 함
                if len(trough_indices) == 3:
                    left, middle, right = trough_indices

                    if left < middle < right:
                        left_trough = window_data.loc[left]['low']
                        middle_trough = window_data.loc[middle]['low']
                        right_trough = window_data.loc[right]['low']

                        # 중간 저점(헤드)이 가장 낮고, 좌우 저점(숄더)이 비슷해야 함
                        if middle_trough < left_trough and middle_trough < right_trough:
                            # 좌우 저점 높이 차이가 20% 이내
                            shoulder_diff = abs(left_trough - right_trough) / left_trough
                            if shoulder_diff < 0.2:
                                # 목선(neckline) 확인 - 두 숄더 사이의 고점들의 연결선
                                left_peak_idx = window_data.loc[left:middle]['high'].idxmax()
                                right_peak_idx = window_data.loc[middle:right]['high'].idxmax()

                                if left_peak_idx < right_peak_idx:
                                    left_peak = window_data.loc[left_peak_idx]['high']
                                    right_peak = window_data.loc[right_peak_idx]['high']

                                    # 목선 기울기 - Timedelta를 일수로 변환
                                    try:
                                        # datetime 인덱스인 경우
                                        if pd.api.types.is_datetime64_any_dtype([left_peak_idx, right_peak_idx]):
                                            delta = right_peak_idx - left_peak_idx
                                            days_diff = delta.total_seconds() / (24 * 3600)  # 초를 일로 변환
                                        else:
                                            # 정수 인덱스인 경우
                                            days_diff = right_peak_idx - left_peak_idx

                                        if days_diff > 0:
                                            slope = (right_peak - left_peak) / days_diff

                                            # 현재 가격이 목선 위로 올라갔는지 확인 (패턴 완성)
                                            try:
                                                # datetime 인덱스인 경우
                                                if pd.api.types.is_datetime64_any_dtype([df.index[i], right_peak_idx]):
                                                    delta = df.index[i] - right_peak_idx
                                                    days_since_right = delta.total_seconds() / (24 * 3600)
                                                else:
                                                    # 정수 인덱스인 경우
                                                    days_since_right = df.index[i] - right_peak_idx

                                                neckline = right_peak + slope * days_since_right

                                                if df.iloc[i]['close'] > neckline:
                                                    result.iloc[i] = 1
                                            except Exception as e:
                                                self.logger.debug(f"날짜 차이 계산 오류: {e}")
                                    except Exception as e:
                                        self.logger.debug(f"목선 기울기 계산 오류: {e}")

        return result

    def _detect_triangle_pattern(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """삼각형 패턴 감지 (수렴하는 고점과 저점)

        Args:
            df: OHLCV 데이터프레임
            window: 분석 기간

        Returns:
            삼각형 패턴 감지 시리즈 (대칭삼각형: 1, 상승삼각형: 2, 하락삼각형: 3, 미감지: 0)
        """
        result = pd.Series(0, index=df.index)

        for i in range(window, len(df)):
            # 분석 구간
            window_data = df.iloc[i - window:i]

            # 고점과 저점 찾기
            highs = window_data[(window_data['high'] > window_data['high'].shift(1)) &
                                (window_data['high'] > window_data['high'].shift(-1))]['high']

            lows = window_data[(window_data['low'] < window_data['low'].shift(1)) &
                               (window_data['low'] < window_data['low'].shift(-1))]['low']

            # 최소 2개의 고점과 저점이 필요
            if len(highs) >= 2 and len(lows) >= 2:
                # 고점과 저점의 인덱스와 값을 추출
                high_indices = highs.index
                high_values = highs.values

                low_indices = lows.index
                low_values = lows.values

                # 최소 3개 이상의 접점으로 추세선 계산
                if len(high_indices) >= 3 and len(low_indices) >= 3:
                    # 고점 추세선 (x는 시간축을 숫자로 변환)
                    x_high = np.array(range(len(high_indices)))
                    slope_high, intercept_high = np.polyfit(x_high, high_values, 1)

                    # 저점 추세선
                    x_low = np.array(range(len(low_indices)))
                    slope_low, intercept_low = np.polyfit(x_low, low_values, 1)

                    # 삼각형 패턴의 세 가지 유형 판별
                    # 대칭삼각형: 고점은 하락, 저점은 상승 추세
                    if slope_high < -0.001 and slope_low > 0.001:
                        result.iloc[i] = 1
                        # 상승삼각형: 고점은 수평, 저점은 상승 추세
                    elif abs(slope_high) < 0.001 and slope_low > 0.001:
                        result.iloc[i] = 2
                        # 하락삼각형: 고점은 하락, 저점은 수평 추세
                    elif slope_high < -0.001 and abs(slope_low) < 0.001:
                        result.iloc[i] = 3

        return result

    def detect_volatility_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """변동성 기반 패턴 감지

        Args:
            df: OHLCV 데이터프레임

        Returns:
            변동성 패턴이 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()

        try:
            # 일일 변동성 계산 (고가-저가)/시가
            result['daily_volatility'] = (result['high'] - result['low']) / result['open'] * 100

            # 평균 변동성 (20일)
            result['avg_volatility_20'] = result['daily_volatility'].rolling(20).mean()

            # 변동성 급증 감지
            result['volatility_surge'] = (result['daily_volatility'] > 2 * result['avg_volatility_20']).astype(int)

            # 변동성 수축 감지
            result['volatility_squeeze'] = (result['daily_volatility'] < 0.5 * result['avg_volatility_20']).astype(
                int)

            # 볼린저 밴드 폭 계산
            if 'BB_upper_20' not in result.columns or 'BB_lower_20' not in result.columns:
                upper, middle, lower = talib.BBANDS(
                    result['close'],
                    timeperiod=20,
                    nbdevup=2,
                    nbdevdn=2
                )
                result['BB_upper_20'] = upper
                result['BB_middle_20'] = middle
                result['BB_lower_20'] = lower

                # 볼린저 밴드 폭
            result['BB_width'] = (result['BB_upper_20'] - result['BB_lower_20']) / result['BB_middle_20']

            # 볼린저 밴드 수축 감지
            result['BB_squeeze'] = (result['BB_width'] < result['BB_width'].rolling(20).quantile(0.2)).astype(int)

            # 볼린저 밴드 돌파 감지
            result['BB_breakout_up'] = ((result['close'] > result['BB_upper_20']) &
                                        (result['close'].shift(1) <= result['BB_upper_20'].shift(1))).astype(int)

            result['BB_breakout_down'] = ((result['close'] < result['BB_lower_20']) &
                                          (result['close'].shift(1) >= result['BB_lower_20'].shift(1))).astype(int)

            self.logger.info("변동성 기반 패턴 감지 완료")
        except Exception as e:
            self.logger.error(f"변동성 패턴 감지 중 오류 발생: {e}")

        return result

    def get_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 패턴 감지 및 종합 신호 생성

        Args:
            df: OHLCV 데이터프레임

        Returns:
            종합 신호가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        try:
            # 1. 캔들스틱 패턴 감지
            result = self.detect_all_patterns(df)

            # 2. 차트 패턴 감지
            result = self.find_chart_patterns(result)

            # 3. 고급 차트 패턴 감지
            result = self.detect_advanced_patterns(result)

            # 4. 변동성 패턴 감지
            result = self.detect_volatility_patterns(result)

            # 5. 패턴 설명 추가
            result = self.get_pattern_descriptions(result)

            # 6. 매매 신호 추가
            result = self.get_pattern_signals(result)

            # 7. 종합 매수/매도 신호 생성
            bull_columns = ['CONFIRMED_BUY_SIGNAL', 'DOUBLE_BOTTOM', 'INVERSE_HEAD_AND_SHOULDERS',
                            'GOLDEN_CROSS', 'RESISTANCE_BREAKOUT']

            bear_columns = ['CONFIRMED_SELL_SIGNAL', 'DOUBLE_TOP', 'HEAD_AND_SHOULDERS',
                            'DEATH_CROSS', 'SUPPORT_BREAKOUT']

            # 존재하는 컬럼만 사용
            bull_cols = [col for col in bull_columns if col in result.columns]
            bear_cols = [col for col in bear_columns if col in result.columns]

            # 불리시 신호 점수 계산 (각 신호에 가중치 부여 가능)
            if bull_cols:
                result['BULLISH_SCORE'] = result[bull_cols].sum(axis=1)
            else:
                result['BULLISH_SCORE'] = 0

                # 베어리시 신호 점수 계산
            if bear_cols:
                result['BEARISH_SCORE'] = result[bear_cols].sum(axis=1)
            else:
                result['BEARISH_SCORE'] = 0

                # 종합 신호 강도 (-5 ~ +5 사이)
            result['SIGNAL_STRENGTH'] = result['BULLISH_SCORE'] - result['BEARISH_SCORE']

            # 신호 해석 (강한 매수, 약한 매수, 중립, 약한 매도, 강한 매도)
            def interpret_signal(strength):
                if strength >= 3:
                    return "강한 매수 신호"
                elif strength >= 1:
                    return "약한 매수 신호"
                elif strength <= -3:
                    return "강한 매도 신호"
                elif strength <= -1:
                    return "약한 매도 신호"
                else:
                    return "중립 신호"

            result['SIGNAL_INTERPRETATION'] = result['SIGNAL_STRENGTH'].apply(interpret_signal)

            self.logger.info("모든 패턴 감지 및 종합 신호 생성 완료")
        except Exception as e:
            self.logger.error(f"종합 신호 생성 중 오류 발생: {e}")
            return df

        return result


# 모듈 테스트를 위한 코드
if __name__ == "__main__":
    from src.data_collection.collectors import DataCollector
    from src.data_processing.indicators import TechnicalIndicators
    import matplotlib.pyplot as plt

    # 데이터 수집
    collector = DataCollector()
    df = collector.get_historical_data("BTCUSDT", "4h", "3 months ago UTC")

    # 기술적 지표 계산
    indicators = TechnicalIndicators()
    df_with_indicators = indicators.add_all_indicators(df)

    # 패턴 감지
    pattern_recognition = PatternRecognition()
    result_df = pattern_recognition.get_all_signals(df_with_indicators)

    # 결과 확인
    print("\n패턴 감지 결과 (마지막 10행):")
    print(result_df[['close', 'PATTERN_STRENGTH', 'SIGNAL_STRENGTH', 'SIGNAL_INTERPRETATION',
                     'PATTERN_DESCRIPTIONS']].tail(10))

    # 특정 날짜의 패턴 상세 확인
    interesting_dates = result_df[result_df['SIGNAL_STRENGTH'].abs() >= 2].index
    if len(interesting_dates) > 0:
        interesting_date = interesting_dates[-1]  # 가장 최근 강한 신호
        print(f"\n흥미로운 패턴 발견 날짜: {interesting_date.strftime('%Y-%m-%d')}")

        # 해당 날짜의 모든 패턴 정보 출력
        pattern_cols = [col for col in result_df.columns if
                        col.startswith('PATTERN_') and col != 'PATTERN_DESCRIPTIONS' and col != 'PATTERN_STRENGTH']
        chart_pattern_cols = ['DOUBLE_BOTTOM', 'DOUBLE_TOP', 'HEAD_AND_SHOULDERS', 'INVERSE_HEAD_AND_SHOULDERS',
                              'TRIANGLE_PATTERN', 'GOLDEN_CROSS', 'DEATH_CROSS', 'SUPPORT_BREAKOUT',
                              'RESISTANCE_BREAKOUT']

        active_patterns = []
        for col in pattern_cols:
            if result_df.loc[interesting_date, col] != 0:
                value = result_df.loc[interesting_date, col]
                pattern_name = col[8:]  # PATTERN_ 프리픽스 제거
                direction = "불리시" if value == 100 else "베어리시" if value == -100 else "중립"
                active_patterns.append(f"{pattern_name}: {direction}")

        for col in chart_pattern_cols:
            if col in result_df.columns and result_df.loc[interesting_date, col] != 0:
                active_patterns.append(f"{col}: {result_df.loc[interesting_date, col]}")

        if active_patterns:
            print("활성화된 패턴:")
            for pattern in active_patterns:
                print(f"- {pattern}")
        else:
            print("활성화된 패턴이 없습니다.")

        print(f"패턴 설명: {result_df.loc[interesting_date, 'PATTERN_DESCRIPTIONS']}")
        print(f"신호 강도: {result_df.loc[interesting_date, 'SIGNAL_STRENGTH']}")
        print(f"신호 해석: {result_df.loc[interesting_date, 'SIGNAL_INTERPRETATION']}")

    # 차트 시각화
    def plot_patterns(df, last_n_days=30):
        """패턴 감지 결과 시각화"""
        plt.figure(figsize=(14, 10))

        # 데이터 준비
        plot_df = df.iloc[-last_n_days:].copy() if len(df) > last_n_days else df.copy()

        # 메인 차트 (캔들스틱)
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

        # 캔들스틱 그리기
        for i in range(len(plot_df)):
            open_price = plot_df['open'].iloc[i]
            close = plot_df['close'].iloc[i]
            high = plot_df['high'].iloc[i]
            low = plot_df['low'].iloc[i]

            if close >= open_price:  # 양봉
                color = 'g'
                body_height = close - open_price
            else:  # 음봉
                color = 'r'
                body_height = open_price - close

            # 캔들 몸통
            rect = plt.Rectangle((i, min(open_price, close)), 0.8, body_height,
                                 color=color, alpha=0.5)
            ax1.add_patch(rect)

            # 꼬리 (wick)
            ax1.plot([i + 0.4, i + 0.4], [low, min(open_price, close)], color='black')
            ax1.plot([i + 0.4, i + 0.4], [max(open_price, close), high], color='black')

        # 이동평균선 추가
        if 'MA_20' in plot_df.columns:
            ax1.plot(range(len(plot_df)), plot_df['MA_20'], color='blue', label='MA 20')
        if 'MA_50' in plot_df.columns:
            ax1.plot(range(len(plot_df)), plot_df['MA_50'], color='orange', label='MA 50')

        # 볼린저 밴드 추가
        if 'BB_upper_20' in plot_df.columns and 'BB_lower_20' in plot_df.columns:
            ax1.plot(range(len(plot_df)), plot_df['BB_upper_20'], color='purple', linestyle='--', label='BB Upper')
            ax1.plot(range(len(plot_df)), plot_df['BB_lower_20'], color='purple', linestyle='--', label='BB Lower')

        # 패턴 마커 추가
        bull_markers = []
        bear_markers = []

        for i in range(len(plot_df)):
            # 불리시 패턴 마커
            if plot_df['BULLISH_PATTERNS'].iloc[i] > 0 or \
                    (plot_df['SIGNAL_STRENGTH'].iloc[i] >= 2):
                bull_markers.append((i, plot_df['low'].iloc[i] * 0.99))

            # 베어리시 패턴 마커
            if plot_df['BEARISH_PATTERNS'].iloc[i] > 0 or \
                    (plot_df['SIGNAL_STRENGTH'].iloc[i] <= -2):
                bear_markers.append((i, plot_df['high'].iloc[i] * 1.01))

        if bull_markers:
            bull_x, bull_y = zip(*bull_markers)
            ax1.scatter(bull_x, bull_y, s=100, marker='^', color='green', label='Bullish Pattern')

        if bear_markers:
            bear_x, bear_y = zip(*bear_markers)
            ax1.scatter(bear_x, bear_y, s=100, marker='v', color='red', label='Bearish Pattern')

        ax1.set_ylabel('Price')
        ax1.set_title('Price Chart with Detected Patterns')
        ax1.grid(True)
        ax1.legend()

        # 볼륨 차트
        ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=ax1)
        ax2.bar(range(len(plot_df)), plot_df['volume'], color=[
            'g' if plot_df['close'].iloc[i] >= plot_df['open'].iloc[i] else 'r'
            for i in range(len(plot_df))
        ], alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)

        # 신호 강도 차트
        ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
        ax3.bar(range(len(plot_df)), plot_df['SIGNAL_STRENGTH'], color=[
            'g' if plot_df['SIGNAL_STRENGTH'].iloc[i] >= 0 else 'r'
            for i in range(len(plot_df))
        ], alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.axhline(y=2, color='g', linestyle='--', alpha=0.3)
        ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.3)
        ax3.set_ylabel('Signal Strength')
        ax3.set_ylim(-5, 5)
        ax3.grid(True)

        # x축 날짜 설정 - 인덱스 타입에 따라 다르게 처리
        tick_positions = range(0, len(plot_df), max(1, len(plot_df) // 10))
        ax3.set_xticks(tick_positions)

        # 인덱스 타입에 따라 레이블 생성
        if pd.api.types.is_datetime64_any_dtype(plot_df.index):
            # datetime 인덱스인 경우 strftime 사용
            date_labels = [plot_df.index[i].strftime('%m-%d') for i in tick_positions]
        else:
            # 인덱스가 datetime이 아닌 경우(정수 등) 문자열로 변환
            date_labels = [str(plot_df.index[i]) for i in tick_positions if i < len(plot_df.index)]

        ax3.set_xticklabels(date_labels, rotation=45)

        plt.tight_layout()
        plt.show()

    # 마지막 30일간의 패턴 시각화
    if len(result_df) > 0:
        plot_patterns(result_df)