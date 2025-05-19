# src/data_processing/indicators.py

import pandas as pd
import numpy as np
import talib
from src.utils.logger import get_logger
from typing import Union, List, Dict, Optional


class TechnicalIndicators:
    """기술적 지표 계산을 위한 클래스"""

    def __init__(self):
        """초기화"""
        self.logger = get_logger(__name__)

    # 데이터프레임 유효성 검사 메서드
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

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self.logger.error(f"데이터프레임에 참조할 컬럼이 없습니다: {missing}")
            return False

        return True

    def add_moving_averages(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """이동평균선 추가

        Args:
            df: OHLCV 데이터프레임
            periods: MA 기간 목록

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            for period in periods:
                result[f'MA_{period}'] = talib.SMA(result['close'], timeperiod=period)
                result[f'EMA_{period}'] = talib.EMA(result['close'], timeperiod=period)

            self.logger.info(f"이동평균선 지표 계산 완료 (기간: {periods})")
        except Exception as e:
            self.logger.error(f"이동평균선 계산중 오류 발생 : {e}")

        return result

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, stdev_factor: float = 2.0) -> pd.DataFrame:
        """볼린저 밴드 추가

        Args:
            df: OHLCV 데이터프레임
            period: 기간
            stdev_factor: 표준편차 계수

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # 볼린저 밴드 계산
            upper, middle, lower = talib.BBANDS(
                result['close'],
                timeperiod=period,
                nbdevup=stdev_factor,
                nbdevdn=stdev_factor
            )

            result[f'BB_upper_{period}'] = upper
            result[f'BB_middle_{period}'] = middle
            result[f'BB_lower_{period}'] = lower

            # 변동성 추가 (상단-하단)/중간
            result[f'BB_width_{period}'] = (upper - lower) / middle

            self.logger.info(f"볼린저 밴드 지표 계산 완료 (기간: {period}, 계수: {stdev_factor})")
        except Exception as e:
            self.logger.error(f"볼린저 밴드 계산 중 오류 발생 : {e}")

        return result

    def add_rsi(self, df: pd.DataFrame, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """RSI(상대강도지수) 추가

        Args:
            df: OHLCV 데이터프레임
            periods: RSI 기간 목록

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            for period in periods:
                result[f'RSI_{period}'] = talib.RSI(result['close'], timeperiod=period)
            self.logger.info(f"RSI 지표 계산 완료 (기간: {periods})")
        except Exception as e:
            self.logger.error(f"RSI 지표 계산 중 오류 발생 : {e}")

        return result

    def add_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9) -> pd.DataFrame:
        """MACD(이동평균 수렴확산) 추가

        Args:
            df: OHLCV 데이터프레임
            fast_period: 빠른 EMA 기간
            slow_period: 느린 EMA 기간
            signal_period: 시그널 EMA 기간

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # MACD, MACD Signal, MACD Histogram 계산
            macd, signal, hist = talib.MACD(
                result['close'],
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )

            result['MACD'] = macd
            result['MACD_signal'] = signal
            result['MACD_hist'] = hist

            self.logger.info(f"MACD 지표 계산 완료 (빠른기간: {fast_period}, 느린기간: {slow_period}, 시그널기간: {signal_period})")
        except Exception as e:
            self.logger.error(f"MACD 지표 계산 중 오류 발생 : {e}")

        return result

    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
        """스토캐스틱 오실레이터 추가

        Args:
            df: OHLCV 데이터프레임
            k_period: %K 기간
            d_period: %D 기간
            slowing: 슬로잉 기간

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # 스토캐스틱 %K, %D 계산
            k, d = talib.STOCH(
                result['high'],
                result['low'],
                result['close'],
                fastk_period=k_period,
                slowk_period=slowing,
                slowk_matype=0,
                slowd_period=d_period,
                slowd_matype=0
            )

            result['STOCH_K'] = k
            result['STOCH_D'] = d

            self.logger.info(f"스토캐스틱 지표 계산 완료 (K기간: {k_period}, D기간: {d_period}, 슬로잉: {slowing})")
        except Exception as e:
            self.logger.error(f"스토캐스틱 지표 계산 중 오류 발생 : {e}")

        return result

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ATR(평균진폭) 추가

        Args:
            df: OHLCV 데이터프레임
            period: ATR 기간

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            result[f'ATR_{period}'] = talib.ATR(
                result['high'],
                result['low'],
                result['close'],
                timeperiod=period
            )

            # ATR 퍼센트 추가 (ATR/종가)
            result[f'ATR_percent_{period}'] = result[f'ATR_{period}'] / result['close'] * 100

            self.logger.info(f"ATR 지표 계산 완료 (기간: {period})")
        except Exception as e:
            self.logger.error(f"ATR 지표 계산 중 오류 발생 : {e}")

        return result

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ADX(평균방향지수) 추가

        Args:
            df: OHLCV 데이터프레임
            period: ADX 기간

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            result[f'ADX_{period}'] = talib.ADX(
                result['high'],
                result['low'],
                result['close'],
                timeperiod=period
            )

            # 방향성 지표 추가 (DMI)
            result[f'PLUS_DI_{period}'] = talib.PLUS_DI(
                result['high'],
                result['low'],
                result['close'],
                timeperiod=period
            )

            result[f'MINUS_DI_{period}'] = talib.MINUS_DI(
                result['high'],
                result['low'],
                result['close'],
                timeperiod=period
            )

            self.logger.info(f"ADX 지표 계산 완료 (기간: {period})")
        except Exception as e:
            self.logger.error(f"ADX 지표 계산 중 오류 발생 : {e}")

        return result

    def add_ichimoku(self, df: pd.DataFrame,
                     tenkan_period: int = 9,
                     kijun_period: int = 26,
                     senkou_b_period: int = 52) -> pd.DataFrame:
        """일목균형표 추가

        Args:
            df: OHLCV 데이터프레임
            tenkan_period: 전환선 기간
            kijun_period: 기준선 기간
            senkou_b_period: 선행스팬 B 기간

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            # 전환선 (Conversion Line)
            tenkan_high = result['high'].rolling(window=tenkan_period).max()
            tenkan_low = result['low'].rolling(window=tenkan_period).min()
            result['ichimoku_tenkan'] = (tenkan_high + tenkan_low) / 2

            # 기준선 (Base Line)
            kijun_high = result['high'].rolling(window=kijun_period).max()
            kijun_low = result['low'].rolling(window=kijun_period).min()
            result['ichimoku_kijun'] = (kijun_high + kijun_low) / 2

            # 선행스팬 A (Leading Span A)
            result['ichimoku_senkou_a'] = ((result['ichimoku_tenkan'] + result['ichimoku_kijun']) / 2).shift(kijun_period)

            # 선행스팬 B (Leading Span B)
            senkou_b_high = result['high'].rolling(window=senkou_b_period).max()
            senkou_b_low = result['low'].rolling(window=senkou_b_period).min()
            result['ichimoku_senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)

            # 후행스팬 (Lagging Span)
            result['ichimoku_chikou'] = result['close'].shift(-kijun_period)

            self.logger.info(f"일목균형표 지표 계산 완료")
        except Exception as e:
            self.logger.error(f"일목균형표 지표 계산 중 오류 발생 : {e}")

        return result

    def add_volume_indicators(self, df: pd.DataFrame, periods: List[int] = [20]) -> pd.DataFrame:
        """거래량 기반 지표 추가

        Args:
            df: OHLCV 데이터프레임
            periods: 기간 목록

        Returns:
            지표가 추가된 데이터프레임
        """
        if not self._validate_dataframe(df):
            return df

        result = df.copy()
        try:
            for period in periods:
                # 거래량 이동평균
                result[f'volume_MA_{period}'] = talib.SMA(result['volume'], timeperiod=period)

                # OBV (On-Balance Volume)
                result['OBV'] = talib.OBV(result['close'], result['volume'])

                # 거래량 변화율
                result['volume_change'] = result['volume'].pct_change() * 100

                # 상대적 거래량 (현재거래량/평균거래량)
                result[f'rel_volume_{period}'] = result['volume'] / result[f'volume_MA_{period}']

            self.logger.info(f"거래량 지표 계산 완료 (기간: {periods})")
        except Exception as e:
            self.logger.error(f"거래량 지표 계산 중 오류 발생 : {e}")

        return result

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 기본 기술적 지표를 한 번에 추가

        Args:
            df: OHLCV 데이터프레임

        Returns:
            지표가 추가된 데이터프레임
        """
        result = df.copy()

        # 이동평균선
        result = self.add_moving_averages(result)

        # 볼린저 밴드
        result = self.add_bollinger_bands(result)

        # RSI
        result = self.add_rsi(result)

        # MACD
        result = self.add_macd(result)

        # 스토캐스틱
        result = self.add_stochastic(result)

        # ATR
        result = self.add_atr(result)

        # 거래량 지표
        result = self.add_volume_indicators(result)

        self.logger.info("모든 기본 기술적 지표 계산 완료")
        return result


# 모듈 테스트를 위한 코드
if __name__ == "__main__":
    from src.data_collection.collectors import DataCollector

    # 데이터 수집
    collector = DataCollector()
    df = collector.get_historical_data("BTCUSDT", "1h", "1 month ago UTC")

    # 지표 계산
    indicators = TechnicalIndicators()
    result_df = indicators.add_all_indicators(df)

    # 결과 확인
    print("\n지표가 추가된 데이터프레임 (마지막 5행):")
    print(result_df.tail(5).T)  # 전치해서 열로 표시
    print(f"\n전체 컬럼 수: {len(result_df.columns)}")
    print(
        f"지표 컬럼 목록: {[col for col in result_df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades_count', 'taker_buy_base', 'taker_buy_quote']]}")