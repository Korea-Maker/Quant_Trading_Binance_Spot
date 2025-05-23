# collectors.py
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
import sys

# 경로 추가 (직접 실행 시 필요)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.data_collection.binance_api import BinanceAPI


class DataCollector:
    """시장 데이터 수집 클래스 (개선된 버전)"""

    def __init__(self, api=None):
        self.logger = get_logger(__name__)
        self.api = api or BinanceAPI()

    def _convert_start_str_to_timestamp(self, start_str):
        """시작 시간 문자열을 타임스탬프로 변환"""
        try:
            if "days ago" in start_str:
                days = int(start_str.split()[0])
                start_time = datetime.now() - timedelta(days=days)
            elif "months ago" in start_str:
                months = int(start_str.split()[0])
                start_time = datetime.now() - timedelta(days=months * 30)
            elif "weeks ago" in start_str:
                weeks = int(start_str.split()[0])
                start_time = datetime.now() - timedelta(weeks=weeks)
            else:
                # 기본적으로 바이낸스 클라이언트가 처리하도록 반환
                return start_str

            return start_time.strftime("%Y-%m-%d")
        except:
            return start_str

    def get_historical_data(self, symbol, interval, start_str, end_str=None, limit=1000):
        """과거 캔들스틱 데이터 수집 (개선된 버전)"""
        try:
            # 날짜 형식 변환
            converted_start = self._convert_start_str_to_timestamp(start_str)

            self.logger.info(f"{symbol} {interval} 과거 데이터 수집 중...")
            self.logger.info(f"요청 기간: {start_str} -> {converted_start} ~ {end_str or 'now'}")
            self.logger.info(f"요청 limit: {limit}")

            # 데이터 수집
            klines = self.api.get_klines(symbol, interval, converted_start, end_str, limit)

            self.logger.info(f"API 응답 데이터 개수: {len(klines) if klines else 0}")

            if not klines:
                self.logger.warning(f"{symbol} {interval} 데이터가 없습니다.")
                return pd.DataFrame()

                # 데이터프레임으로 변환
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_base', 'taker_buy_quote', 'ignored'
            ])

            self.logger.info(f"DataFrame 생성 후 행 수: {len(df)}")

            # 타입 변환
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)

                # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # 중복 제거 (혹시 모를 중복 데이터)
            df = df.drop_duplicates(subset=['timestamp'])

            self.logger.info(f"중복 제거 후 행 수: {len(df)}")

            # 필요 없는 컬럼 제거
            df = df.drop(['ignored'], axis=1)

            # timestamp를 index로 설정
            df = df.set_index('timestamp')

            # 날짜순 정렬
            df = df.sort_index()

            self.logger.info(f"{symbol} {interval} 최종 데이터 {len(df)}개 수집 완료")
            self.logger.info(f"데이터 기간: {df.index[0]} ~ {df.index[-1]}")

            return df

        except Exception as e:
            self.logger.error(f"데이터 수집 중 오류: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()

    def get_multiple_periods_data(self, symbol, interval, total_days=365):
        """긴 기간의 데이터를 여러 번 나누어서 수집"""
        try:
            self.logger.info(f"{symbol} {interval} 장기간 데이터 수집 시작 (총 {total_days}일)")

            all_data = []
            max_days_per_request = 1000  # 1일봉 기준 최대 1000일

            if interval == '1d':
                max_days_per_request = 1000
            elif interval == '1h':
                max_days_per_request = 41  # 1000시간 ≈ 41일
            elif interval == '4h':
                max_days_per_request = 166  # 1000개 ≈ 166일
            elif interval == '15m':
                max_days_per_request = 10  # 1000개 ≈ 10일
            elif interval == '5m':
                max_days_per_request = 3  # 1000개 ≈ 3일
            elif interval == '1m':
                max_days_per_request = 1  # 1000개 ≈ 1일

            current_end = datetime.now()

            while total_days > 0:
                days_to_get = min(total_days, max_days_per_request)
                start_time = current_end - timedelta(days=days_to_get)

                self.logger.info(f"수집 중: {start_time.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")

                # 데이터 수집
                df = self.get_historical_data(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_time.strftime("%Y-%m-%d"),
                    end_str=current_end.strftime("%Y-%m-%d"),
                    limit=1000
                )

                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"수집 완료: {len(df)}개 행")
                else:
                    self.logger.warning(f"데이터 없음: {start_time} ~ {current_end}")

                    # 다음 기간 설정
                current_end = start_time
                total_days -= days_to_get

                # API 제한 방지
                time.sleep(0.1)

            if all_data:
                # 모든 데이터 합치기
                final_df = pd.concat(all_data)
                # 중복 제거 및 정렬
                final_df = final_df[~final_df.index.duplicated(keep='first')]
                final_df = final_df.sort_index()

                self.logger.info(f"최종 데이터 수집 완료: {len(final_df)}개 행")
                self.logger.info(f"전체 기간: {final_df.index[0]} ~ {final_df.index[-1]}")

                return final_df
            else:
                self.logger.error("수집된 데이터가 없습니다.")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"장기간 데이터 수집 중 오류: {e}")
            return pd.DataFrame()


class EnhancedDataCollector(DataCollector):
    """강화된 데이터 수집기"""

    def get_max_available_data(self, symbol, interval, max_days=1000):
        """가능한 최대 데이터 수집"""
        try:
            self.logger.info(f"{symbol} {interval} 최대 데이터 수집 시도")

            # 바이낸스 API 최대 limit (1000) 사용
            klines = self.api.get_klines(
                symbol=symbol,
                interval=interval,
                start_str=f"{max_days} days ago UTC",
                limit=1000
            )

            if not klines:
                self.logger.warning(f"{symbol} 데이터가 없습니다")
                return pd.DataFrame()

                # 데이터프레임 변환
            df = self._convert_klines_to_dataframe(klines)

            self.logger.info(f"{symbol} 최대 데이터 수집 완료: {len(df)}개 행")
            return df

        except Exception as e:
            self.logger.error(f"최대 데이터 수집 실패: {e}")
            return pd.DataFrame()

    def get_data_chunks(self, symbol, interval, total_days, chunk_days=365):
        """청크 단위로 데이터 수집"""
        all_data = []
        current_end = datetime.now()

        while total_days > 0:
            days_to_get = min(total_days, chunk_days)
            start_time = current_end - timedelta(days=days_to_get)

            self.logger.info(f"청크 수집: {start_time.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")

            try:
                klines = self.api.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_time.strftime('%Y-%m-%d'),
                    end_str=current_end.strftime('%Y-%m-%d'),
                    limit=1000
                )

                if klines:
                    df = self._convert_klines_to_dataframe(klines)
                    all_data.append(df)
                    self.logger.info(f"청크 수집 완료: {len(df)}개 행")
                else:
                    self.logger.warning("청크에서 데이터 없음")

            except Exception as e:
                self.logger.error(f"청크 수집 실패: {e}")

            current_end = start_time
            total_days -= days_to_get
            time.sleep(0.2)  # API 제한 방지

        if all_data:
            final_df = pd.concat(all_data)
            final_df = final_df[~final_df.index.duplicated(keep='first')]
            final_df = final_df.sort_index()

            self.logger.info(f"전체 청크 수집 완료: {len(final_df)}개 행")
            return final_df

        return pd.DataFrame()

    def _convert_klines_to_dataframe(self, klines):
        """klines 데이터를 DataFrame으로 변환"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count',
            'taker_buy_base', 'taker_buy_quote', 'ignored'
        ])

        # 타입 변환
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = df[col].astype(float)

            # 타임스탬프 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # 필요 없는 컬럼 제거 및 인덱스 설정
        df = df.drop(['ignored'], axis=1)
        df = df.set_index('timestamp')

        return df