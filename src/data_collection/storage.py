# storage.py

import os
import pandas as pd
import json
from datetime import datetime
import sys

# 경로 추가 (직접 실행 시 필요)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir_exists
from src.data_collection.collectors import DataCollector

# 기본 데이터 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


class DataStorage:
    """데이터 저장 및 관리 클래스"""

    def __init__(self, raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR):
        """
        데이터 저장소 초기화

        Args:
            raw_dir (str): 원시 데이터 저장 경로
            processed_dir (str): 처리된 데이터 저장 경로
        """
        self.logger = get_logger(__name__)
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir

        # 디렉토리 생성
        ensure_dir_exists(self.raw_dir)
        ensure_dir_exists(self.processed_dir)

    def get_symbol_dir(self, symbol, data_type="klines", is_raw=True):
        """심볼별 데이터 디렉토리 경로 반환

        Args:
            symbol (str): 거래 쌍
            data_type (str): 데이터 유형 (klines, orderbook 등)
            is_raw (bool): 원시 데이터 여부

        Returns:
            str: 디렉토리 경로
        """
        base_dir = self.raw_dir if is_raw else self.processed_dir
        symbol_dir = os.path.join(base_dir, symbol, data_type)
        ensure_dir_exists(symbol_dir)
        return symbol_dir

    def save_dataframe(self, df, symbol, interval=None, data_type="klines", is_raw=True):
        """데이터프레임을 CSV 파일로 저장

        Args:
            df (DataFrame): 저장할 데이터프레임
            symbol (str): 거래 쌍
            interval (str, optional): 시간 간격 (캔들스틱 데이터인 경우)
            data_type (str): 데이터 유형
            is_raw (bool): 원시 데이터 여부

        Returns:
            str: 저장된 파일 경로
        """
        if df.empty:
            self.logger.warning(f"{symbol} 저장할 데이터가 없습니다.")
            return None

        # 디렉토리 경로 생성
        symbol_dir = self.get_symbol_dir(symbol, data_type, is_raw)

        # 파일명 생성
        today = datetime.now().strftime('%Y-%m-%d')

        if interval:
            filename = f"{symbol}_{interval}_{today}.csv"
        else:
            filename = f"{symbol}_{today}.csv"

        filepath = os.path.join(symbol_dir, filename)

        # 데이터 저장
        df.to_csv(filepath, index=False)
        self.logger.info(f"데이터 저장 완료: {filepath}")

        return filepath

    def save_orderbook(self, orderbook, symbol, is_raw=True):
        """오더북 데이터 저장

        Args:
            orderbook (dict): 오더북 데이터
            symbol (str): 거래 쌍
            is_raw (bool): 원시 데이터 여부

        Returns:
            str: 저장된 파일 경로
        """
        if not orderbook:
            self.logger.warning(f"{symbol} 저장할 오더북 데이터가 없습니다.")
            return None

        # 디렉토리 경로 생성
        symbol_dir = self.get_symbol_dir(symbol, "orderbook", is_raw)

        # 파일명 생성
        ts = orderbook['timestamp']
        if isinstance(ts, datetime):
            timestamp = ts.strftime('%Y%m%d_%H%M%S')
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f"{symbol}_orderbook_{timestamp}.json"
        filepath = os.path.join(symbol_dir, filename)

        # JSON 직렬화를 위해 datetime 객체 변환
        orderbook_copy = orderbook.copy()
        if isinstance(orderbook_copy['timestamp'], datetime):
            orderbook_copy['timestamp'] = orderbook_copy['timestamp'].isoformat()

        # 데이터 저장
        with open(filepath, 'w') as f:
            json.dump(orderbook_copy, f, indent=2)

        self.logger.info(f"오더북 데이터 저장 완료: {filepath}")
        return filepath

    def load_dataframe(self, symbol, interval=None, date=None, data_type="klines", is_raw=True):
        """CSV 파일에서 데이터프레임 로드

        Args:
            symbol (str): 거래 쌍
            interval (str, optional): 시간 간격
            date (str, optional): 날짜 (YYYY-MM-DD)
            data_type (str): 데이터 유형
            is_raw (bool): 원시 데이터 여부

        Returns:
            DataFrame: 로드된 데이터프레임
        """
        # 디렉토리 경로 생성
        symbol_dir = self.get_symbol_dir(symbol, data_type, is_raw)

        # 파일 찾기
        if date is None:
            # 날짜가 지정되지 않으면 최신 파일 사용
            pattern = f"{symbol}_{interval}_" if interval else f"{symbol}_"
            files = [f for f in os.listdir(symbol_dir) if f.startswith(pattern) and f.endswith('.csv')]

            if not files:
                self.logger.warning(f"{symbol} 데이터 파일을 찾을 수 없습니다.")
                return pd.DataFrame()

            # 파일명으로 정렬하여 최신 파일 선택
            files.sort(reverse=True)
            filename = files[0]
        else:
            # 날짜가 지정되면 해당 날짜의 파일 사용
            filename = f"{symbol}_{interval}_{date}.csv" if interval else f"{symbol}_{date}.csv"

        filepath = os.path.join(symbol_dir, filename)

        if not os.path.exists(filepath):
            self.logger.warning(f"데이터 파일을 찾을 수 없습니다: {filepath}")
            return pd.DataFrame()

        # 데이터 로드
        df = pd.read_csv(filepath)

        # 날짜/시간 열 변환
        date_columns = ['timestamp', 'close_time', 'open_time', 'date', 'time']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        self.logger.info(f"데이터 로드 완료: {filepath}")
        return df

    def get_available_data_dates(self, symbol, interval=None, data_type="klines", is_raw=True):
        """사용 가능한 데이터의 날짜 목록 반환

        Args:
            symbol (str): 거래 쌍
            interval (str, optional): 시간 간격
            data_type (str): 데이터 유형
            is_raw (bool): 원시 데이터 여부

        Returns:
            list: 날짜 목록
        """
        symbol_dir = self.get_symbol_dir(symbol, data_type, is_raw)

        if not os.path.exists(symbol_dir):
            return []

        pattern = f"{symbol}_{interval}_" if interval else f"{symbol}_"
        files = [f for f in os.listdir(symbol_dir) if f.startswith(pattern) and f.endswith('.csv')]

        # 파일명에서 날짜 추출
        dates = []
        for file in files:
            parts = file.split('_')
            if len(parts) >= 3:
                date_part = parts[-1].replace('.csv', '')
                if len(date_part) == 10 and date_part.count('-') == 2:  # YYYY-MM-DD 형식 확인
                    dates.append(date_part)

        return sorted(dates, reverse=True)


# 모듈 테스트를 위한 코드
if __name__ == "__main__":
    # 데이터 수집 및 저장 테스트
    collector = DataCollector()
    storage = DataStorage()

    # 비트코인 1시간 데이터 수집
    symbol = "BTCUSDT"
    interval = "1h"
    df = collector.get_historical_data(symbol, interval, "1 week ago UTC")

    if not df.empty:
        # 데이터 저장
        filepath = storage.save_dataframe(df, symbol, interval)
        print(f"\n데이터가 저장된 경로: {filepath}")

        # 저장된 데이터 로드
        loaded_df = storage.load_dataframe(symbol, interval)
        print(f"\n로드된 데이터 샘플 (첫 5행):")
        print(loaded_df.head())

        # 사용 가능한 데이터 날짜 출력
        dates = storage.get_available_data_dates(symbol, interval)
        print(f"\n사용 가능한 {symbol} {interval} 데이터 날짜: {dates}")
