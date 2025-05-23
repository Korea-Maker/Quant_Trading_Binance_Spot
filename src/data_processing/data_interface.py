# src/data_processing/data_interface.py
# 통합 데이터 인터페이스

import pandas as pd
from pandas import DataFrame

from src.utils.logger import get_logger
from typing import Callable

class DataInterface:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.data_source = None
        self.preprocessing_steps = []

    def set_data_source(self, source: str, **kwargs):
        self.data_source = source
        self.source_config = kwargs
        self.logger.info(f"데이터 소스 설정 : {source}")

    def add_preprocessing_step(self, setp_func = Callable):
        self.preprocessing_steps.append(setp_func)
        self.logger.info(f"데이터 전처리 함수 추가 : {setp_func.__name__}")
        return self

    def get_data(self, symbol: str, interval: str, **kwargs) -> pd.DataFrame | None:
        """통합 인터페이스로 데이터 조회"""
        if self.data_source == "historical":
            return self._get_historical_data(symbol, interval, **kwargs)
        elif self.data_source == "realtime":
            return self._get_realtime_data(symbol, interval, **kwargs)
        elif self.data_source == "backtest":
            return self._get_backtest_data(symbol, interval, **kwargs)
        else:
            self.logger.error(f"{self.data_source} 데이터 소스는 지원하지 않는 소스입니다.")
            return pd.DataFrame()

    def _get_historical_data(self, symbol, interval, **kwargs):
        from src.data_collection.collectors import DataCollector

        start_str = kwargs.get("start_str", "1 month ago UTC")
        end_str = kwargs.get("end_str", None)

        collection = DataCollector()
        return collection.get_historical_data(symbol, interval, start_str, end_str)

    def _get_realtime_data(self):
        """실시간 데이터 조회 (웹소켓 구현)"""
        #추후 구현 예정
        self.logger.warning("실시간 데이터 기능은 개발중")
        return pd.DataFrame()

    def _get_backtest_data(self, symbol, interval, **kwargs):
        """백테스트용 저장 데이터 조회"""
        from src.data_collection.storage import DataStorage

        storage = DataStorage()
        return storage.load_dataframe(symbol, interval, data_type="backtest", is_raw=False)