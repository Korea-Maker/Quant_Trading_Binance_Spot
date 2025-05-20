# collectors.py

import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import sys

# 경로 추가 (직접 실행 시 필요)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.data_collection.binance_api import BinanceAPI


class DataCollector:
    """시장 데이터 수집 클래스"""
    def __init__(self, api=None):
        """
        데이터 수집기 초기화

        Args:
            api (BinanceAPI, optional): 바이낸스 API 인스턴스
        """
        self.logger = get_logger(__name__)
        self.api = api or BinanceAPI()

    def get_current_price(self, symbol):
        """현재 가격 정보 조회

        Args:
            symbol (str): 거래 쌍 (예: 'BTCUSDT')

        Returns:
            dict: 현재 가격 정보
        """
        try:
            ticker = self.api.get_ticker(symbol)
            price_info = {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'bid': float(ticker['bidPrice']),
                'ask': float(ticker['askPrice']),
                'volume': float(ticker['volume']),
                'change_percent': float(ticker['priceChangePercent']),
                'timestamp': datetime.now()
            }
            self.logger.info(f"{symbol} 현재 가격: {price_info['price']} USDT")
            return price_info
        except Exception as e:
            self.logger.error(f"현재 가격 조회 실패: {e}")
            raise

    def get_historical_data(self, symbol, interval, start_str, end_str=None, limit=1000):
        """과거 캔들스틱 데이터 수집

        Args:
            symbol (str): 거래 쌍 (예: 'BTCUSDT')
            interval (str): 시간 간격 (예: '1m', '1h', '1d')
            start_str (str): 시작 시간 (예: '1 day ago UTC', '1 Jan, 2021')
            end_str (str, optional): 종료 시간
            limit (int, optional): 최대 데이터 수

        Returns:
            DataFrame: OHLCV 데이터
        """
        try:
            self.logger.info(f"{symbol} {interval} 과거 데이터 수집 중... ({start_str} ~ {end_str or 'now'})")

            # 데이터 수집
            klines = self.api.get_klines(symbol, interval, start_str, end_str, limit)

            if not klines:
                self.logger.warning(f"{symbol} {interval} 데이터가 없습니다.")
                return pd.DataFrame()

            # 데이터프레임으로 변환
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

            # 필요 없는 컬럼 제거
            df = df.drop(['ignored'], axis=1)

            # ✅ ✅ ✅ 반드시 timestamp를 index로 (핵심!)
            df = df.set_index('timestamp')

            self.logger.info(f"{symbol} {interval} 과거 데이터 {len(df)}개 수집 완료")
            return df
        except Exception as e:
            self.logger.error(f"데이터 수집 중 오류: {e}")
            return pd.DataFrame()

    def get_orderbook_data(self, symbol, limit=20):
        """오더북 데이터 수집

        Args:
            symbol (str): 거래 쌍 (예: 'BTCUSDT')
            limit (int, optional): 주문 수준 깊이

        Returns:
            dict: 매수/매도 호가 정보
        """
        try:
            self.logger.info(f"{symbol} 오더북 데이터 수집 중...")
            depth = self.api.get_depth(symbol, limit)

            # 타임스탬프 추가
            depth['timestamp'] = datetime.now()

            # 문자열을 숫자로 변환
            depth['bids'] = [[float(p), float(q)] for p, q in depth['bids']]
            depth['asks'] = [[float(p), float(q)] for p, q in depth['asks']]

            self.logger.info(f"{symbol} 오더북 데이터 수집 완료")
            return depth

        except Exception as e:
            self.logger.error(f"오더북 데이터 수집 실패: {e}")
            raise

    def get_multiple_symbols_data(self, symbols, interval, start_str, end_str=None, limit=1000):
        """여러 심볼의 과거 데이터 수집

        Args:
            symbols (list): 거래 쌍 목록
            interval (str): 시간 간격
            start_str (str): 시작 시간
            end_str (str, optional): 종료 시간
            limit (int, optional): 최대 데이터 수

        Returns:
            dict: 심볼별 데이터프레임
        """
        result = {}
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol, interval, start_str, end_str, limit)
                if not df.empty:
                    result[symbol] = df
                # API 호출 제한 방지를 위한 딜레이
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"{symbol} 데이터 수집 중 오류: {e}")

        return result


# 모듈 테스트를 위한 코드
if __name__ == "__main__":
    collector = DataCollector()

    # 비트코인 현재 가격 출력
    btc_price = collector.get_current_price("BTCUSDT")
    print(f"\n비트코인 현재 가격: {btc_price['price']} USDT")

    # 과거 4시간 데이터 출력 (최근 10개)
    btc_data = collector.get_historical_data("BTCUSDT", "4h", "1 month ago UTC")
    print("\n비트코인 최근 4시간 캔들 데이터:")
    print(btc_data.tail(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']])

    # 오더북 데이터 출력
    order_book = collector.get_orderbook_data("BTCUSDT", 5)
    print("\n비트코인 오더북 데이터:")
    print(f"매도 호가 (상위 5개): {order_book['asks']}")
    print(f"매수 호가 (상위 5개): {order_book['bids']}")