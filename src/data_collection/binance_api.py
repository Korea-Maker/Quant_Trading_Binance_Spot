# binance_api.py

from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
import os
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, TEST_MODE

class BinanceAPI:
    """
    Binance API Connect Class
    """
    def __init__(self, api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET):
        """
        Args:
            api_key: 바이낸스 API Key
            api_secret: 바이낸스 API Secret
        """
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.client = None
        self.connect()

    def connect(self):
        """
        Binance API Connect Method
        """
        try:
            self.client = Client(self.api_key, self.api_secret, testnet=TEST_MODE)

            # 서버 시간 확인
            server_time = self.client.get_server_time()
            self.logger.info(f"바이낸스 서버 시간: {server_time}")

            # 계정 정보 확인 (API 키 권한 확인)
            try:
                account_info = self.client.get_account()
                self.logger.info("API 키 권한 확인 성공")
            except:
                self.logger.warning("API 키로 계정 정보 접근 불가 (퍼블릭 데이터만 사용)")

            self.client.ping()
            self.logger.info("바이낸스 API 연결 성공")
        except BinanceAPIException as e:
            self.logger.error(f"바이낸스 API 연결 실패: {e}")
            raise

    def get_exchange_info(self):
        """거래소 정보 조회"""
        try:
            return self.client.get_exchange_info()
        except BinanceAPIException as e:
            self.logger.error(f"거래소 정보 조회 실패 : {e}")
            raise

    def get_ticker(self, symbol):
        """
        특정 심볼의 24시간 가격 통계 조회

        Args:
            symbol: 거래 심볼 => 예시) 'BTCUSDT'

        Returns:
            dict: 24시간 가격 통계
        """
        try:
            res = self.client.get_ticker(symbol=symbol)
            # self.logger.info(f"특정 심볼의 24시간 가격 통계 조회 성공 : {res}") # 테스트
            return res
        except BinanceAPIException as e:
            self.logger.error(f"특정 심볼의 24시간 가격 통계 조회 실패 : {e}")
            raise

    def get_klines(self, symbol, interval, start_str, end_str=None, limit=1000):
        """
        캔들스틱 데이터 조회 (개선된 버전)

        Args:
            symbol: 거래 심볼
            interval: 시간 간격
            start_str: 시작 시간
            end_str: 종료 시간 (optional)
            limit: 최대 데이터 수 (기본값을 1000으로 증가)

        Returns:
            list: 캔들스틱 데이터 목록
        """
        try:
            self.logger.info(f"API 요청: {symbol} {interval} {start_str} ~ {end_str} (limit: {limit})")

            result = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )

            self.logger.info(f"API 응답: {len(result) if result else 0}개 데이터")

            return result

        except BinanceAPIException as e:
            self.logger.error(f"캔들스틱 데이터 조회 실패: {e}")
            self.logger.error(
                f"요청 파라미터: symbol={symbol}, interval={interval}, start_str={start_str}, end_str={end_str}, limit={limit}")
            raise

    def get_depth(self, symbol, limit=100):
        """
        Args:
            symbol: 거래 심볼
            limit: 주문 수준 깊이

        Returns:
            dict: 오더북 데이터
        """
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            self.logger.error(f"오더북 데이터 조회 실패 : {e}")
            raise

if __name__ == "__main__":
    api = BinanceAPI()
    print("서버 시간:", api.client.get_server_time())
    print(f"\nBTC/USDT 티커 정보:{api.get_ticker('BTCUSDT')}")
    print(f"\nBTC/USDT 캔들스틱 데이터:{api.get_klines('BTCUSDT', '1m', '1 day ago UTC')}")
    print(f"\nBTC/USDT 오더북 데이터:{api.get_depth('BTCUSDT')}")