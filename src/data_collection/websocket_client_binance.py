# src/data_collection/websocket_client_binance.py
"""
python-binance의 BinanceSocketManager를 사용하는 웹소켓 클라이언트

websocket-client 의존성 없이 python-binance의 내장 기능을 사용합니다.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional
from collections import deque
from src.utils.logger import get_logger
from src.data_collection.binance_api import BinanceAPI
from src.config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, TEST_MODE

# python-binance의 BinanceSocketManager 사용
try:
    from binance.client import Client
    from binance.websockets import BinanceSocketManager
    BINANCE_WEBSOCKET_AVAILABLE = True
except ImportError:
    BINANCE_WEBSOCKET_AVAILABLE = False
    import warnings
    warnings.warn(
        "python-binance의 BinanceSocketManager를 사용할 수 없습니다.",
        ImportWarning
    )


class BinanceWebSocketClient:
    """바이낸스 웹소켓 실시간 데이터 수집 (python-binance 사용)"""

    def __init__(self, callback_func: Callable = None):
        self.logger = get_logger(__name__)
        
        if not BINANCE_WEBSOCKET_AVAILABLE:
            error_msg = (
                "python-binance의 BinanceSocketManager를 사용할 수 없습니다.\n"
                "python-binance가 설치되어 있는지 확인하세요: pip install python-binance"
            )
            self.logger.error(error_msg)
            raise ImportError("python-binance가 필요합니다.")
        
        # BinanceAPI 인스턴스 생성
        self.binance_api = BinanceAPI()
        self.client = self.binance_api.client
        
        # BinanceSocketManager 생성
        self.bm = BinanceSocketManager(self.client)
        
        self.callback = callback_func
        self.subscriptions = {}
        self.data_buffer = {}
        self.socket_keys = []  # 구독 키 저장
        self.running = False
        self.connected = False

    def subscribe_kline(self, symbol: str, interval: str):
        """K선 데이터 구독"""
        self.logger.info(f"구독 추가: {symbol} {interval}")
        
        # 데이터 버퍼 초기화
        buffer_key = f"{symbol}_{interval}"
        if buffer_key not in self.data_buffer:
            self.data_buffer[buffer_key] = deque(maxlen=1000)
        
        # 콜백 함수 정의
        def process_message(msg):
            try:
                kline = msg['k']
                processed_data = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'symbol': symbol,
                    'interval': interval,
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'close_time': pd.to_datetime(kline['T'], unit='ms'),
                    'is_closed': kline['x']  # 캔들 완성 여부
                }
                
                # 버퍼에 저장
                self.data_buffer[buffer_key].append(processed_data)
                
                # 콜백 실행
                if self.callback:
                    stream_info = {
                        'symbol': symbol,
                        'interval': interval,
                        'type': 'kline'
                    }
                    self.callback(processed_data, stream_info)
            except Exception as e:
                self.logger.error(f"K선 메시지 처리 오류: {e}")
        
        # 구독 시작
        socket_key = self.bm.start_kline_socket(symbol, process_message, interval=interval)
        self.socket_keys.append(socket_key)
        self.subscriptions[socket_key] = {
            'symbol': symbol,
            'interval': interval,
            'type': 'kline'
        }

    def subscribe_ticker(self, symbol: str):
        """실시간 티커 데이터 구독"""
        self.logger.info(f"티커 구독 추가: {symbol}")
        
        # 콜백 함수 정의
        def process_message(msg):
            try:
                processed_data = {
                    'timestamp': pd.to_datetime(msg['E'], unit='ms'),
                    'symbol': symbol,
                    'price': float(msg['c']),
                    'volume_24h': float(msg['v']),
                    'price_change_percent': float(msg['P'])
                }
                
                # 콜백 실행
                if self.callback:
                    stream_info = {
                        'symbol': symbol,
                        'type': 'ticker'
                    }
                    self.callback(processed_data, stream_info)
            except Exception as e:
                self.logger.error(f"티커 메시지 처리 오류: {e}")
        
        # 구독 시작
        socket_key = self.bm.start_symbol_ticker_socket(symbol, process_message)
        self.socket_keys.append(socket_key)
        self.subscriptions[socket_key] = {
            'symbol': symbol,
            'type': 'ticker'
        }

    def get_buffer_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """버퍼에서 데이터 조회"""
        buffer_key = f"{symbol}_{interval}"
        if buffer_key in self.data_buffer:
            data_list = list(self.data_buffer[buffer_key])
            if data_list:
                df = pd.DataFrame(data_list)
                df.set_index('timestamp', inplace=True)
                return df
        return pd.DataFrame()

    def start(self):
        """웹소켓 연결 시작"""
        if not self.socket_keys:
            self.logger.error("구독할 스트림이 없습니다")
            return
        
        try:
            self.bm.start()
            self.running = True
            self.connected = True
            self.logger.info("웹소켓 연결 시작됨 (python-binance)")
        except Exception as e:
            self.logger.error(f"웹소켓 연결 시작 실패: {e}")
            raise

    def stop(self):
        """웹소켓 연결 종료"""
        self.running = False
        self.connected = False
        
        try:
            # 모든 소켓 종료
            for socket_key in self.socket_keys:
                self.bm.stop_socket(socket_key)
            
            # BinanceSocketManager 종료
            self.bm.close()
            self.logger.info("웹소켓 연결 종료됨")
        except Exception as e:
            self.logger.error(f"웹소켓 종료 중 오류: {e}")


if __name__ == "__main__":
    def print_callback(data, stream_info):
        print(f"콜백 데이터: {data} / 스트림 정보: {stream_info}")

    # 클라이언트 생성
    ws_client = BinanceWebSocketClient(callback_func=print_callback)

    # BTCUSDT 1분봉 구독 예시
    ws_client.subscribe_kline("BTCUSDT", "1m")

    # 실시간 ticker 구독 예시
    ws_client.subscribe_ticker("BTCUSDT")

    try:
        ws_client.start()
        # 무한 대기 (실제로는 스레드에서 실행)
        import time
        time.sleep(60)  # 60초 실행
    except KeyboardInterrupt:
        pass
    finally:
        ws_client.stop()

