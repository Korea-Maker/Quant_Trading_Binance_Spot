# src/data_collection/websocket_client.py

import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List
import websocket
from collections import deque
from src.utils.logger import get_logger


class BinanceWebSocketClient:
    """바이낸스 웹소켓 실시간 데이터 수집"""

    def __init__(self, callback_func: Callable = None):
        self.logger = get_logger(__name__)
        self.ws_base_url = "wss://stream.binance.com:9443/ws"
        self.callback = callback_func
        self.subscriptions = {}
        self.data_buffer = {}
        self.ws = None
        self.running = False

    def subscribe_kline(self, symbol: str, interval: str):
        """K선 데이터 구독"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        self.subscriptions[stream_name] = {
            'symbol': symbol,
            'interval': interval,
            'type': 'kline'
        }

        # 데이터 버퍼 초기화
        buffer_key = f"{symbol}_{interval}"
        if buffer_key not in self.data_buffer:
            self.data_buffer[buffer_key] = deque(maxlen=1000)  # 최근 1000개 저장

        self.logger.info(f"구독 추가: {symbol} {interval}")

    def subscribe_ticker(self, symbol: str):
        """실시간 티커 데이터 구독"""
        stream_name = f"{symbol.lower()}@ticker"
        self.subscriptions[stream_name] = {
            'symbol': symbol,
            'type': 'ticker'
        }

    def _on_message(self, ws, message):
        """웹소켓 메시지 처리"""
        try:
            data = json.loads(message)

            # 스트림 이름 확인
            if 'stream' in data:
                stream_info = self.subscriptions.get(data['stream'])
                if stream_info:
                    processed_data = self._process_message(data['data'], stream_info)

                    # 버퍼에 저장
                    if stream_info['type'] == 'kline':
                        buffer_key = f"{stream_info['symbol']}_{stream_info['interval']}"
                        self.data_buffer[buffer_key].append(processed_data)

                    # 콜백 실행
                    if self.callback:
                        self.callback(processed_data, stream_info)

        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}")

    def _process_message(self, data: dict, stream_info: dict) -> dict:
        """메시지 타입별 처리"""
        if stream_info['type'] == 'kline':
            kline = data['k']
            return {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'symbol': stream_info['symbol'],
                'interval': stream_info['interval'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'close_time': pd.to_datetime(kline['T'], unit='ms'),
                'is_closed': kline['x']  # 캔들 완성 여부
            }
        elif stream_info['type'] == 'ticker':
            return {
                'timestamp': pd.to_datetime(data['E'], unit='ms'),
                'symbol': stream_info['symbol'],
                'price': float(data['c']),
                'volume_24h': float(data['v']),
                'price_change_percent': float(data['P'])
            }
        else:
            self.logger.error(f"지원하지 않는 스트림 타입 : {stream_info['type']}")
            raise ValueError(f"지원하지 않는 스트림 타입 : {stream_info['type']}")

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

    def _on_error(self, ws, error):
        """에러 처리"""
        self.logger.error(f"웹소켓 에러: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """웹소켓 연결 종료 콜백 - 파라미터 수정"""
        self.logger.info(f"웹소켓 연결 종료: {close_status_code} - {close_msg}")
        self.connected = False

    def _on_open(self, ws):
        """연결 시작 처리"""
        self.logger.info("웹소켓 연결 성공")

        # 구독 메시지 전송
        if self.subscriptions:
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": list(self.subscriptions.keys()),
                "id": 1
            }
            ws.send(json.dumps(subscribe_message))

    def start(self):
        """웹소켓 연결 시작"""
        if not self.subscriptions:
            self.logger.error("구독할 스트림이 없습니다")
            return

        # 스트림 URL 생성
        streams = "/".join(self.subscriptions.keys())
        url = f"{self.ws_base_url}/{streams}"

        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )

        self.running = True
        self.ws.run_forever()

    def stop(self):
        """웹소켓 연결 종료"""
        self.running = False
        if self.ws:
            self.ws.close()


if __name__ == "__main__":

    def print_callback(data, stream_info):
        print("콜백 데이터:", data, " / 스트림 정보:", stream_info)

    # 클라이언트 생성
    ws_client = BinanceWebSocketClient(callback_func=print_callback)

    # BTCUSDT 1분봉 구독 예시
    ws_client.subscribe_kline("BTCUSDT", "1m")

    # 실시간 ticker 구독 예시
    ws_client.subscribe_ticker("BTCUSDT")

    try:
        ws_client.start()
    except KeyboardInterrupt:
        ws_client.stop()