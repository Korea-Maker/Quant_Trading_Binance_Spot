# src/data_collection/websocket_client_async.py
"""
websockets 라이브러리를 사용하는 비동기 웹소켓 클라이언트

websocket-client 대신 websockets 라이브러리를 사용합니다.
이미 requirements.txt에 포함되어 있습니다.
"""

import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional
from collections import deque
from src.utils.logger import get_logger

# websockets 라이브러리 사용 (비동기)
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    import warnings
    warnings.warn(
        "websockets 라이브러리가 설치되지 않았습니다.\n"
        "설치: pip install websockets",
        ImportWarning
    )


class BinanceWebSocketClient:
    """바이낸스 웹소켓 실시간 데이터 수집 (websockets 라이브러리 사용)"""

    def __init__(self, callback_func: Callable = None):
        self.logger = get_logger(__name__)
        
        if not WEBSOCKETS_AVAILABLE:
            error_msg = (
                "websockets 라이브러리가 설치되지 않아 웹소켓 클라이언트를 사용할 수 없습니다.\n"
                "설치 방법: pip install websockets"
            )
            self.logger.error(error_msg)
            raise ImportError("websockets 라이브러리가 필요합니다.")
        
        self.ws_base_url = "wss://stream.binance.com:9443/ws"
        self.callback = callback_func
        self.subscriptions = {}
        self.data_buffer = {}
        self.ws = None
        self.running = False
        self.connected = False
        self.loop = None
        self.task = None

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
            self.data_buffer[buffer_key] = deque(maxlen=1000)

        self.logger.info(f"구독 추가: {symbol} {interval}")

    def subscribe_ticker(self, symbol: str):
        """실시간 티커 데이터 구독"""
        stream_name = f"{symbol.lower()}@ticker"
        self.subscriptions[stream_name] = {
            'symbol': symbol,
            'type': 'ticker'
        }

    async def _handle_message(self, message: str):
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
                'is_closed': kline['x']
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
            self.logger.error(f"지원하지 않는 스트림 타입: {stream_info['type']}")
            raise ValueError(f"지원하지 않는 스트림 타입: {stream_info['type']}")

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

    async def _run_websocket(self):
        """웹소켓 실행 (비동기)"""
        if not self.subscriptions:
            self.logger.error("구독할 스트림이 없습니다")
            return

        # 스트림 URL 생성
        streams = "/".join(self.subscriptions.keys())
        url = f"{self.ws_base_url}/{streams}"

        try:
            async with websockets.connect(url) as ws:
                self.ws = ws
                self.connected = True
                self.logger.info("웹소켓 연결 성공 (websockets 라이브러리)")

                # 구독 메시지 전송
                subscribe_message = {
                    "method": "SUBSCRIBE",
                    "params": list(self.subscriptions.keys()),
                    "id": 1
                }
                await ws.send(json.dumps(subscribe_message))

                # 메시지 수신 루프
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.warning("웹소켓 연결이 종료되었습니다")
                        break
                    except Exception as e:
                        self.logger.error(f"웹소켓 메시지 수신 오류: {e}")
                        break

        except Exception as e:
            self.logger.error(f"웹소켓 연결 오류: {e}")
        finally:
            self.connected = False
            self.ws = None

    def start(self):
        """웹소켓 연결 시작 (동기 래퍼)"""
        if self.running:
            self.logger.warning("웹소켓이 이미 실행 중입니다")
            return

        self.running = True
        
        # 새 이벤트 루프 생성 (스레드에서 실행)
        def run_in_thread():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._run_websocket())

        import threading
        self.task = threading.Thread(target=run_in_thread, daemon=True)
        self.task.start()

    def stop(self):
        """웹소켓 연결 종료"""
        self.running = False
        if self.ws:
            try:
                # 비동기 종료는 루프에서 처리
                if self.loop and self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            except Exception as e:
                self.logger.error(f"웹소켓 종료 중 오류: {e}")
        
        if self.task:
            self.task.join(timeout=5)
        
        self.connected = False


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
        import time
        time.sleep(60)  # 60초 실행
    except KeyboardInterrupt:
        pass
    finally:
        ws_client.stop()

