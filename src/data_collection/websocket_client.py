# src/data_collection/websocket_client.py

import asyncio
import json
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional
from collections import deque
from src.utils.logger import get_logger

# websocket import (안전한 처리)
# 우선순위: 1) websocket-client (직접 사용, 가장 안정적), 2) python-binance의 BinanceSocketManager (선택적)

USE_BINANCE_SOCKET_MANAGER = False
WEBSOCKET_AVAILABLE = False
websocket = None
websocket_client = None  # websocket-client 모듈 저장

# 1. websocket-client 시도 (가장 안정적이고 직접 제어 가능)
# websocket-client는 'websocket'이라는 이름으로 import되지만, WebSocketApp 속성이 있어야 함
try:
    # websocket-client를 명시적으로 import 시도
    import websocket
    # WebSocketApp 속성이 있는지 확인 (websocket-client의 올바른 모듈인지 확인)
    if hasattr(websocket, 'WebSocketApp'):
        websocket_client = websocket
        WEBSOCKET_AVAILABLE = True
    else:
        # 다른 websocket 모듈이 import된 경우 (예: websockets가 websocket으로 import됨)
        # websocket-client가 제대로 설치되지 않았을 수 있음
        websocket = None
        websocket_client = None
        raise AttributeError("websocket 모듈에 WebSocketApp이 없습니다. websocket-client가 올바르게 설치되지 않았을 수 있습니다.")
except (ImportError, AttributeError):
    websocket = None
    websocket_client = None

# 2. python-binance의 BinanceSocketManager 시도 (선택적, websocket-client가 없을 때만)
# 주의: 최신 python-binance 버전에서는 binance.websockets가 없을 수 있음
if not WEBSOCKET_AVAILABLE:
    try:
        # 먼저 websockets가 있는지 확인
        import websockets
        from binance.client import Client
        from binance.websockets import BinanceSocketManager
        USE_BINANCE_SOCKET_MANAGER = True
        WEBSOCKET_AVAILABLE = True
    except ImportError:
        # python-binance의 BinanceSocketManager를 사용할 수 없음
        USE_BINANCE_SOCKET_MANAGER = False

# 3. 최종 확인
if not WEBSOCKET_AVAILABLE:
    websocket = None
    websocket_client = None
    import warnings
    warnings.warn(
        "websocket 라이브러리를 찾을 수 없습니다.\n"
        "다음 중 하나를 설치하세요:\n"
        "  1. pip install websocket-client (권장)\n"
        "  2. pip install websockets (python-binance용)\n"
        "  3. python install_dependencies.py 실행",
        ImportWarning
    )


class BinanceWebSocketClient:
    """바이낸스 웹소켓 실시간 데이터 수집"""

    def __init__(self, callback_func: Callable = None):
        self.logger = get_logger(__name__)
        
        if not WEBSOCKET_AVAILABLE:
            # 디버깅 정보 수집
            debug_info = []
            try:
                import websockets
                debug_info.append("✓ websockets 설치됨")
            except ImportError:
                debug_info.append("✗ websockets 미설치")
            
            try:
                from binance.websockets import BinanceSocketManager
                debug_info.append("✓ BinanceSocketManager 사용 가능")
            except ImportError as e:
                debug_info.append(f"✗ BinanceSocketManager 사용 불가: {e}")
            
            try:
                import websocket
                debug_info.append("✓ websocket-client 설치됨")
            except ImportError:
                debug_info.append("✗ websocket-client 미설치")
            
            error_msg = (
                "websocket 라이브러리를 찾을 수 없습니다.\n"
                f"디버깅 정보: {', '.join(debug_info)}\n"
                "다음 중 하나를 설치하세요:\n"
                "  1. pip install websockets (python-binance용, 권장)\n"
                "  2. pip install websocket-client\n"
                "  3. python install_dependencies.py 실행"
            )
            self.logger.error(error_msg)
            raise ImportError("websocket 라이브러리가 필요합니다.")
        
        # python-binance의 BinanceSocketManager 사용 여부
        self.use_binance_manager = USE_BINANCE_SOCKET_MANAGER
        
        if self.use_binance_manager:
            # python-binance 사용
            from src.data_collection.binance_api import BinanceAPI
            self.binance_api = BinanceAPI()
            self.client = self.binance_api.client
            from binance.websockets import BinanceSocketManager
            self.bm = BinanceSocketManager(self.client)
            self.socket_keys = []
            self.logger.info("python-binance의 BinanceSocketManager 사용 (websocket-client 의존성 없음)")
        else:
            # websocket-client 사용 (기존 방식)
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
            self.ws = None
            self.logger.info("websocket-client 사용")
        
        self.callback = callback_func
        self.subscriptions = {}
        self.data_buffer = {}
        self.running = False
        self.connected = False

    def subscribe_kline(self, symbol: str, interval: str):
        """K선 데이터 구독"""
        if self.use_binance_manager:
            # python-binance 방식
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
                        'is_closed': kline['x']
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
        else:
            # websocket-client 방식 (기존)
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
            self.logger.debug(f"웹소켓 메시지 수신: {len(message)} bytes")
            data = json.loads(message)
            data_keys = list(data.keys()) if isinstance(data, dict) else type(data)
            self.logger.debug(f"파싱된 데이터 키: {data_keys}")

            # 1. 멀티 스트림 형식 확인 ('stream' 키가 있는 경우)
            if 'stream' in data:
                stream_name = data['stream']
                self.logger.debug(f"멀티 스트림 형식: {stream_name}")
                stream_info = self.subscriptions.get(stream_name)
                
                if stream_info:
                    processed_data = self._process_message(data['data'], stream_info)
                    if processed_data:
                        # 버퍼에 저장
                        if stream_info['type'] == 'kline':
                            buffer_key = f"{stream_info['symbol']}_{stream_info['interval']}"
                            self.data_buffer[buffer_key].append(processed_data)
                        
                        # 콜백 실행
                        if self.callback:
                            self.callback(processed_data, stream_info)
                else:
                    self.logger.warning(f"스트림 정보를 찾을 수 없음: {stream_name}")
            
            # 2. 단일 스트림 형식 확인 ('e' 키가 있는 경우 - kline 이벤트)
            elif 'e' in data and data['e'] == 'kline':
                self.logger.debug(f"단일 스트림 형식: kline 이벤트")
                # 구독된 kline 스트림 찾기
                kline_stream = None
                for stream_name, stream_info in self.subscriptions.items():
                    if stream_info.get('type') == 'kline':
                        kline_stream = stream_info
                        break
                
                if kline_stream:
                    # kline 데이터 처리
                    processed_data = self._process_message(data, kline_stream)
                    if processed_data:
                        # 버퍼에 저장
                        buffer_key = f"{kline_stream['symbol']}_{kline_stream['interval']}"
                        self.data_buffer[buffer_key].append(processed_data)
                        self.logger.info(f"Kline 데이터 처리: close={processed_data.get('close', 'N/A')}, is_closed={processed_data.get('is_closed', False)}")
                        
                        # 콜백 실행
                        if self.callback:
                            self.logger.info(f"콜백 함수 호출: close={processed_data.get('close', 'N/A')}")
                            self.callback(processed_data, kline_stream)
                        else:
                            self.logger.warning("콜백 함수가 설정되지 않았습니다!")
                else:
                    self.logger.warning("구독된 kline 스트림을 찾을 수 없습니다")
            
            # 3. 구독 확인 메시지
            elif 'result' in data or 'id' in data:
                self.logger.debug(f"구독 확인 메시지: {data}")
            
            # 4. 알 수 없는 형식
            else:
                self.logger.debug(f"알 수 없는 메시지 형식: {data_keys}")

        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}", exc_info=True)

    def _process_message(self, data: dict, stream_info: dict) -> dict:
        """메시지 타입별 처리"""
        self.logger.debug(f"_process_message 호출: type={stream_info['type']}")
        if stream_info['type'] == 'kline':
            # 'k' 키가 있으면 kline 데이터, 없으면 data 자체가 kline
            kline = data.get('k', data)
            
            if not isinstance(kline, dict):
                self.logger.error(f"Kline 데이터 형식 오류: {type(kline)}")
                return None
            
            try:
                result = {
                    'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                    'symbol': stream_info['symbol'],
                    'interval': stream_info['interval'],
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'close_time': pd.to_datetime(kline['T'], unit='ms'),
                    'is_closed': kline.get('x', False)  # 캔들 완성 여부
                }
                self.logger.debug(f"처리된 Kline: close={result['close']}, is_closed={result['is_closed']}")
                return result
            except KeyError as e:
                self.logger.error(f"Kline 데이터 키 누락: {e}, keys={list(kline.keys())}")
                return None
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
        self.logger.info("웹소켓 연결 성공 (on_open 호출됨)")
        self.logger.info(f"구독된 스트림 수: {len(self.subscriptions)}")

        # 구독 메시지 전송
        if self.subscriptions:
            stream_names = list(self.subscriptions.keys())
            self.logger.info(f"구독 메시지 전송: {stream_names}")
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": stream_names,
                "id": 1
            }
            ws.send(json.dumps(subscribe_message))
            self.logger.info("구독 메시지 전송 완료")
        else:
            self.logger.warning("구독할 스트림이 없습니다!")

    def start(self):
        """웹소켓 연결 시작"""
        if not self.subscriptions:
            self.logger.error("구독할 스트림이 없습니다")
            return

        if self.use_binance_manager:
            # python-binance 방식
            try:
                self.bm.start()
                self.running = True
                self.connected = True
                self.logger.info("웹소켓 연결 시작됨 (python-binance)")
            except Exception as e:
                self.logger.error(f"웹소켓 연결 시작 실패: {e}")
                raise
        else:
            # websocket-client 방식 (기존)
            if websocket_client is None or not hasattr(websocket_client, 'WebSocketApp'):
                raise ImportError(
                    "websocket-client가 올바르게 설치되지 않았습니다.\n"
                    "재설치: pip install --upgrade --force-reinstall websocket-client"
                )
            
            # 스트림 URL 생성
            streams = "/".join(self.subscriptions.keys())
            url = f"{self.ws_base_url}/{streams}"

            self.ws = websocket_client.WebSocketApp(
                url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            self.running = True
            self.logger.info("WebSocket run_forever() 시작...")
            self.ws.run_forever()
            self.logger.info("WebSocket run_forever() 종료")

    def stop(self):
        """웹소켓 연결 종료"""
        self.running = False
        self.connected = False
        
        if self.use_binance_manager:
            # python-binance 방식
            try:
                # 모든 소켓 종료
                for socket_key in self.socket_keys:
                    self.bm.stop_socket(socket_key)
                
                # BinanceSocketManager 종료
                self.bm.close()
                self.logger.info("웹소켓 연결 종료됨 (python-binance)")
            except Exception as e:
                self.logger.error(f"웹소켓 종료 중 오류: {e}")
        else:
            # websocket-client 방식 (기존)
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