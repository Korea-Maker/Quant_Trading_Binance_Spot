# src/data_collection/binance_api.py

from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
import os
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, TEST_MODE, TRADING_TYPE, SSL_VERIFY

# SSL 검증 비활성화 시 경고 메시지 억제
if not SSL_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BinanceAPI:
    """
    Binance API Connect Class
    스팟 거래와 선물 거래를 모두 지원합니다.
    """
    def __init__(self, api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, trading_type=None):
        """
        Args:
            api_key: 바이낸스 API Key
            api_secret: 바이낸스 API Secret
            trading_type: 'spot' 또는 'futures' (None이면 설정에서 가져옴)
        """
        self.logger = get_logger(__name__)
        self.api_key = api_key
        self.api_secret = api_secret
        self.trading_type = trading_type or TRADING_TYPE
        self.client = None
        self.futures_client = None
        self.connect()

    def connect(self):
        """
        Binance API Connect Method
        """
        try:
            # 스팟 거래 클라이언트 (항상 초기화 - 데이터 수집용)
            # SSL 검증 설정: requests_params를 통해 verify 옵션 전달
            requests_params = {'verify': SSL_VERIFY}
            self.client = Client(
                self.api_key, 
                self.api_secret, 
                testnet=TEST_MODE,
                requests_params=requests_params
            )
            
            if not SSL_VERIFY:
                self.logger.warning(
                    "[주의] SSL 인증서 검증이 비활성화되었습니다. "
                    "개발/테스트 환경에서만 사용하세요. 프로덕션 환경에서는 SSL_VERIFY=True로 설정하세요."
                )
            
            # 선물 거래 클라이언트 (선물 거래 사용 시)
            # python-binance의 Client 클래스는 futures_* 메서드를 제공하며,
            # testnet=True일 때 자동으로 올바른 URL을 사용합니다.
            # 하지만 선물 거래 테스트넷은 별도의 URL이 필요할 수 있습니다.
            # 참고: https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
            # - 테스트넷 REST: https://demo-fapi.binance.com
            # - 테스트넷 WebSocket: wss://fstream.binancefuture.com
            if self.trading_type == 'futures':
                # python-binance는 Client 클래스 하나로 스팟과 선물을 모두 지원
                # futures_* 메서드를 사용하면 자동으로 선물 거래 엔드포인트로 요청
                # testnet=True일 때 선물 거래 테스트넷 URL도 자동으로 사용됨
                self.futures_client = self.client  # 같은 클라이언트 사용
                
                if TEST_MODE:
                    self.logger.info("선물 거래 테스트넷 모드 활성화 (demo-fapi.binance.com)")
                else:
                    self.logger.info("선물 거래 메인넷 모드 활성화 (fapi.binance.com)")
            
            # 연결 정보 로깅
            network_type = "테스트넷" if TEST_MODE else "메인넷"
            trading_type_str = "선물 거래" if self.trading_type == 'futures' else "스팟 거래"
            
            if TEST_MODE:
                self.logger.info(f"[바이낸스 {network_type}] {trading_type_str} 모드 - 테스트넷에서 실제 주문 실행 (메인넷에는 영향 없음)")
            else:
                self.logger.warning(f"[바이낸스 {network_type}] {trading_type_str} 모드 - 메인넷에서 실제 주문 실행, 실제 자금 사용")

            # 서버 시간 확인
            server_time = self.client.get_server_time()
            self.logger.info(f"바이낸스 서버 시간: {server_time}")

            # 계정 정보 확인 (API 키 권한 확인)
            try:
                if self.trading_type == 'futures':
                    # 선물 거래 계정 정보 확인
                    try:
                        account_info = self.client.futures_account()
                        self.logger.info("선물 거래 API 키 권한 확인 성공")
                    except BinanceAPIException as e:
                        error_code = getattr(e, 'code', None)
                        if error_code == -2015:
                            self.logger.error(
                                f"선물 거래 API 키 오류 (code={error_code}): Invalid API-key, IP, or permissions for action.\n"
                                f"해결 방법:\n"
                                f"1. 바이낸스 테스트넷에서 선물 거래용 API 키 생성 확인\n"
                                f"2. API 키 권한: 'Enable Futures Trading' 활성화 확인\n"
                                f"3. IP 제한 설정 확인\n"
                                f"4. 테스트넷 URL 확인: https://demo-fapi.binance.com"
                            )
                        else:
                            self.logger.warning(f"선물 거래 계정 정보 접근 실패: {e}")
                        # 스팟 계정 정보로 fallback (권한 확인용)
                        try:
                            account_info = self.client.get_account()
                            self.logger.info("스팟 거래 API 키 권한 확인 성공 (선물 거래 권한은 없음)")
                        except:
                            self.logger.warning("스팟 거래 계정 정보도 접근 불가")
                else:
                    # 스팟 거래 계정 정보 확인
                    account_info = self.client.get_account()
                    self.logger.info("스팟 거래 API 키 권한 확인 성공")
            except BinanceAPIException as e:
                error_code = getattr(e, 'code', None)
                if error_code == -2015:
                    self.logger.error(
                        f"API 키 오류 (code={error_code}): Invalid API-key, IP, or permissions for action.\n"
                        f"해결 방법:\n"
                        f"1. 바이낸스 테스트넷에서 API 키가 올바르게 생성되었는지 확인\n"
                        f"2. API 키 권한 확인: {'Enable Futures Trading' if self.trading_type == 'futures' else 'Enable Spot & Margin Trading'}\n"
                        f"3. IP 제한 설정 확인 (필요시 현재 IP 추가)\n"
                        f"4. 테스트넷/메인넷 API 키 혼동 확인\n"
                        f"5. API 키가 만료되지 않았는지 확인"
                    )
                else:
                    self.logger.warning(f"API 키로 계정 정보 접근 불가: {e}")

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