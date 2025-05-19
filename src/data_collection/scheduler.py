import schedule
import time
import threading
from datetime import datetime
import sys
import os

# 경로 추가 (직접 실행 시 필요)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import get_logger
from src.data_collection.collectors import DataCollector
from src.data_collection.storage import DataStorage


class DataScheduler:
    """데이터 수집 스케줄러 클래스"""

    def __init__(self):
        """스케줄러 초기화"""
        self.logger = get_logger(__name__)
        self.collector = DataCollector()
        self.storage = DataStorage()
        self.is_running = False
        self.scheduler_thread = None
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # 기본 수집 대상
        self.intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]  # 기본 시간 간격

    def collect_and_save_klines(self, symbol, interval, lookback_period="1 day ago UTC"):
        """캔들스틱 데이터 수집 및 저장

        Args:
            symbol (str): 거래 쌍
            interval (str): 시간 간격
            lookback_period (str): 조회 기간
        """
        try:
            self.logger.info(f"{symbol} {interval} 정기 데이터 수집 시작")
            df = self.collector.get_historical_data(symbol, interval, lookback_period)

            if not df.empty:
                filepath = self.storage.save_dataframe(df, symbol, interval)
                self.logger.info(f"{symbol} {interval} 데이터 저장 완료: {filepath}")
            else:
                self.logger.warning(f"{symbol} {interval} 수집된 데이터가 없습니다.")

        except Exception as e:
            self.logger.error(f"{symbol} {interval} 데이터 수집 및 저장 실패: {e}")

    def collect_and_save_orderbook(self, symbol, limit=20):
        """오더북 데이터 수집 및 저장

        Args:
            symbol (str): 거래 쌍
            limit (int): 주문 깊이
        """
        try:
            self.logger.info(f"{symbol} 오더북 정기 데이터 수집 시작")
            orderbook = self.collector.get_orderbook_data(symbol, limit)

            if orderbook:
                filepath = self.storage.save_orderbook(orderbook, symbol)
                self.logger.info(f"{symbol} 오더북 데이터 저장 완료: {filepath}")
            else:
                self.logger.warning(f"{symbol} 수집된 오더북 데이터가 없습니다.")

        except Exception as e:
            self.logger.error(f"{symbol} 오더북 데이터 수집 및 저장 실패: {e}")

    def collect_all_symbols(self):
        """모든 심볼의 데이터 수집"""
        self.logger.info("모든 심볼 정기 데이터 수집 시작")

        # 각 심볼 및 시간 간격별 데이터 수집
        for symbol in self.symbols:
            for interval in self.intervals:
                # 시간 간격에 따라 조회 기간 조정
                if interval in ["1m", "5m"]:
                    lookback = "6 hours ago UTC"
                elif interval in ["15m", "30m"]:
                    lookback = "1 day ago UTC"
                elif interval in ["1h", "4h"]:
                    lookback = "7 days ago UTC"
                else:
                    lookback = "30 days ago UTC"

                self.collect_and_save_klines(symbol, interval, lookback)
                time.sleep(1)  # API 속도 제한 방지

            # 오더북 데이터 수집
            self.collect_and_save_orderbook(symbol)
            time.sleep(1)  # API 속도 제한 방지

        self.logger.info("모든 심볼 정기 데이터 수집 완료")

    def start_scheduler(self):
        """스케줄러 시작"""
        if self.is_running:
            self.logger.warning("스케줄러가 이미 실행 중입니다.")
            return

        self.logger.info("데이터 수집 스케줄러 시작")

        # 스케줄 설정
        # 1분 캔들은 5분마다 수집
        schedule.every(5).minutes.do(
            lambda: [self.collect_and_save_klines(s, "1m", "30 minutes ago UTC") for s in self.symbols]
        )

        # 5분, 15분 캔들은 15분마다 수집
        schedule.every(15).minutes.do(
            lambda: [self.collect_and_save_klines(s, i, "2 hours ago UTC")
                     for s in self.symbols for i in ["5m", "15m"]]
        )

        # 1시간 캔들은 1시간마다 수집
        schedule.every(1).hours.do(
            lambda: [self.collect_and_save_klines(s, "1h", "1 day ago UTC") for s in self.symbols]
        )

        # 4시간 캔들은 4시간마다 수집
        schedule.every(4).hours.do(
            lambda: [self.collect_and_save_klines(s, "4h", "5 days ago UTC") for s in self.symbols]
        )

        # 일봉은 매일 자정에 수집
        schedule.every().day.at("00:01").do(
            lambda: [self.collect_and_save_klines(s, "1d", "30 days ago UTC") for s in self.symbols]
        )

        # 오더북은 30분마다 수집
        schedule.every(30).minutes.do(
            lambda: [self.collect_and_save_orderbook(s) for s in self.symbols]
        )

        # 시작 시 즉시 한 번 수집
        self.collect_all_symbols()

        # 백그라운드 스레드에서 스케줄러 실행
        self.is_running = True

        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)

        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()

        self.logger.info("스케줄러가 백그라운드에서 실행 중입니다.")

    def stop_scheduler(self):
        """스케줄러 중지"""
        if not self.is_running:
            self.logger.warning("스케줄러가 실행 중이 아닙니다.")
            return

        self.logger.info("데이터 수집 스케줄러 중지 중...")
        self.is_running = False

        # 모든 작업 취소
        schedule.clear()

        # 스레드 종료 대기
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        self.logger.info("데이터 수집 스케줄러가 중지되었습니다.")

    def set_symbols(self, symbols):
        """수집 대상 심볼 설정

        Args:
            symbols (list): 거래 쌍 목록
        """
        self.symbols = symbols
        self.logger.info(f"수집 대상 심볼이 설정되었습니다: {symbols}")

    def set_intervals(self, intervals):
        """수집 대상 시간 간격 설정

        Args:
            intervals (list): 시간 간격 목록
        """
        self.intervals = intervals
        self.logger.info(f"수집 대상 시간 간격이 설정되었습니다: {intervals}")


# 모듈 테스트를 위한 코드
if __name__ == "__main__":
    import signal

    scheduler = DataScheduler()

    # 테스트를 위해 수집 대상 제한
    scheduler.set_symbols(["BTCUSDT"])
    scheduler.set_intervals(["1m", "4h"])


    # Ctrl+C로 종료할 수 있도록 시그널 핸들러 설정
    def signal_handler(sig, frame):
        print("\n프로그램 종료 중...")
        scheduler.stop_scheduler()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    print("데이터 수집 스케줄러 테스트 시작 (종료하려면 Ctrl+C를 누르세요)")
    print("수집 대상:", scheduler.symbols)
    print("시간 간격:", scheduler.intervals)

    # 스케줄러 시작
    scheduler.start_scheduler()

    # 메인 스레드 유지
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.stop_scheduler()
        print("프로그램이 종료되었습니다.")