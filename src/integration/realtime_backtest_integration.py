# src/integration/realtime_backtest_integration.py

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from collections import deque
import json
import threading

from src.utils.logger import get_logger
from src.data_processing.unified_processor import UnifiedDataProcessor, BacktestRealtimeAdapter
from src.data_collection.websocket_client import BinanceWebSocketClient
from src.backtesting.backtest_engine import BacktestEngine
from src.data_collection.collectors import DataCollector


class RealtimeBacktestIntegration:
    """실시간 거래와 백테스팅을 통합 관리하는 시스템"""

    def __init__(self,
                 symbol: str = "BTCUSDT",
                 interval: str = "1m",
                 backtest_lookback_days: int = 30,
                 feedback_update_hours: int = 24):
        """
        Args:
            symbol: 거래 심볼
            interval: 시간 간격
            backtest_lookback_days: 백테스팅 과거 데이터 기간
            feedback_update_hours: 피드백 업데이트 주기
        """
        self.logger = get_logger(__name__)
        self.symbol = symbol
        self.interval = interval
        self.backtest_lookback_days = backtest_lookback_days
        self.feedback_update_hours = feedback_update_hours

        # 컴포넌트 초기화
        self.unified_processor = UnifiedDataProcessor(
            buffer_size=1000,
            min_data_points=200,
            enable_ml_features=True
        )

        # 실시간 데이터를 위한 큐
        self.realtime_data_queue = asyncio.Queue()

        # 웹소켓 클라이언트는 콜백과 함께 초기화
        self.realtime_client = None

        self.data_collector = DataCollector()

        # BacktestEngine 초기화 시 initial_capital 설정
        self.backtest_engine = BacktestEngine(initial_capital=10000)

        self.adapter = BacktestRealtimeAdapter(self.unified_processor)

        # 상태 관리
        self.is_running = False
        self.last_feedback_update = None
        self.performance_history = deque(maxlen=1000)

        # 실시간 거래 추적
        self.active_positions = {}
        self.trade_history = []

        # 웹소켓 스레드
        self.ws_thread = None

    def _websocket_callback(self, data: dict, stream_info: dict):
        """웹소켓 콜백 함수 - 동기 함수에서 비동기 큐로 데이터 전달"""
        # 완성된 캔들만 처리
        if stream_info['type'] == 'kline' and data.get('is_closed', False):
            # asyncio 이벤트 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                # 비동기 태스크로 큐에 추가
                asyncio.run_coroutine_threadsafe(
                    self.realtime_data_queue.put(data),
                    loop
                )
            except RuntimeError:
                # 이벤트 루프가 없는 경우 (테스트 환경 등)
                self.logger.warning("이벤트 루프를 찾을 수 없습니다.")

    async def start(self, strategy_func: Callable):
        """통합 시스템 시작

        Args:
            strategy_func: 거래 전략 함수
        """
        self.is_running = True
        self.logger.info(f"통합 시스템 시작: {self.symbol} {self.interval}")

        try:
            # 초기 백테스팅 실행 및 피드백 설정
            await self._initial_backtest_and_feedback(strategy_func)

            # 웹소켓 클라이언트 생성 및 시작
            self._start_websocket()

            # 실시간 처리 및 주기적 백테스팅 태스크 시작
            tasks = [
                asyncio.create_task(self._realtime_trading_loop(strategy_func)),
                asyncio.create_task(self._periodic_backtest_loop(strategy_func))
            ]

            await asyncio.gather(*tasks)

        except Exception as e:
            self.logger.error(f"시스템 오류: {e}")
            raise
        finally:
            self.is_running = False
            self._stop_websocket()

    def _start_websocket(self):
        """웹소켓 연결 시작"""
        self.realtime_client = BinanceWebSocketClient(callback_func=self._websocket_callback)
        self.realtime_client.subscribe_kline(self.symbol, self.interval)

        # 별도 스레드에서 웹소켓 실행
        self.ws_thread = threading.Thread(target=self.realtime_client.start)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        self.logger.info("웹소켓 연결 시작됨")

    def _stop_websocket(self):
        """웹소켓 연결 중지"""
        if self.realtime_client:
            self.realtime_client.stop()
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        self.logger.info("웹소켓 연결 중지됨")

    async def stop(self):
        """시스템 중지"""
        self.logger.info("통합 시스템 중지 중...")
        self.is_running = False
        self._stop_websocket()

    async def _initial_backtest_and_feedback(self, strategy_func: Callable):
        """초기 백테스팅 실행 및 피드백 설정"""
        self.logger.info("초기 백테스팅 실행 중...")

        try:
            # 과거 데이터 수집 날짜 계산
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.backtest_lookback_days)

            # 먼저 전략을 백테스트 엔진에 추가
            strategy_name = "adaptive_strategy"
            self.backtest_engine.add_strategy(strategy_name, strategy_func)

            # 백테스팅 실행 - 올바른 시그니처 사용
            backtest_results = self.backtest_engine.run_backtest(
                symbol=self.symbol,
                interval=self.interval,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                strategy_name=strategy_name
            )

            if not backtest_results:
                self.logger.error("백테스팅 결과가 없습니다.")
                return

            # 피드백 추출 및 적용
            feedback = self.adapter.extract_feedback_from_backtest(backtest_results)
            self.unified_processor.update_feedback(feedback)

            self.last_feedback_update = datetime.now()

            # 성과 정보 로깅
            performance = backtest_results.get('performance', {})
            self.logger.info(f"초기 백테스팅 완료. 수익률: {performance.get('total_return', 0):.2%}")

        except Exception as e:
            self.logger.error(f"초기 백테스팅 중 오류: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    async def _realtime_trading_loop(self, strategy_func: Callable):
        """실시간 거래 루프"""
        self.logger.info("실시간 거래 루프 시작")

        # 데이터 처리용 큐
        signal_queue = asyncio.Queue()

        # 데이터 처리 태스크
        process_task = asyncio.create_task(
            self.unified_processor.process_realtime_stream(self.realtime_data_queue, signal_queue)
        )

        # 신호 처리 루프
        while self.is_running:
            try:
                # 신호 대기
                signal = await asyncio.wait_for(signal_queue.get(), timeout=1.0)

                # 전략 실행
                action = strategy_func(signal)

                # 거래 실행
                if action != 'hold':
                    await self._execute_trade(action, signal)

                # 성능 추적
                self._track_performance(signal)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"실시간 처리 오류: {e}")

        # 정리
        await self.realtime_data_queue.put(None)
        await process_task

    async def _periodic_backtest_loop(self, strategy_func: Callable):
        """주기적 백테스팅 및 피드백 업데이트 루프"""
        while self.is_running:
            try:
                # 다음 업데이트까지 대기
                hours_since_update = 0
                if self.last_feedback_update:
                    hours_since_update = (datetime.now() - self.last_feedback_update).total_seconds() / 3600

                if hours_since_update >= self.feedback_update_hours:
                    self.logger.info("주기적 백테스팅 실행 중...")

                    # 최근 데이터로 백테스팅
                    await self._run_incremental_backtest(strategy_func)

                    # 피드백 업데이트
                    self.last_feedback_update = datetime.now()

                # 1시간 대기
                await asyncio.sleep(3600)

            except Exception as e:
                self.logger.error(f"주기적 백테스팅 오류: {e}")
                await asyncio.sleep(3600)

    async def _run_incremental_backtest(self, strategy_func: Callable):
        """증분 백테스팅 실행"""
        # 최근 거래 기록 기반 백테스팅
        if len(self.trade_history) < 10:
            return

        # 거래 기록을 DataFrame으로 변환
        trades_df = pd.DataFrame(self.trade_history)

        # 성과 분석
        performance_metrics = self._analyze_recent_performance(trades_df)

        # 피드백 데이터 생성
        feedback = {
            'signal_accuracy': performance_metrics.get('signal_accuracy', {}),
            'pattern_success_rate': performance_metrics.get('pattern_success', {}),
            'indicator_thresholds': self._optimize_thresholds(trades_df)
        }

        # 피드백 적용
        self.unified_processor.update_feedback(feedback)

        self.logger.info(f"증분 백테스팅 완료. 최근 정확도: {performance_metrics.get('overall_accuracy', 0):.2%}")

    async def _execute_trade(self, action: str, signal: Dict):
        """거래 실행 (시뮬레이션)

        Args:
            action: 'buy' 또는 'sell'
            signal: 거래 신호 정보
        """
        trade_record = {
            'timestamp': signal['timestamp'],
            'action': action,
            'price': signal['indicators'].get('close', 0),
            'signal_strength': signal['signal_strength'],
            'confidence': signal['confidence'],
            'patterns': signal['patterns'],
            'position_size': self._calculate_position_size(signal)
        }

        # 포지션 업데이트
        if action == 'buy':
            self.active_positions[signal['timestamp']] = trade_record
        elif action == 'sell' and self.active_positions:
            # 가장 오래된 포지션 청산
            oldest_position = min(self.active_positions.keys())
            entry_trade = self.active_positions.pop(oldest_position)

            # 수익 계산
            trade_record['entry_price'] = entry_trade['price']
            trade_record['profit'] = (trade_record['price'] - entry_trade['price']) / entry_trade['price']
            trade_record['holding_time'] = (trade_record['timestamp'] - oldest_position).total_seconds() / 3600

        # 거래 기록 저장
        self.trade_history.append(trade_record)

        self.logger.info(f"거래 실행: {action} @ {trade_record['price']:.2f}")

    def _calculate_position_size(self, signal: Dict) -> float:
        """포지션 크기 계산

        Args:
            signal: 거래 신호

        Returns:
            포지션 크기 (0-1)
        """
        # 신뢰도와 리스크 기반 포지션 크기 결정
        base_size = 0.1  # 기본 10%

        # 신뢰도 조정
        confidence_multiplier = signal['confidence']

        # 리스크 조정
        risk_multiplier = {
            'LOW': 1.2,
            'MEDIUM': 1.0,
            'HIGH': 0.5
        }.get(signal['risk_level'], 1.0)

        # 최종 포지션 크기
        position_size = base_size * confidence_multiplier * risk_multiplier

        return min(position_size, 0.3)  # 최대 30%

    def _track_performance(self, signal: Dict):
        """성능 추적

        Args:
            signal: 거래 신호
        """
        performance_record = {
            'timestamp': signal['timestamp'],
            'signal_strength': signal['signal_strength'],
            'confidence': signal['confidence'],
            'active_positions': len(self.active_positions),
            'total_trades': len(self.trade_history)
        }

        # 최근 거래 수익률 계산
        if self.trade_history:
            recent_trades = [t for t in self.trade_history[-10:] if 'profit' in t]
            if recent_trades:
                performance_record['recent_avg_profit'] = np.mean([t['profit'] for t in recent_trades])
                performance_record['recent_win_rate'] = len([t for t in recent_trades if t['profit'] > 0]) / len(
                    recent_trades)

        self.performance_history.append(performance_record)

    def _analyze_recent_performance(self, trades_df: pd.DataFrame) -> Dict:
        """최근 거래 성과 분석

        Args:
            trades_df: 거래 기록 DataFrame

        Returns:
            성과 지표
        """
        metrics = {}

        # 전체 정확도
        if 'profit' in trades_df.columns:
            profitable_trades = trades_df[trades_df['profit'] > 0]
            metrics['overall_accuracy'] = len(profitable_trades) / len(trades_df)

        # 신호별 정확도
        signal_accuracy = {}
        for pattern in trades_df['patterns'].explode().unique():
            if pd.notna(pattern):
                pattern_trades = trades_df[
                    trades_df['patterns'].apply(lambda x: pattern in x if isinstance(x, list) else False)]
                if len(pattern_trades) > 0 and 'profit' in pattern_trades.columns:
                    pattern_profitable = pattern_trades[pattern_trades['profit'] > 0]
                    signal_accuracy[pattern] = len(pattern_profitable) / len(pattern_trades)

        metrics['signal_accuracy'] = signal_accuracy

        # 패턴 성공률
        metrics['pattern_success'] = signal_accuracy  # 동일하게 사용

        return metrics

    def _optimize_thresholds(self, trades_df: pd.DataFrame) -> Dict:
        """거래 결과 기반 임계값 최적화

        Args:
            trades_df: 거래 기록 DataFrame

        Returns:
            최적화된 임계값
        """
        thresholds = {}

        # RSI 임계값 최적화
        if 'signal_strength' in trades_df.columns:
            # 수익성 있는 거래의 신호 강도 분석
            profitable_trades = trades_df[trades_df.get('profit', 0) > 0]
            if len(profitable_trades) > 0:
                # 상위/하위 20% 분위수를 새로운 임계값으로
                thresholds['RSI_14'] = {
                    'upper': profitable_trades['signal_strength'].quantile(0.8),
                    'lower': profitable_trades['signal_strength'].quantile(0.2)
                }

        return thresholds

    def get_performance_summary(self) -> Dict:
        """현재 성능 요약 반환"""
        summary = {
            'total_trades': len(self.trade_history),
            'active_positions': len(self.active_positions),
            'last_feedback_update': self.last_feedback_update
        }

        # 실시간 버퍼 데이터 확인
        if self.realtime_client:
            buffer_data = self.realtime_client.get_buffer_data(self.symbol, self.interval)
            summary['buffer_size'] = len(buffer_data)

        # 최근 거래 통계
        if self.trade_history:
            recent_trades = [t for t in self.trade_history[-20:] if 'profit' in t]
            if recent_trades:
                profits = [t['profit'] for t in recent_trades]
                summary['recent_avg_profit'] = np.mean(profits)
                summary['recent_win_rate'] = len([p for p in profits if p > 0]) / len(profits)
                summary['recent_max_profit'] = max(profits)
                summary['recent_max_loss'] = min(profits)

        return summary


# 테스트 전략 함수
def adaptive_strategy(data: pd.DataFrame, position: float) -> int:
    """백테스트 엔진용 적응형 전략

    Args:
        data: 현재까지의 데이터 (DataFrame)
        position: 현재 포지션

    Returns:
        신호 (1: 매수, -1: 매도, 0: 홀드)
    """
    if len(data) < 20:  # 최소 데이터 필요
        return 0

    # 간단한 이동평균 크로스오버 전략
    if 'MA_5' not in data.columns or 'MA_20' not in data.columns:
        # 이동평균 계산
        data['MA_5'] = data['close'].rolling(5).mean()
        data['MA_20'] = data['close'].rolling(20).mean()

    current = data.iloc[-1]
    prev = data.iloc[-2]

    # 골든 크로스 - 매수
    if prev['MA_5'] <= prev['MA_20'] and current['MA_5'] > current['MA_20'] and position == 0:
        return 1

    # 데드 크로스 - 매도
    elif prev['MA_5'] >= prev['MA_20'] and current['MA_5'] < current['MA_20'] and position > 0:
        return -1

    # RSI 기반 신호 (있는 경우)
    if 'RSI_14' in data.columns:
        rsi = current['RSI_14']
        if rsi < 30 and position == 0:  # 과매도 - 매수
            return 1
        elif rsi > 70 and position > 0:  # 과매수 - 매도
            return -1

    return 0  # 홀드


# 사용 예시
if __name__ == "__main__":
    async def test_integration():
        # 통합 시스템 생성
        integration = RealtimeBacktestIntegration(
            symbol="BTCUSDT",
            interval="1m",
            backtest_lookback_days=7,  # 테스트를 위해 짧게
            feedback_update_hours=1  # 테스트를 위해 짧게
        )

        try:
            # 시스템 시작 (실제로는 무한 루프)
            # 테스트를 위해 짧은 시간만 실행
            await asyncio.wait_for(
                integration.start(adaptive_strategy),
                timeout=60  # 1분간 실행
            )
        except asyncio.TimeoutError:
            print("테스트 시간 초과 - 정상 종료")

        # 성능 요약 출력
        summary = integration.get_performance_summary()
        print("\n성능 요약:")
        print(json.dumps(summary, indent=2, default=str))

        # 시스템 중지
        await integration.stop()


    # 실행
    asyncio.run(test_integration())
