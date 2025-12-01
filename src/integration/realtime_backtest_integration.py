# src/integration/realtime_backtest_integration.py

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import json
import threading

from src.utils.logger import get_logger
from src.data_processing.unified_processor import UnifiedDataProcessor, BacktestRealtimeAdapter
from src.data_collection.websocket_client import BinanceWebSocketClient
from src.backtesting.backtest_engine import BacktestEngine
from src.data_collection.collectors import DataCollector
from src.execution.spot_order_manager import SpotOrderManager
from src.execution.order_manager import FuturesOrderManager, OrderSide, PositionType
from src.config.settings import TRADING_TYPE
from src.strategy.base import BaseStrategy
from src.monitoring.performance import PerformanceMonitor
from src.monitoring.dashboard import TradingDashboard
from src.risk_management import ExposureManager


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
            min_data_points=50,  # 200 -> 50으로 감소 (더 빠른 신호 생성)
            enable_ml_features=True
        )

        # 실시간 데이터를 위한 큐 (최대 크기 제한으로 메모리 보호)
        self.realtime_data_queue = asyncio.Queue(maxsize=1000)

        # 웹소켓 클라이언트는 콜백과 함께 초기화
        self.realtime_client = None
        
        # 메인 이벤트 루프 저장 (웹소켓 콜백에서 사용)
        self.main_loop = None

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
        
        # 주문 관리자 (스팟 거래 또는 선물 거래)
        self.order_manager: Optional[SpotOrderManager | FuturesOrderManager] = None
        self.strategy: Optional[BaseStrategy] = None
        self.total_capital: float = 10000.0  # 기본 자본
        
        # ExposureManager 초기화 (기본값 사용: max_total_exposure_pct=0.3, max_per_symbol_exposure_pct=0.1, max_concurrent_positions=5)
        self.exposure_manager = ExposureManager()
        
        # 모니터링
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.dashboard: Optional[TradingDashboard] = None

    async def _safe_queue_put(self, data: dict):
        """안전하게 큐에 데이터 추가 (큐가 가득 찬 경우 처리)"""
        try:
            # 큐 크기 확인 (가득 찬 경우 경고)
            try:
                queue_size = self.realtime_data_queue.qsize()
                if queue_size > 500:  # 큐가 절반 이상 찬 경우
                    self.logger.warning(
                        f"큐가 가득 차고 있습니다. 현재 크기: {queue_size}/1000. "
                        f"데이터 처리 속도를 확인하세요."
                    )
            except AttributeError:
                # qsize()가 없는 경우 (일부 Python 버전)
                pass
            
            # put_nowait 시도 (논블로킹)
            self.realtime_data_queue.put_nowait(data)
        except asyncio.QueueFull:
            # 큐가 가득 찬 경우, 가장 오래된 항목 제거 후 추가
            try:
                self.realtime_data_queue.get_nowait()  # 가장 오래된 항목 제거
                self.realtime_data_queue.put_nowait(data)  # 새 항목 추가
                self.logger.warning("큐가 가득 차서 가장 오래된 데이터를 제거하고 새 데이터를 추가했습니다.")
            except asyncio.QueueEmpty:
                # 큐가 비어있는 경우 (동시성 문제) - 일반 put 사용
                await self.realtime_data_queue.put(data)
        except Exception as e:
            self.logger.error(f"큐에 데이터 추가 중 오류: {e}", exc_info=True)
            raise

    def _websocket_callback(self, data: dict, stream_info: dict):
        """웹소켓 콜백 함수 - 동기 함수에서 비동기 큐로 데이터 전달"""
        self.logger.info(f"웹소켓 콜백 호출됨: stream_type={stream_info.get('type', 'unknown')}")
        
        # kline 데이터 처리 (완성된 캔들 우선, 진행 중인 캔들도 처리)
        if stream_info.get('type') == 'kline':
            is_closed = data.get('is_closed', False)
            close_price = data.get('close', 'N/A')
            
            self.logger.info(f"Kline 데이터 수신: is_closed={is_closed}, close={close_price}")
            
            # 메인 이벤트 루프 사용 (웹소켓은 별도 스레드에서 실행되므로)
            if self.main_loop is None:
                self.logger.error("메인 이벤트 루프가 설정되지 않았습니다. start() 메서드가 호출되었는지 확인하세요.")
                return
            
            try:
                # 이벤트 루프가 닫혔는지 확인
                if self.main_loop.is_closed():
                    self.logger.warning("메인 이벤트 루프가 닫혔습니다. 큐에 데이터를 추가할 수 없습니다.")
                    return
                
                # 비동기 태스크로 큐에 추가 (메인 루프 사용)
                # _safe_queue_put을 사용하여 큐가 가득 찬 경우 자동 처리
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._safe_queue_put(data),
                        self.main_loop
                    )
                    # 완료 대기 (타임아웃 2초)
                    try:
                        future.result(timeout=2.0)
                        if is_closed:
                            self.logger.debug(f"[OK] 완성된 캔들 데이터 큐에 추가됨: {close_price}")
                        else:
                            self.logger.debug(f"[OK] 진행 중인 캔들 데이터 큐에 추가됨: {close_price}")
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            f"큐에 데이터 추가 타임아웃 (2초 초과). "
                            f"큐가 가득 찼거나 처리 속도가 느립니다. "
                            f"데이터를 건너뜁니다: {close_price}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"큐에 데이터 추가 실패: {type(e).__name__}: {str(e)}",
                            exc_info=True
                        )
                except RuntimeError as e:
                    # 이벤트 루프 관련 오류
                    if "loop is closed" in str(e) or "Event loop is closed" in str(e):
                        self.logger.warning("이벤트 루프가 닫혔습니다. 웹소켓 콜백을 중단합니다.")
                    else:
                        self.logger.error(f"웹소켓 콜백 처리 중 RuntimeError: {e}", exc_info=True)
                    
            except RuntimeError as e:
                # 이벤트 루프 관련 오류
                if "loop is closed" in str(e) or "Event loop is closed" in str(e):
                    self.logger.warning("이벤트 루프가 닫혔습니다. 웹소켓 콜백을 중단합니다.")
                else:
                    self.logger.error(f"웹소켓 콜백 처리 중 RuntimeError: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"웹소켓 콜백 처리 중 오류: {type(e).__name__}: {str(e)}", exc_info=True)
        else:
            self.logger.debug(f"kline이 아닌 데이터 타입: {stream_info.get('type')}")

    async def start(self, strategy: BaseStrategy, total_capital: float = 10000.0):
        """통합 시스템 시작

        Args:
            strategy: 거래 전략 객체 (BaseStrategy)
            total_capital: 총 자본
        """
        self.is_running = True
        self.strategy = strategy
        self.total_capital = total_capital
        
        # 메인 이벤트 루프 저장 (웹소켓 콜백에서 사용)
        self.main_loop = asyncio.get_running_loop()
        
        self.logger.info(f"통합 시스템 시작: {self.symbol} {self.interval}")

        try:
            # ExposureManager에 총 자본 설정
            self.exposure_manager.set_total_capital(self.total_capital)
            
            # 주문 관리자 초기화 (거래 타입에 따라 선택, ExposureManager 전달)
            if TRADING_TYPE == 'futures':
                self.order_manager = FuturesOrderManager(
                    symbol=self.symbol,
                    initial_leverage=5.0,  # 기본 레버리지 (필요시 설정에서 가져올 수 있음)
                    position_type=PositionType.ISOLATED,  # 격리 마진 모드
                    exposure_manager=self.exposure_manager
                )
                self.logger.info(f"선물 거래 주문 관리자 초기화: {self.symbol} (ExposureManager 연동)")
            else:
                self.order_manager = SpotOrderManager(
                    symbol=self.symbol,
                    exposure_manager=self.exposure_manager
                )
                self.logger.info(f"스팟 거래 주문 관리자 초기화: {self.symbol} (ExposureManager 연동)")
            
            # 모니터링 초기화
            self.performance_monitor = PerformanceMonitor(initial_capital=total_capital)
            self.dashboard = TradingDashboard(performance_monitor=self.performance_monitor)
            self.dashboard.update_status("RUNNING")
            
            # 전략 활성화
            self.strategy.is_active = True
            
            # 웹소켓 클라이언트 생성 및 시작 (먼저 시작하여 데이터 수집)
            self._start_websocket()
            
            # 초기 데이터 수집 대기 (최소 데이터 포인트 확보)
            self.logger.info(f"초기 데이터 수집 대기 중... (최소 {self.unified_processor.min_data_points}개 필요)")
            await asyncio.sleep(5)  # 웹소켓 연결 안정화 대기
            
            # 초기 백테스팅 실행 및 피드백 설정
            await self._initial_backtest_and_feedback(self._strategy_func_wrapper)

            # 실시간 처리 및 주기적 백테스팅 태스크 시작
            tasks = [
                asyncio.create_task(self._realtime_trading_loop(self._strategy_func_wrapper)),
                asyncio.create_task(self._periodic_backtest_loop(self._strategy_func_wrapper)),
                asyncio.create_task(self._risk_management_loop()),
                asyncio.create_task(self._dashboard_update_loop()),
                asyncio.create_task(self._adaptive_adjustment_loop())  # 피드백 루프 추가
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
        
        if self.dashboard:
            self.dashboard.update_status("STOPPED")
            self.dashboard.print_dashboard()
        
        self._stop_websocket()

    async def _initial_backtest_and_feedback(self, strategy_func: Callable):
        """초기 백테스팅 실행 및 피드백 설정"""
        self.logger.info("초기 백테스팅 실행 중...")

        try:
            # 과거 데이터 수집 날짜 계산
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.backtest_lookback_days)

            # 먼저 전략을 백테스트 엔진에 추가 (백테스팅용 어댑터 사용)
            strategy_name = "adaptive_strategy"
            self.backtest_engine.add_strategy(strategy_name, self._backtest_strategy_adapter)

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

    def _strategy_func_wrapper(self, signal: Dict) -> str:
        """전략 함수 래퍼 (실시간 거래용)"""
        if self.strategy:
            return self.strategy.generate_signal(signal)
        return 'hold'
    
    def _backtest_strategy_adapter(self, data: pd.DataFrame, position: float) -> int:
        """
        백테스팅 엔진용 전략 어댑터
        
        백테스팅 엔진은 (data: pd.DataFrame, position: float) -> int 시그니처를 기대하지만,
        실제 전략은 Dict를 받으므로 어댑터가 필요합니다.
        
        Args:
            data: OHLCV 데이터 DataFrame
            position: 현재 포지션 (양수: 롱, 0: 없음)
            
        Returns:
            int: 1 (매수), -1 (매도), 0 (홀드)
        """
        if self.strategy is None:
            return 0
        
        try:
            # DataFrame의 마지막 행을 Dict로 변환
            if data.empty or len(data) == 0:
                return 0
            
            latest = data.iloc[-1]
            
            # 신호 딕셔너리 생성 (UnifiedDataProcessor 형식과 유사하게)
            signal_dict = {
                'primary_signal': 'HOLD',
                'signal_strength': 0.0,
                'confidence': 50.0,
                'indicators': {
                    'close': float(latest.get('close', 0)),
                    'open': float(latest.get('open', 0)),
                    'high': float(latest.get('high', 0)),
                    'low': float(latest.get('low', 0)),
                    'volume': float(latest.get('volume', 0))
                },
                'patterns': {},
                'timestamp': latest.get('timestamp', datetime.now()) if 'timestamp' in latest else datetime.now()
            }
            
            # 간단한 이동평균 크로스오버 로직 (백테스팅용)
            if len(data) >= 20:
                ma_short = data['close'].rolling(5).mean().iloc[-1]
                ma_long = data['close'].rolling(20).mean().iloc[-1]
                
                if not pd.isna(ma_short) and not pd.isna(ma_long):
                    # 골든 크로스 - 매수
                    if ma_short > ma_long and position == 0:
                        signal_dict['primary_signal'] = 'BUY'
                        signal_dict['signal_strength'] = 1.0
                        signal_dict['confidence'] = 70.0
                    # 데드 크로스 - 매도
                    elif ma_short < ma_long and position > 0:
                        signal_dict['primary_signal'] = 'SELL'
                        signal_dict['signal_strength'] = -1.0
                        signal_dict['confidence'] = 70.0
            
            # 전략 실행
            action = self.strategy.generate_signal(signal_dict)
            
            # 문자열을 int로 변환
            if action == 'buy':
                return 1
            elif action == 'sell':
                return -1
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"백테스팅 전략 어댑터 오류: {e}", exc_info=True)
            return 0
    
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
        signal_count = 0
        while self.is_running:
            try:
                # 신호 대기
                signal = await asyncio.wait_for(signal_queue.get(), timeout=1.0)
                signal_count += 1
                
                self.logger.info(f"신호 수신 (#{signal_count}): {signal.get('primary_signal', 'UNKNOWN')}")

                # 전략 실행
                if self.strategy:
                    action = self.strategy.generate_signal(signal)
                    self.logger.info(f"전략 실행 결과: {action} (신뢰도: {signal.get('confidence', 0):.1f}%)")
                    
                    # 신호 검증
                    if not self.strategy.validate_signal(action, signal):
                        self.logger.warning(f"신호 검증 실패: {action} -> hold로 변경")
                        action = 'hold'
                else:
                    action = strategy_func(signal)
                    self.logger.info(f"전략 함수 실행 결과: {action}")

                # 거래 실행
                if action != 'hold':
                    self.logger.info(f"거래 실행 시작: {action}")
                    await self._execute_trade(action, signal)
                    # 전략에 거래 로깅
                    if self.strategy:
                        self.strategy.log_trade(action, signal, executed=True)
                    self.logger.info(f"거래 실행 완료: {action}")
                else:
                    self.logger.debug(f"거래 실행 스킵: {action} (hold)")

                # 대시보드 업데이트
                if self.dashboard:
                    self.dashboard.update_signal(signal)
                    if 'indicators' in signal and 'close' in signal['indicators']:
                        self.dashboard.update_price(signal['indicators']['close'])
                
                # 성능 추적
                self._track_performance(signal)

            except asyncio.TimeoutError:
                # 타임아웃은 정상 (신호 대기 중)
                continue
            except Exception as e:
                self.logger.error(f"실시간 처리 오류: {e}", exc_info=True)

        # 정리
        await self.realtime_data_queue.put(None)
        await process_task

    async def _risk_management_loop(self):
        """리스크 관리 루프 (손절/익절 확인)"""
        while self.is_running:
            try:
                if self.order_manager:
                    # 현재 가격으로 리스크 관리 확인
                    if TRADING_TYPE == 'futures' and isinstance(self.order_manager, FuturesOrderManager):
                        # 선물 거래: check_risk_management(current_price) 필요
                        try:
                            # 선물 거래의 경우 get_current_price 사용 (더 안전)
                            current_price = self.order_manager.get_current_price()
                            if current_price and current_price > 0:
                                self.order_manager.check_risk_management(current_price)
                            else:
                                self.logger.warning("선물 거래 가격 조회 실패: 가격이 유효하지 않음")
                                current_price = 0.0
                        except Exception as e:
                            self.logger.error(f"선물 거래 가격 조회 실패: {e}")
                            current_price = 0.0
                    else:
                        # 스팟 거래: check_risk_management() 또는 check_risk_management(current_price)
                        try:
                            current_price = self.order_manager.get_current_price()
                            if hasattr(self.order_manager, 'check_risk_management'):
                                # check_risk_management가 current_price를 받는지 확인
                                import inspect
                                sig = inspect.signature(self.order_manager.check_risk_management)
                                if len(sig.parameters) > 0:
                                    self.order_manager.check_risk_management(current_price)
                                else:
                                    self.order_manager.check_risk_management()
                        except Exception as e:
                            self.logger.error(f"스팟 거래 리스크 관리 확인 실패: {e}")
                            current_price = 0.0
                    
                    # 대시보드 가격 업데이트
                    if self.dashboard and current_price > 0:
                        self.dashboard.update_price(current_price)
                    
                    # 실제 계정 잔액 조회 및 성능 통계 업데이트
                    await self._update_performance_from_account()
                
                # 10초마다 확인
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"리스크 관리 루프 오류: {e}")
                await asyncio.sleep(10)
    
    async def _update_performance_from_account(self):
        """실제 계정 잔액을 조회하여 성능 통계 업데이트"""
        try:
            if not self.performance_monitor or not self.order_manager:
                return
            
            # 거래 타입에 따라 잔액 조회
            if TRADING_TYPE == 'futures':
                # 선물 거래: 계정 잔액 + 미실현 손익
                try:
                    account_info = self.order_manager.api.client.futures_account()
                    total_wallet_balance = float(account_info.get('totalWalletBalance', 0))
                    available_balance = float(account_info.get('availableBalance', 0))
                    
                    # 현재 포지션의 미실현 손익 포함
                    current_position = self.order_manager.get_current_position()
                    unrealized_pnl = 0.0
                    if current_position:
                        unrealized_pnl = current_position.unrealized_pnl if hasattr(current_position, 'unrealized_pnl') else 0.0
                    
                    # 현재 자본 = 총 잔액 (미실현 손익 포함)
                    current_capital = total_wallet_balance
                    
                    # 성능 모니터 업데이트
                    self.performance_monitor.update_current_capital(current_capital)
                    
                    self.logger.debug(
                        f"성능 통계 업데이트: 자본={current_capital:.2f}, "
                        f"사용 가능={available_balance:.2f}, 미실현 손익={unrealized_pnl:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"선물 계정 잔액 조회 실패: {e}")
            else:
                # 스팟 거래: USDT 잔액 + 보유 자산 가치
                try:
                    balance = self.order_manager.get_account_balance('USDT')
                    usdt_balance = balance.get('free', 0.0) + balance.get('locked', 0.0)
                    
                    # 보유 자산 가치 계산
                    current_position = self.order_manager.get_current_position()
                    asset_value = 0.0
                    if current_position and current_position.quantity > 0:
                        current_price = self.order_manager.get_current_price()
                        asset_value = current_position.quantity * current_price
                    
                    # 현재 자본 = USDT 잔액 + 보유 자산 가치
                    current_capital = usdt_balance + asset_value
                    
                    # 성능 모니터 업데이트
                    self.performance_monitor.update_current_capital(current_capital)
                    
                    self.logger.debug(
                        f"성능 통계 업데이트: 자본={current_capital:.2f}, "
                        f"USDT={usdt_balance:.2f}, 자산 가치={asset_value:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"스팟 계정 잔액 조회 실패: {e}")
                    
        except Exception as e:
            self.logger.error(f"성능 통계 업데이트 중 오류: {e}")
    
    async def _dashboard_update_loop(self):
        """대시보드 업데이트 루프"""
        while self.is_running:
            try:
                if self.dashboard:
                    # 주기적으로 대시보드 출력 (30초마다)
                    self.dashboard.print_dashboard()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"대시보드 업데이트 오류: {e}")
                await asyncio.sleep(30)
    
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

                    # 최근 데이터로 백테스팅 (백테스팅용 어댑터 사용)
                    await self._run_incremental_backtest(self._backtest_strategy_adapter)

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
        """거래 실행 (실제 주문)

        Args:
            action: 'buy' 또는 'sell'
            signal: 거래 신호 정보
        """
        if not self.order_manager:
            self.logger.error("주문 관리자가 초기화되지 않았습니다.")
            return
        
        try:
            current_price = signal.get('indicators', {}).get('close', 0)
            if current_price == 0:
                current_price = self.order_manager.get_current_price()
            
            # 포지션 크기 계산
            if self.strategy:
                position_size_pct = self.strategy.calculate_position_size(
                    action, signal, self.total_capital
                )
            else:
                position_size_pct = self._calculate_position_size(signal)
            
            trade_record = {
                'timestamp': signal.get('timestamp', datetime.now()),
                'action': action,
                'price': current_price,
                'signal_strength': signal.get('signal_strength', 0),
                'confidence': signal.get('confidence', 0),
                'patterns': signal.get('patterns', {}),
                'position_size_pct': position_size_pct,
                'trading_type': TRADING_TYPE  # 거래 타입 추가
            }
            
            # 실제 주문 실행 (거래 타입에 따라 분기)
            if TRADING_TYPE == 'futures':
                # 선물 거래
                if isinstance(self.order_manager, FuturesOrderManager):
                    if action == 'buy':
                        # 현재 포지션 확인
                        current_position = self.order_manager.get_current_position()
                        if current_position:
                            if current_position.side == OrderSide.LONG:
                                self.logger.warning(
                                    f"이미 롱 포지션이 있습니다. 중복 매수 방지: "
                                    f"{current_position.quantity:.6f} @ {current_position.entry_price:.2f}"
                                )
                                trade_record['executed'] = False
                                trade_record['reason'] = 'already_has_long_position'
                                return
                            elif current_position.side == OrderSide.SHORT:
                                self.logger.warning(
                                    f"숏 포지션이 있습니다. 롱 포지션 오픈 전에 먼저 청산하세요: "
                                    f"{current_position.quantity:.6f} @ {current_position.entry_price:.2f}"
                                )
                                trade_record['executed'] = False
                                trade_record['reason'] = 'has_opposite_position'
                                return
                        
                        # 롱 포지션 오픈
                        position = self.order_manager.open_position(
                            side=OrderSide.LONG,
                            total_capital=self.total_capital,
                            risk_percentage=position_size_pct
                        )
                        if position:
                            trade_record['order_id'] = f"futures_long_{position.entry_time}"
                            trade_record['quantity'] = position.quantity
                            trade_record['executed'] = True
                            trade_record['entry_price'] = position.entry_price
                            self.logger.info(
                                f"롱 포지션 오픈: {position.quantity:.6f} @ {position.entry_price:.2f}"
                            )
                        else:
                            trade_record['executed'] = False
                            self.logger.error("롱 포지션 오픈 실패")
                    elif action == 'sell':
                        # 포지션 청산
                        if self.order_manager.get_current_position():
                            success = self.order_manager.close_position(reason="Signal Sell")
                            if success:
                                current_position = self.order_manager.get_current_position()
                                if current_position:
                                    trade_record['order_id'] = f"futures_close_{datetime.now()}"
                                    trade_record['quantity'] = current_position.quantity
                                    trade_record['executed'] = True
                                    trade_record['entry_price'] = current_position.entry_price
                                    trade_record['profit'] = current_position.realized_pnl / (current_position.entry_price * current_position.quantity) if current_position.entry_price > 0 else 0.0
                                    trade_record['profit_amount'] = current_position.realized_pnl
                                    self.logger.info(f"포지션 청산 완료: 실현 손익 {current_position.realized_pnl:.2f}")
                                else:
                                    trade_record['executed'] = True
                                    self.logger.info("포지션 청산 완료")
                            else:
                                trade_record['executed'] = False
                                self.logger.error("포지션 청산 실패")
                        else:
                            trade_record['executed'] = False
                            self.logger.warning("청산할 포지션이 없습니다.")
            else:
                # 스팟 거래
                if isinstance(self.order_manager, SpotOrderManager):
                    if action == 'buy':
                        # 현재 포지션 확인
                        current_position = self.order_manager.get_current_position()
                        if current_position and current_position.quantity > 0:
                            self.logger.warning(
                                f"이미 보유 중입니다. 중복 매수 방지: "
                                f"{current_position.quantity:.6f} @ {current_position.avg_price:.2f}"
                            )
                            trade_record['executed'] = False
                            trade_record['reason'] = 'already_has_position'
                            return
                        
                        # 매수 주문
                        quote_amount = self.total_capital * position_size_pct
                        order = self.order_manager.place_market_buy_order(
                            quote_amount=quote_amount,
                            current_price=current_price
                        )
                        
                        if order:
                            trade_record['order_id'] = order.client_order_id
                            trade_record['quantity'] = order.filled_quantity
                            trade_record['executed'] = True
                            self.logger.info(
                                f"매수 주문 완료: {order.filled_quantity:.6f} @ {order.avg_price:.2f}"
                            )
                        else:
                            trade_record['executed'] = False
                            self.logger.error("매수 주문 실패")
                    
                    elif action == 'sell':
                        # 매도 주문
                        current_position = self.order_manager.get_current_position()
                        if current_position and current_position.quantity > 0:
                            order = self.order_manager.place_market_sell_order(
                                quantity=current_position.quantity * position_size_pct
                            )
                            
                            if order:
                                trade_record['order_id'] = order.client_order_id
                                trade_record['quantity'] = order.filled_quantity
                                trade_record['executed'] = True
                                
                                # 수익 계산
                                if current_position.avg_price > 0:
                                    trade_record['entry_price'] = current_position.avg_price
                                    trade_record['profit'] = (
                                        (order.avg_price - current_position.avg_price) / 
                                        current_position.avg_price
                                    )
                                    trade_record['profit_amount'] = (
                                        (order.avg_price - current_position.avg_price) * 
                                        order.filled_quantity
                                    )
                                else:
                                    trade_record['entry_price'] = current_position.avg_price if current_position.avg_price > 0 else order.avg_price
                                    trade_record['profit'] = 0.0
                                    trade_record['profit_amount'] = 0.0
                                
                                self.logger.info(
                                    f"매도 주문 완료: {order.filled_quantity:.6f} @ {order.avg_price:.2f}"
                                )
                            else:
                                trade_record['executed'] = False
                                self.logger.error("매도 주문 실패")
                        else:
                            trade_record['executed'] = False
                            self.logger.warning("매도할 포지션이 없습니다.")
            
            # 거래 기록 저장
            self.trade_history.append(trade_record)
            
            # 거래 기록에 timestamp 추가 (없는 경우)
            if 'timestamp' not in trade_record:
                trade_record['timestamp'] = datetime.now()
            
            # 대시보드 업데이트
            if self.dashboard:
                self.dashboard.update_trade(trade_record)
            
            # 성능 모니터에 거래 기록 (매도 시에만 수익 기록)
            if self.performance_monitor and trade_record.get('executed', False):
                if action == 'sell' and 'profit' in trade_record:
                    # 매도 시 수익 기록
                    self.performance_monitor.record_trade({
                        'timestamp': trade_record['timestamp'],
                        'action': action,
                        'price': trade_record.get('price', current_price),
                        'quantity': trade_record.get('quantity', 0),
                        'profit': trade_record.get('profit', 0),
                        'profit_amount': trade_record.get('profit_amount', 0),
                        'trading_type': trade_record.get('trading_type', TRADING_TYPE)
                    })
                elif action == 'buy':
                    # 매수 시 거래 기록 (수익 없음)
                    self.performance_monitor.record_trade({
                        'timestamp': trade_record['timestamp'],
                        'action': action,
                        'price': trade_record.get('price', current_price),
                        'quantity': trade_record.get('quantity', 0),
                        'trading_type': trade_record.get('trading_type', TRADING_TYPE)
                    })
            
            # 성능 통계 즉시 업데이트 (거래 후)
            if self.performance_monitor:
                # 실제 계정 잔액으로 업데이트
                await self._update_performance_from_account()
            
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류: {e}", exc_info=True)

    def _calculate_position_size(self, signal: Dict) -> float:
        """포지션 크기 계산

        Args:
            signal: 거래 신호

        Returns:
            포지션 크기 (0-1)
        """
        # 신뢰도와 리스크 기반 포지션 크기 결정
        base_size = 0.1  # 기본 10%

        # 신뢰도 조정 (0-100 범위를 0-1로 정규화)
        confidence = signal.get('confidence', 50)
        confidence_multiplier = confidence / 100.0  # 0-100 -> 0-1 범위로 변환

        # 리스크 조정
        risk_multiplier = {
            'LOW': 1.2,
            'MEDIUM': 1.0,
            'HIGH': 0.5
        }.get(signal.get('risk_level', 'MEDIUM'), 1.0)

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
    
    async def _performance_based_risk_reassessment(self) -> Dict[str, Any]:
        """
        성과 기반 리스크 재평가
        
        Returns:
            리스크 평가 결과 딕셔너리
        """
        try:
            if not self.performance_monitor:
                return {'risk_level': 'MEDIUM', 'adjustment_needed': False}
            
            # 최근 성과 분석 (7일)
            recent_perf = self.performance_monitor.get_recent_performance(days=7)
            win_rate = recent_perf.get('win_rate', 50.0)
            total_profit = recent_perf.get('profit', 0.0)
            
            # 전체 통계도 확인
            stats = self.performance_monitor.get_statistics()
            total_win_rate = stats.get('win_rate', 50.0)
            max_drawdown_pct = stats.get('max_drawdown_pct', 0.0)
            
            # 리스크 레벨 결정
            risk_level = 'MEDIUM'
            adjustment_needed = False
            
            # 성과가 나쁘면 리스크 관리 강화
            if win_rate < 40.0 or total_profit < 0:
                risk_level = 'HIGH'
                adjustment_needed = True
                self.logger.warning(
                    f"성과 저조로 인한 리스크 레벨 상향: "
                    f"승률={win_rate:.1f}%, 수익={total_profit:.2f}"
                )
            elif win_rate < 50.0:
                risk_level = 'MEDIUM'
                adjustment_needed = True
            elif max_drawdown_pct > 20.0:
                risk_level = 'HIGH'
                adjustment_needed = True
                self.logger.warning(
                    f"최대 낙폭 초과로 인한 리스크 레벨 상향: "
                    f"최대 낙폭={max_drawdown_pct:.2f}%"
                )
            elif win_rate > 70.0 and total_profit > 0 and max_drawdown_pct < 10.0:
                risk_level = 'LOW'
                adjustment_needed = True
                self.logger.info(
                    f"성과 우수로 인한 리스크 레벨 하향: "
                    f"승률={win_rate:.1f}%, 수익={total_profit:.2f}"
                )
            
            return {
                'risk_level': risk_level,
                'adjustment_needed': adjustment_needed,
                'win_rate': win_rate,
                'total_win_rate': total_win_rate,
                'total_profit': total_profit,
                'max_drawdown_pct': max_drawdown_pct,
                'trading_type': TRADING_TYPE
            }
            
        except Exception as e:
            self.logger.error(f"성과 기반 리스크 재평가 중 오류: {e}", exc_info=True)
            return {'risk_level': 'MEDIUM', 'adjustment_needed': False}
    
    def _adjust_data_collection_params(self, risk_assessment: Dict[str, Any]) -> None:
        """
        리스크 평가 결과에 따라 데이터 수집 파라미터 조정
        
        Args:
            risk_assessment: 리스크 평가 결과
        """
        try:
            risk_level = risk_assessment.get('risk_level', 'MEDIUM')
            adjustment_needed = risk_assessment.get('adjustment_needed', False)
            
            if not adjustment_needed:
                return
            
            # 리스크가 높으면 더 자주 데이터 수집 (더 빠른 반응)
            if risk_level == 'HIGH' or risk_level == 'CRITICAL':
                # min_data_points 감소 (더 빠른 신호 생성)
                if hasattr(self.unified_processor, 'min_data_points'):
                    # 최소값은 30으로 제한 (너무 낮으면 신호 품질 저하)
                    new_min_points = max(self.unified_processor.min_data_points - 10, 30)
                    if new_min_points != self.unified_processor.min_data_points:
                        self.unified_processor.min_data_points = new_min_points
                        self.logger.info(
                            f"높은 리스크로 인한 데이터 수집 파라미터 조정: "
                            f"min_data_points={new_min_points}"
                        )
            
            # 리스크가 낮으면 데이터 수집 빈도 감소 (성능 최적화)
            elif risk_level == 'LOW':
                # min_data_points 증가 (더 안정적인 신호)
                if hasattr(self.unified_processor, 'min_data_points'):
                    # 최대값은 200으로 제한
                    new_min_points = min(self.unified_processor.min_data_points + 10, 200)
                    if new_min_points != self.unified_processor.min_data_points:
                        self.unified_processor.min_data_points = new_min_points
                        self.logger.info(
                            f"낮은 리스크로 인한 데이터 수집 파라미터 조정: "
                            f"min_data_points={new_min_points}"
                        )
                        
        except Exception as e:
            self.logger.error(f"데이터 수집 파라미터 조정 중 오류: {e}", exc_info=True)
    
    async def _adaptive_adjustment_loop(self):
        """
        실시간 성과 기반 자동 조정 루프
        
        성과 기록 → 리스크 재평가 → 전략 조정 파이프라인을 주기적으로 실행합니다.
        """
        self.logger.info("적응형 조정 루프 시작")
        
        while self.is_running:
            try:
                # 1시간마다 실행
                await asyncio.sleep(3600)
                
                if not self.performance_monitor or not self.strategy:
                    continue
                
                self.logger.info("성과 기반 자동 조정 실행 중...")
                
                # 1. 성과 기록 분석
                recent_perf = self.performance_monitor.get_recent_performance(days=7)
                stats = self.performance_monitor.get_statistics()
                
                # Spot/Futures별 성과 확인
                spot_stats = stats.get('spot_performance', {})
                futures_stats = stats.get('futures_performance', {})
                
                # 현재 거래 타입에 맞는 성과 사용
                if TRADING_TYPE == 'futures':
                    trading_type_perf = futures_stats
                else:
                    trading_type_perf = spot_stats
                
                win_rate = trading_type_perf.get('win_rate', stats.get('win_rate', 50.0))
                
                # 2. 리스크 재평가
                risk_assessment = await self._performance_based_risk_reassessment()
                
                # 3. 리스크 관리 → 데이터 수집 피드백
                self._adjust_data_collection_params(risk_assessment)
                
                # 4. 전략 파라미터 조정
                if hasattr(self.strategy, 'adjust_parameters'):
                    performance_feedback = {
                        'win_rate': win_rate,
                        'recent_performance': recent_perf,
                        'risk_level': risk_assessment.get('risk_level', 'MEDIUM'),
                        'trading_type': TRADING_TYPE
                    }
                    self.strategy.adjust_parameters(performance_feedback)
                    self.logger.info(
                        f"전략 파라미터 조정 완료: "
                        f"승률={win_rate:.1f}%, 리스크 레벨={risk_assessment.get('risk_level', 'MEDIUM')}"
                    )
                
                # 5. 리스크 관리 모듈 파라미터 조정 (OrderManager의 리스크 관리 모듈)
                if self.order_manager and risk_assessment.get('adjustment_needed', False):
                    await self._adjust_risk_management_params(risk_assessment)
                
            except Exception as e:
                self.logger.error(f"적응형 조정 루프 오류: {e}", exc_info=True)
                await asyncio.sleep(3600)
    
    async def _adjust_risk_management_params(self, risk_assessment: Dict[str, Any]) -> None:
        """
        리스크 관리 모듈 파라미터 조정
        
        Args:
            risk_assessment: 리스크 평가 결과
        """
        try:
            if not self.order_manager:
                return
            
            risk_level = risk_assessment.get('risk_level', 'MEDIUM')
            
            # OrderManager의 리스크 관리 모듈에 접근
            if hasattr(self.order_manager, 'stop_loss_manager'):
                stop_loss_manager = self.order_manager.stop_loss_manager
                
                if risk_level == 'HIGH' or risk_level == 'CRITICAL':
                    # 리스크가 높으면 손절 비율 감소 (더 빠른 손절)
                    if hasattr(stop_loss_manager, 'stop_loss_pct'):
                        # 현재 값의 80%로 감소 (최소 1%로 제한)
                        new_stop_loss = max(stop_loss_manager.stop_loss_pct * 0.8, 0.01)
                        if abs(new_stop_loss - stop_loss_manager.stop_loss_pct) > 0.001:
                            stop_loss_manager.stop_loss_pct = new_stop_loss
                            self.logger.info(
                                f"높은 리스크로 인한 손절 비율 조정: "
                                f"{stop_loss_manager.stop_loss_pct:.2%} -> {new_stop_loss:.2%}"
                            )
                
                elif risk_level == 'LOW':
                    # 리스크가 낮으면 손절 비율 약간 증가 (더 여유 있게)
                    if hasattr(stop_loss_manager, 'stop_loss_pct'):
                        # 현재 값의 110%로 증가 (최대 5%로 제한)
                        new_stop_loss = min(stop_loss_manager.stop_loss_pct * 1.1, 0.05)
                        if abs(new_stop_loss - stop_loss_manager.stop_loss_pct) > 0.001:
                            stop_loss_manager.stop_loss_pct = new_stop_loss
                            self.logger.info(
                                f"낮은 리스크로 인한 손절 비율 조정: "
                                f"{stop_loss_manager.stop_loss_pct:.2%} -> {new_stop_loss:.2%}"
                            )
            
            # PositionSizer 파라미터 조정
            if hasattr(self.order_manager, 'position_sizer'):
                position_sizer = self.order_manager.position_sizer
                
                if risk_level == 'HIGH' or risk_level == 'CRITICAL':
                    # 리스크가 높으면 최대 포지션 크기 감소
                    if hasattr(position_sizer, 'max_position_size_pct'):
                        new_max_size = max(position_sizer.max_position_size_pct * 0.8, 0.05)
                        if abs(new_max_size - position_sizer.max_position_size_pct) > 0.001:
                            position_sizer.max_position_size_pct = new_max_size
                            self.logger.info(
                                f"높은 리스크로 인한 최대 포지션 크기 조정: "
                                f"{position_sizer.max_position_size_pct:.2%} -> {new_max_size:.2%}"
                            )
                
                elif risk_level == 'LOW':
                    # 리스크가 낮으면 최대 포지션 크기 약간 증가
                    if hasattr(position_sizer, 'max_position_size_pct'):
                        new_max_size = min(position_sizer.max_position_size_pct * 1.1, 0.3)
                        if abs(new_max_size - position_sizer.max_position_size_pct) > 0.001:
                            position_sizer.max_position_size_pct = new_max_size
                            self.logger.info(
                                f"낮은 리스크로 인한 최대 포지션 크기 조정: "
                                f"{position_sizer.max_position_size_pct:.2%} -> {new_max_size:.2%}"
                            )
                            
        except Exception as e:
            self.logger.error(f"리스크 관리 파라미터 조정 중 오류: {e}", exc_info=True)

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
