# tests/live_futures_trading_test_with_api.py

import asyncio
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.execution.order_manager import FuturesOrderManager, PositionType
from src.config.settings import BINANCE_API_KEY, BINANCE_API_SECRET, TEST_MODE


class RealAPIFuturesTradingTestSystem:
    """실제 API를 사용한 선물 거래 테스트 시스템"""

    def __init__(self,
                 test_capital: float = 500.0,  # 매우 소액으로 설정
                 symbols: List[str] = ["BTCUSDT"],
                 max_positions: int = 1,  # 최대 1개 포지션만
                 default_leverage: int = 5):  # 낮은 레버리지
        """
        Args:
            test_capital: 테스트용 자본금 (매우 소액)
            symbols: 테스트할 심볼 목록
            max_positions: 최대 동시 포지션 수
            default_leverage: 기본 레버리지 배수
        """
        self.logger = get_logger(__name__)
        self.test_capital = test_capital
        self.symbols = symbols
        self.max_positions = max_positions
        self.default_leverage = default_leverage

        # API 키 확인
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

        # 안전 장치 - 테스트넷만 허용
        if not TEST_MODE:
            raise ValueError("실제 환경에서는 테스트를 실행할 수 없습니다. TEST_MODE=True로 설정하세요.")

        # 선물 거래 안전 장치
        self.max_daily_loss = test_capital * 0.1  # 일일 최대 손실 10%
        self.max_position_value = test_capital * 0.3  # 단일 포지션 최대 30%
        self.emergency_stop = False

        # 실제 주문 실행 여부
        self.enable_real_orders = True

        # 시스템 컴포넌트
        self.order_managers = {}

        # 거래 추적
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.start_time = datetime.now()
        self.last_signal_time = {}

        # 성과 추적
        self.performance_log = []
        self.trade_log = []
        self.api_calls_log = []

    async def initialize_system(self):
        """실제 API 연결 시스템 초기화"""
        try:
            self.logger.info("=== 실제 API 선물 거래 테스트 시스템 초기화 시작 ===")
            self.logger.warning(f"선물 테스트넷 모드: {TEST_MODE}")
            self.logger.warning(f"테스트 자본: {self.test_capital} USDT")
            self.logger.warning(f"실제 주문 실행: {self.enable_real_orders}")

            # 선물 심볼별 초기화 (실제 API 연결)
            for symbol in self.symbols:
                success = await self.initialize_symbol_with_real_api(symbol)
                if success:
                    self.logger.info(f"✅ {symbol} 실제 API 초기화 성공")
                    self.last_signal_time[symbol] = datetime.now()
                else:
                    self.logger.error(f"❌ {symbol} 실제 API 초기화 실패")
                    return False

            # 계정 정보 확인
            await self._verify_account_status()

            self.logger.info("실제 API 선물 거래 시스템 초기화 완료")
            return True

        except Exception as e:
            self.logger.error(f"실제 API 시스템 초기화 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    async def initialize_symbol_with_real_api(self, symbol: str) -> bool:
        """실제 API를 사용한 심볼 초기화"""
        try:
            # settings.py에서 자동으로 API 키를 가져오는 FuturesOrderManager 생성
            order_manager = FuturesOrderManager(
                symbol=symbol,
                initial_leverage=self.default_leverage,
                position_type=PositionType.ISOLATED
            )

            # API 연결 테스트
            if hasattr(order_manager, '_has_api_credentials') and order_manager._has_api_credentials():
                account_info = order_manager.get_account_info()
                if account_info:
                    balance = account_info.get('totalWalletBalance', 0)
                    self.logger.info(f"{symbol} API 연결 성공 - 잔고: {balance} USDT")
                else:
                    self.logger.warning(f"{symbol} API 연결되었지만 계정 정보 조회 실패")
            else:
                self.logger.error(f"{symbol} API 인증 정보가 없습니다")
                return False

            # 주문 관리자 저장
            self.order_managers[symbol] = order_manager

            return True

        except Exception as e:
            self.logger.error(f"{symbol} 실제 API 초기화 실패: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    async def _verify_account_status(self):
        """계정 상태 확인"""
        try:
            for symbol, order_manager in self.order_managers.items():
                # 계정 정보 조회
                account_info = order_manager.get_account_info()
                if account_info:
                    total_balance = float(account_info.get('totalWalletBalance', 0))
                    available_balance = float(account_info.get('availableBalance', 0))

                    self.logger.info(f"계정 상태 - 총 잔고: {total_balance} USDT, 사용 가능: {available_balance} USDT")

                    # 최소 잔고 확인
                    if available_balance < self.test_capital * 0.1:  # 테스트 자본의 10%
                        self.logger.warning(f"테스트넷 잔고가 부족합니다: {available_balance} USDT")

                # 현재 포지션 확인
                position_info = order_manager.get_position_info()
                if position_info:
                    for pos in position_info:
                        if float(pos['positionAmt']) != 0:
                            self.logger.info(f"기존 포지션 발견: {symbol} {pos['positionAmt']} @ {pos['entryPrice']}")

        except Exception as e:
            self.logger.error(f"계정 상태 확인 실패: {e}")

    async def start_real_api_futures_trading(self, duration_minutes: float = 30.0):
        """실제 API를 사용한 선물 매매 시작"""
        try:
            self.logger.info("=== 실제 API 선물 매매 테스트 시작 ===")

            # 안전 확인
            if not self._api_safety_check():
                return False

            # 초기화
            if not await self.initialize_system():
                return False

            # 모니터링 태스크 시작
            monitor_task = asyncio.create_task(self._api_monitoring_loop())
            safety_task = asyncio.create_task(self._api_safety_monitoring())

            # 실제 선물 거래 루프
            end_time = datetime.now() + timedelta(minutes=duration_minutes)

            self.logger.info(f"실제 API 선물 매매 시작 - 종료 예정: {end_time}")

            signal_count = 0
            max_signals = 3  # 최대 3번의 신호만 처리

            while datetime.now() < end_time and not self.emergency_stop and signal_count < max_signals:
                try:
                    # 각 심볼에 대해 신호 체크 및 실제 거래 실행
                    for symbol in self.symbols:
                        if signal_count >= max_signals:
                            break

                        signal_generated = await self._process_real_api_symbol_trading(symbol)
                        if signal_generated:
                            signal_count += 1
                            self.logger.info(f"신호 처리 완료 ({signal_count}/{max_signals})")

                    # 포트폴리오 상태 로깅
                    await self._log_real_api_portfolio_status()

                    # 30초 대기
                    await asyncio.sleep(30)

                except Exception as e:
                    self.logger.error(f"실제 API 거래 루프 오류: {e}")
                    await asyncio.sleep(60)

            # 시스템 종료
            await self._shutdown_real_api_system()

            # 모니터링 태스크 정리
            monitor_task.cancel()
            safety_task.cancel()

            # 최종 리포트
            self._generate_real_api_final_report()

        except Exception as e:
            self.logger.error(f"실제 API 선물 매매 시스템 오류: {e}")
            await self._emergency_real_api_shutdown()

    async def _process_real_api_symbol_trading(self, symbol: str) -> bool:
        """실제 API를 사용한 심볼별 거래 처리"""
        try:
            self.logger.info(f"=== {symbol} 거래 처리 시작 ===")

            # 신호 간격 체크 (최소 5분 간격)
            time_since_last = datetime.now() - self.last_signal_time.get(symbol, datetime.now())
            self.logger.info(f"{symbol} 마지막 신호로부터 경과 시간: {time_since_last.total_seconds()}초")

            if time_since_last.total_seconds() < 300:  # 5분
                self.logger.info(f"{symbol} 신호 간격 부족 (5분 미만), 건너뜀")
                return False

            # 현재 포지션 확인
            current_positions = await self._get_real_positions(symbol)
            has_position = len(current_positions) > 0
            self.logger.info(f"{symbol} 현재 포지션 보유 여부: {has_position}")

            # 시장 데이터 수집
            self.logger.info(f"{symbol} 시장 데이터 수집 중...")
            market_data = await self._get_real_market_data(symbol)
            if not market_data:
                self.logger.error(f"{symbol} 시장 데이터 수집 실패")
                return False

            self.logger.info(
                f"{symbol} 현재 가격: {market_data['current_price']}, 변동률: {market_data['price_change_percent']}%")

            # 신호 생성
            signal_data = await self._generate_test_trading_signal(symbol, market_data, has_position)
            if not signal_data:
                self.logger.warning(f"{symbol} 신호 생성 실패")
                return False

            self.logger.info(f"{symbol} 생성된 신호: {signal_data['signal']} (신뢰도: {signal_data['confidence']}%)")

            # 실제 거래 실행 (높은 신뢰도에서만)
            if signal_data['confidence'] >= 80:
                self.logger.info(f"{symbol} 신뢰도 조건 만족 (>= 80%), 거래 실행 시도")

                if signal_data['signal'] == 'BUY' and not has_position:
                    self.logger.info(f"{symbol} 매수 신호 실행")
                    success = await self._execute_real_futures_long_order(symbol, signal_data)
                    if success:
                        self.last_signal_time[symbol] = datetime.now()
                        return True
                elif signal_data['signal'] == 'SELL' and not has_position:
                    self.logger.info(f"{symbol} 매도 신호 실행")
                    success = await self._execute_real_futures_short_order(symbol, signal_data)
                    if success:
                        self.last_signal_time[symbol] = datetime.now()
                        return True
                elif signal_data['signal'] == 'CLOSE' and has_position:
                    self.logger.info(f"{symbol} 포지션 청산 신호 실행")
                    success = await self._close_real_futures_position(symbol, signal_data)
                    if success:
                        self.last_signal_time[symbol] = datetime.now()
                        return True
                else:
                    self.logger.info(f"{symbol} 거래 조건 불만족 - 신호: {signal_data['signal']}, 포지션보유: {has_position}")
            else:
                self.logger.info(f"{symbol} 신뢰도 부족 ({signal_data['confidence']}% < 80%), 거래 건너뜀")

            return False

        except Exception as e:
            self.logger.error(f"{symbol} 실제 API 거래 처리 오류: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    async def _get_real_positions(self, symbol: str) -> List[Dict]:
        """실제 API로 포지션 조회"""
        try:
            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                return []

            position_info = order_manager.get_position_info()
            if position_info:
                active_positions = [pos for pos in position_info if float(pos['positionAmt']) != 0]
                return active_positions

            return []

        except Exception as e:
            self.logger.error(f"{symbol} 실제 포지션 조회 오류: {e}")
            return []

    async def _get_real_market_data(self, symbol: str) -> Optional[Dict]:
        """실제 API로 시장 데이터 수집"""
        try:
            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                return None

                # 현재 가격 조회
            current_ticker = order_manager.api.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(current_ticker['price'])

            # 24시간 통계 조회 (올바른 메서드 사용)
            stats = order_manager.api.client.futures_ticker(symbol=symbol)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'volume': float(stats.get('volume', 0)),
                'price_change_percent': float(stats.get('priceChangePercent', 0)),
                'high_price': float(stats.get('highPrice', current_price)),
                'low_price': float(stats.get('lowPrice', current_price)),
                'open_price': float(stats.get('openPrice', current_price)),
                'count': int(stats.get('count', 0))  # 거래 횟수
            }

        except Exception as e:
            self.logger.error(f"{symbol} 실제 시장 데이터 수집 오류: {e}")
            # 에러 발생 시 기본값 반환
            try:
                ticker = order_manager.api.client.futures_symbol_ticker(symbol=symbol)
                return {
                    'symbol': symbol,
                    'current_price': float(ticker['price']),
                    'volume': 0,
                    'price_change_percent': 0,
                    'high_price': float(ticker['price']),
                    'low_price': float(ticker['price'])
                }
            except:
                return None

    async def _generate_test_trading_signal(self, symbol: str, market_data: Dict, has_position: bool) -> Optional[Dict]:
        """테스트용 간단한 거래 신호 생성"""
        try:
            current_price = market_data['current_price']
            price_change = market_data['price_change_percent']

            signal = 'HOLD'
            confidence = 50

            # 더 쉬운 조건으로 테스트
            if not has_position:
                if price_change > 0.1:  # 0.1% 이상 상승 (기존: 1%)
                    signal = 'BUY'
                    confidence = min(85, 80 + abs(price_change) * 2)  # 최소 85% 보장
                elif price_change < -0.1:  # 0.1% 이상 하락 (기존: 1%)
                    signal = 'SELL'
                    confidence = min(85, 80 + abs(price_change) * 2)  # 최소 85% 보장
            else:
                # 포지션이 있으면 청산 신호 (테스트용)
                if abs(price_change) > 0.05:  # 기존: 0.5%
                    signal = 'CLOSE'
                    confidence = 90

            self.logger.info(f"{symbol} 신호 생성 상세: 가격변동={price_change}%, 신호={signal}, 신뢰도={confidence}%")

            return {
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'price_change': price_change,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"{symbol} 테스트 신호 생성 오류: {e}")
            return None

    async def _execute_real_futures_long_order(self, symbol: str, signal_data: Dict) -> bool:
        """실제 API로 선물 롱 주문 실행"""
        try:
            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                return False

            # 주문 수량 계산 (매우 소액)
            current_price = signal_data['current_price']
            order_value = self.max_position_value  # 매우 소액
            quantity = order_value / current_price

            # 수량 정밀도 조정
            if hasattr(order_manager, '_adjust_quantity_precision'):
                quantity = order_manager._adjust_quantity_precision(quantity)
            else:
                quantity = round(quantity, 3)

            # 최소 주문 수량 확인
            if quantity < 0.001:
                quantity = 0.001  # 최소 수량

            self.logger.info(f"{symbol} 실제 롱 주문 실행 시도 - 수량: {quantity}, 가격: {current_price}")

            # 실제 시장가 주문 실행
            order = order_manager.api.client.futures_create_order(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=quantity
            )

            # 주문 결과 로깅
            self.api_calls_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'LONG_ORDER',
                'order_id': order['orderId'],
                'quantity': quantity,
                'price': current_price,
                'status': order['status']
            })

            self.trade_count += 1
            self.logger.info(f"✅ {symbol} 실제 롱 주문 성공: 주문ID {order['orderId']}")

            return True

        except Exception as e:
            self.logger.error(f"{symbol} 실제 롱 주문 실행 오류: {e}")
            return False

    async def _execute_real_futures_short_order(self, symbol: str, signal_data: Dict) -> bool:
        """실제 API로 선물 숏 주문 실행"""
        try:
            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                return False

            # 주문 수량 계산
            current_price = signal_data['current_price']
            order_value = self.max_position_value
            quantity = order_value / current_price

            # 수량 정밀도 조정
            if hasattr(order_manager, '_adjust_quantity_precision'):
                quantity = order_manager._adjust_quantity_precision(quantity)
            else:
                quantity = round(quantity, 3)

            if quantity < 0.001:
                quantity = 0.001

            self.logger.info(f"{symbol} 실제 숏 주문 실행 시도 - 수량: {quantity}, 가격: {current_price}")

            # 실제 시장가 주문 실행
            order = order_manager.api.client.futures_create_order(
                symbol=symbol,
                side='SELL',
                type='MARKET',
                quantity=quantity
            )

            # 주문 결과 로깅
            self.api_calls_log.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'SHORT_ORDER',
                'order_id': order['orderId'],
                'quantity': quantity,
                'price': current_price,
                'status': order['status']
            })

            self.trade_count += 1
            self.logger.info(f"✅ {symbol} 실제 숏 주문 성공: 주문ID {order['orderId']}")

            return True

        except Exception as e:
            self.logger.error(f"{symbol} 실제 숏 주문 실행 오류: {e}")
            return False

    async def _close_real_futures_position(self, symbol: str, signal_data: Dict) -> bool:
        """실제 API로 선물 포지션 청산"""
        try:
            # 현재 포지션 확인
            positions = await self._get_real_positions(symbol)
            if not positions:
                self.logger.warning(f"{symbol} 청산할 포지션이 없습니다")
                return False

            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                return False

            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt == 0:
                    continue

                # 청산 주문 (반대 방향으로 같은 수량 주문)
                side = 'SELL' if position_amt > 0 else 'BUY'
                quantity = abs(position_amt)

                self.logger.info(f"{symbol} 실제 포지션 청산 시도 - {side} {quantity}")

                order = order_manager.api.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )

                # 주문 결과 로깅
                self.api_calls_log.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'action': 'CLOSE_POSITION',
                    'order_id': order['orderId'],
                    'quantity': quantity,
                    'side': side,
                    'status': order['status']
                })

                self.trade_count += 1
                self.logger.info(f"✅ {symbol} 실제 포지션 청산 성공: 주문ID {order['orderId']}")

            return True

        except Exception as e:
            self.logger.error(f"{symbol} 실제 포지션 청산 오류: {e}")
            return False

    async def _api_monitoring_loop(self):
        """실제 API 모니터링 루프"""
        while not self.emergency_stop:
            try:
                # 실제 계정 정보 모니터링
                for symbol in self.symbols:
                    order_manager = self.order_managers.get(symbol)
                    if order_manager:
                        try:
                            # 계정 정보 조회
                            account_info = order_manager.get_account_info()
                            if account_info:
                                balance = float(account_info.get('totalWalletBalance', 0))

                                # 포지션 정보 조회
                                positions = await self._get_real_positions(symbol)

                                self.logger.debug(f"{symbol} 잔고: {balance} USDT, 포지션: {len(positions)}개")

                        except Exception as e:
                            self.logger.debug(f"{symbol} API 모니터링 오류: {e}")

                await asyncio.sleep(60)  # 1분마다 모니터링

            except Exception as e:
                self.logger.error(f"API 모니터링 루프 오류: {e}")
                await asyncio.sleep(30)

    async def _api_safety_monitoring(self):
        """실제 API 안전 모니터링"""
        while not self.emergency_stop:
            try:
                # 일일 손실 한도 체크
                if abs(self.daily_pnl) > (self.max_daily_loss / self.test_capital * 100):
                    self.logger.critical(f"🚨 실제 API 일일 손실 한도 초과: {self.daily_pnl:.2f}%")
                    await self._emergency_real_api_shutdown()
                    break

                # 거래 횟수 제한
                if self.trade_count > 10:  # 최대 10번의 거래
                    self.logger.warning(f"⚠️ 최대 거래 횟수 초과: {self.trade_count}")
                    await self._emergency_real_api_shutdown()
                    break

                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"API 안전 모니터링 오류: {e}")
                await asyncio.sleep(60)

    async def _log_real_api_portfolio_status(self):
        """실제 API 포트폴리오 상태 로깅"""
        try:
            self.logger.info("=" * 50)
            self.logger.info(f"💼 실제 API 선물 포트폴리오 현황")

            for symbol in self.symbols:
                order_manager = self.order_managers.get(symbol)
                if order_manager:
                    try:
                        # 실제 계정 정보
                        account_info = order_manager.get_account_info()
                        if account_info:
                            balance = float(account_info.get('totalWalletBalance', 0))
                            pnl = float(account_info.get('totalUnrealizedProfit', 0))

                            # 실제 포지션 정보
                            positions = await self._get_real_positions(symbol)

                            self.logger.info(f"{symbol} - 잔고: {balance:.2f} USDT, 미실현손익: {pnl:.2f} USDT")
                            self.logger.info(f"{symbol} - 활성 포지션: {len(positions)}개")

                    except Exception as e:
                        self.logger.debug(f"{symbol} 상태 조회 오류: {e}")

            self.logger.info(f"거래 횟수: {self.trade_count}")
            self.logger.info(f"API 호출 로그: {len(self.api_calls_log)}개")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"API 포트폴리오 상태 로깅 오류: {e}")

    def _api_safety_check(self) -> bool:
        """실제 API 안전 점검"""
        checks = []

        # 1. 테스트넷 확인
        if not TEST_MODE:
            self.logger.error("❌ 실제 환경에서는 테스트할 수 없습니다!")
            return False

        # 2. API 키 확인
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            self.logger.error("❌ API 키가 설정되지 않았습니다")
            return False

        # 3. 소액 확인
        if self.test_capital > 50:
            self.logger.warning(f"⚠️ 테스트 자본이 큽니다: ${self.test_capital}")
            checks.append("HIGH_CAPITAL")

        # 4. 레버리지 확인
        if self.default_leverage > 10:
            self.logger.warning(f"⚠️ 높은 레버리지: {self.default_leverage}배")
            checks.append("HIGH_LEVERAGE")

        if checks:
            self.logger.warning(f"안전 점검 경고: {checks}")
            response = input("실제 API로 선물 거래 테스트를 계속 진행하시겠습니까? (yes/no): ")
            return response.lower() == 'yes'

        return True

    async def _shutdown_real_api_system(self):
        """실제 API 시스템 정상 종료"""
        self.logger.info("실제 API 시스템 종료 중...")

        # 모든 포지션 청산 (옵션)
        for symbol in self.symbols:
            positions = await self._get_real_positions(symbol)
            if positions:
                self.logger.info(f"{symbol} 종료 시 포지션 청산")
                await self._close_real_futures_position(symbol, {'current_price': 0})

        self.emergency_stop = True
        self.logger.info("실제 API 시스템 종료 완료")

    async def _emergency_real_api_shutdown(self):
        """실제 API 긴급 종료"""
        self.logger.critical("🚨 실제 API 긴급 종료 실행")

        # 모든 포지션 즉시 청산
        for symbol in self.symbols:
            try:
                positions = await self._get_real_positions(symbol)
                if positions:
                    await self._close_real_futures_position(symbol, {'current_price': 0})
            except Exception as e:
                self.logger.error(f"긴급 청산 오류 {symbol}: {e}")

        self.emergency_stop = True
        self.logger.critical("실제 API 긴급 종료 완료")

    def _generate_real_api_final_report(self):
        """실제 API 최종 리포트 생성"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("📊 실제 API 선물 매매 테스트 최종 리포트")
            self.logger.info("=" * 60)

            # 기본 통계
            test_duration = datetime.now() - self.start_time
            self.logger.info(f"테스트 기간: {test_duration}")
            self.logger.info(f"총 거래 수: {self.trade_count}")
            self.logger.info(f"API 호출 수: {len(self.api_calls_log)}")

            # API 호출 로그 요약
            if self.api_calls_log:
                successful_orders = len([log for log in self.api_calls_log if log.get('status') == 'FILLED'])
                self.logger.info(f"성공한 주문: {successful_orders}/{len(self.api_calls_log)}")

                # 최근 주문들 표시
                self.logger.info("최근 주문 내역:")
                for log in self.api_calls_log[-5:]:  # 마지막 5개
                    self.logger.info(
                        f"  - {log['timestamp'].strftime('%H:%M:%S')} {log['action']} {log['symbol']} (주문ID: {log['order_id']})")

            # 성과 데이터 저장
            report_data = {
                'test_config': {
                    'capital': self.test_capital,
                    'symbols': self.symbols,
                    'leverage': self.default_leverage,
                    'duration': str(test_duration),
                    'testnet': TEST_MODE,
                    'real_api': True
                },
                'performance': {
                    'total_trades': self.trade_count,
                    'api_calls': len(self.api_calls_log),
                    'trade_log': self.trade_log,
                    'api_calls_log': self.api_calls_log
                }
            }

            # output 디렉토리 생성
            import os
            output_dir = './output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # JSON 저장
            with open(f'./output/real_api_futures_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                      'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"실제 API 리포트 생성 오류: {e}")


# 실행 스크립트
async def run_real_api_futures_trading_test():
    """실제 API 선물 거래 테스트 실행"""

    # 실제 API 선물 테스트 시스템 생성
    api_trading_system = RealAPIFuturesTradingTestSystem(
        test_capital=500.0,
        symbols=["BTCUSDT"],  # BTC 선물만 테스트
        max_positions=1,  # 최대 1개 포지션
        default_leverage=5  # 5배 레버리지
    )

    try:
        # 30분간 실제 API 테스트 실행
        await api_trading_system.start_real_api_futures_trading(duration_minutes=30.0)

    except KeyboardInterrupt:
        print("\n사용자가 실제 API 테스트를 중단했습니다.")
        await api_trading_system._emergency_real_api_shutdown()
    except Exception as e:
        print(f"실제 API 테스트 실행 오류: {e}")
        await api_trading_system._emergency_real_api_shutdown()


# 메인 실행부
if __name__ == "__main__":
    print("🚀 실제 API 선물 거래 통합 테스트 시스템")
    print("=" * 50)
    print("⚠️  주의: 실제 테스트넷 API를 사용합니다!")
    print("📋 테스트 설정:")
    print("   - 테스트 자본: $500")
    print(f"   - 선물 테스트넷 모드: {TEST_MODE}")
    print("   - 테스트 심볼: BTCUSDT")
    print("   - 레버리지: 5배")
    print("   - 테스트 기간: 30분")
    print("   - 최대 거래: 10회")
    print("=" * 50)

    # API 키 확인
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("❌ API 키가 설정되지 않았습니다.")
        print("📝 .env 파일에 다음을 추가하세요:")
        print("BINANCE_API_KEY=your_testnet_api_key")
        print("BINANCE_API_SECRET=your_testnet_api_secret")
        print("TEST_MODE=True")
        exit(1)

    # 테스트모드 확인
    if not TEST_MODE:
        print("❌ 테스트 모드가 아닙니다.")
        print("📝 .env 파일에서 TEST_MODE=True로 설정하세요.")
        exit(1)

    confirm = input("실제 테스트넷 API로 선물 거래 테스트를 진행하시겠습니까? (yes/no): ")
    if confirm.lower() == 'yes':
        asyncio.run(run_real_api_futures_trading_test())
    else:
        print("실제 API 테스트가 취소되었습니다.")
