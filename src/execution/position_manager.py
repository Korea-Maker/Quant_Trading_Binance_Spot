# src/execution/position_manager.py

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json

from src.utils.logger import get_logger
from src.execution.order_manager import FuturesOrderManager, FuturesPosition, OrderSide
from src.data_processing.unified_processor import UnifiedDataProcessor
from src.backtesting.backtest_engine import BacktestEngine
from src.execution.advanced_risk_manager import AdvancedVaRCalculator, AdvancedRiskMetrics


@dataclass
class PositionMetrics:
    """포지션 성과 지표"""
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    win_rate: float = 0.0
    avg_holding_time: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0


@dataclass
class PortfolioState:
    """포트폴리오 상태"""
    total_value: float = 0.0
    available_cash: float = 0.0
    total_margin_used: float = 0.0
    total_exposure: float = 0.0
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    positions_count: int = 0
    correlation_risk: float = 0.0


class DynamicPositionSizer:
    """동적 포지션 사이징"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.volatility_lookback = 20
        self.correlation_threshold = 0.7

    def kelly_criterion_sizing(self, win_rate: float, avg_win: float, avg_loss: float,
                               max_position_pct: float = 0.25) -> float:
        """켈리 기준 포지션 사이징"""
        try:
            if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
                return 0.05  # 기본 5%

            win_loss_ratio = abs(avg_win / avg_loss)
            kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

            # 켈리 기준을 보수적으로 조정 (절반만 사용)
            adjusted_kelly = kelly_fraction * 0.5

            # 최대 포지션 크기 제한
            return min(max(adjusted_kelly, 0.01), max_position_pct)

        except Exception as e:
            self.logger.error(f"켈리 기준 계산 오류: {e}")
            return 0.05

    def volatility_adjusted_sizing(self, current_volatility: float,
                                   base_volatility: float = 0.02,
                                   base_size: float = 0.1) -> float:
        """변동성 조정 포지션 사이징"""
        try:
            if current_volatility <= 0:
                return base_size

            volatility_ratio = base_volatility / current_volatility
            adjusted_size = base_size * volatility_ratio

            # 1% ~ 30% 사이로 제한
            return min(max(adjusted_size, 0.01), 0.30)

        except Exception as e:
            self.logger.error(f"변동성 조정 계산 오류: {e}")
            return base_size

    def correlation_adjusted_sizing(self, new_position_correlation: float,
                                    existing_positions_exposure: float,
                                    base_size: float) -> float:
        """상관관계 조정 포지션 사이징"""
        try:
            correlation_penalty = abs(new_position_correlation) * existing_positions_exposure
            adjusted_size = base_size * (1 - correlation_penalty)

            return max(adjusted_size, base_size * 0.5)  # 최소 50%는 유지

        except Exception as e:
            self.logger.error(f"상관관계 조정 계산 오류: {e}")
            return base_size


class RiskManager:
    """포트폴리오 리스크 관리"""

    def __init__(self, max_portfolio_risk: float = 0.02):
        self.logger = get_logger(__name__)
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position_risk = 0.005  # 0.5%
        self.max_correlation_exposure = 0.15  # 15%

    def calculate_portfolio_var(self, positions: Dict[str, FuturesPosition],
                                confidence_level: float = 0.05) -> float:
        """포트폴리오 VaR (Value at Risk) 계산"""
        try:
            if not positions:
                return 0.0

            # 각 포지션의 리스크 계산
            position_risks = []
            for symbol, position in positions.items():
                if position.quantity > 0:
                    # 단순화된 VaR 계산 (실제로는 더 복잡한 모델 필요)
                    position_value = position.quantity * position.entry_price
                    daily_volatility = 0.02  # 가정값, 실제로는 계산 필요
                    var = position_value * daily_volatility * 1.645  # 95% 신뢰구간
                    position_risks.append(var)

            # 포트폴리오 VaR (단순 합계, 실제로는 상관관계 고려 필요)
            return sum(position_risks)

        except Exception as e:
            self.logger.error(f"VaR 계산 오류: {e}")
            return 0.0

    def check_risk_limits(self, portfolio_state: PortfolioState,
                          new_position_size: float,
                          new_position_value: float) -> Tuple[bool, str]:
        """리스크 한도 확인"""
        try:
            # 포트폴리오 전체 리스크 확인
            total_risk_after = (portfolio_state.total_exposure + new_position_value) / portfolio_state.total_value
            if total_risk_after > self.max_portfolio_risk * 10:  # 전체 노출의 20%
                return False, "포트폴리오 전체 리스크 한도 초과"

            # 단일 포지션 리스크 확인
            single_position_risk = new_position_value / portfolio_state.total_value
            if single_position_risk > self.max_single_position_risk * 10:  # 5%
                return False, "단일 포지션 리스크 한도 초과"

            # 마진 사용률 확인
            margin_usage = (portfolio_state.total_margin_used + new_position_value) / portfolio_state.available_cash
            if margin_usage > 0.8:  # 80%
                return False, "마진 사용률 한도 초과"

            return True, "리스크 한도 내"

        except Exception as e:
            self.logger.error(f"리스크 한도 확인 오류: {e}")
            return False, f"리스크 확인 오류: {e}"


class EnhancedRiskManager(RiskManager):
    """향상된 리스크 관리자"""

    def __init__(self, max_portfolio_risk: float = 0.02):
        super().__init__(max_portfolio_risk)
        self.var_calculator = AdvancedVaRCalculator()
        self.risk_metrics = AdvancedRiskMetrics()
        self.price_data_cache = {}

    def calculate_advanced_portfolio_var(self, positions: Dict[str, any],
                                         method: str = 'historical') -> Dict[str, any]:
        """고도화된 포트폴리오 VaR 계산"""
        try:
            # 가격 데이터 수집 (캐시 활용)
            price_data = self._get_price_data_for_positions(positions)

            # VaR 계산
            var_results = self.var_calculator.calculate_portfolio_var(
                positions, price_data, method
            )

            # 컴포넌트 VaR 계산
            if price_data:
                returns_data = self.var_calculator._prepare_returns_data(positions, price_data)
                weights = self.var_calculator._calculate_portfolio_weights(positions, price_data)
                component_vars = self.var_calculator.calculate_component_var(returns_data, weights)
                var_results['component_vars'] = component_vars

            return var_results

        except Exception as e:
            self.logger.error(f"고도화된 VaR 계산 오류: {e}")
            return {}

    def _get_price_data_for_positions(self, positions: Dict[str, any]) -> Dict[str, pd.DataFrame]:
        """포지션별 가격 데이터 수집"""
        from src.data_collection.collectors import DataCollector

        price_data = {}
        collector = DataCollector()

        for symbol in positions.keys():
            try:
                # 캐시 확인
                if symbol in self.price_data_cache:
                    cache_time, data = self.price_data_cache[symbol]
                    if (pd.Timestamp.now() - cache_time).total_seconds() < 3600:  # 1시간 캐시
                        price_data[symbol] = data
                        continue

                # 새 데이터 수집 - 오류 처리 개선
                df = collector.get_historical_data(symbol, "1d", "1 year ago UTC")
                if not df.empty and len(df) > 30:  # 최소 30일 데이터 확인
                    price_data[symbol] = df
                    # 캐시 업데이트
                    self.price_data_cache[symbol] = (pd.Timestamp.now(), df)
                else:
                    self.logger.warning(f"{symbol} 데이터가 부족합니다: {len(df) if not df.empty else 0}개")

            except Exception as e:
                self.logger.warning(f"{symbol} 가격 데이터 수집 실패: {e}")

        return price_data

class AdvancedPositionManager:
    """고급 포지션 관리자 - 실시간/백테스팅 통합"""

    def __init__(self, initial_capital: float = 100000):
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital

        # 컴포넌트 초기화
        self.position_sizer = DynamicPositionSizer()
        self.risk_manager = EnhancedRiskManager()
        self.data_processor = UnifiedDataProcessor()

        # 포지션 관리
        self.active_positions: Dict[str, FuturesPosition] = {}
        self.order_managers: Dict[str, FuturesOrderManager] = {}
        self.position_history: List[FuturesPosition] = []

        # 성과 추적
        self.metrics = PositionMetrics()
        self.portfolio_state = PortfolioState(
            total_value=initial_capital,
            available_cash=initial_capital
        )

        # 실시간 데이터 버퍼
        self.price_buffer: Dict[str, deque] = {}
        self.correlation_matrix = pd.DataFrame()

        # 백테스팅 통합
        self.backtest_engine = BacktestEngine(initial_capital)
        self.backtest_results_cache: Dict[str, dict] = {}

        # 이벤트 기반 처리
        self.event_queue = asyncio.Queue()
        self.is_running = False

    async def initialize_symbol(self, symbol: str, leverage: float = 5) -> bool:
        """새 심볼 초기화"""
        try:
            # OrderManager 생성
            order_manager = FuturesOrderManager(
                symbol=symbol,
                initial_leverage=leverage
            )
            self.order_managers[symbol] = order_manager

            # 가격 버퍼 초기화
            self.price_buffer[symbol] = deque(maxlen=1000)

            self.logger.info(f"{symbol} 포지션 관리 초기화 완료")
            return True

        except Exception as e:
            self.logger.error(f"{symbol} 초기화 실패: {e}")
            return False

    async def open_position_with_analysis(self, symbol: str, side: OrderSide,
                                          signal_data: Dict,
                                          use_backtest_feedback: bool = True) -> Optional[FuturesPosition]:
        """분석 기반 포지션 오픈"""
        try:
            # 1. 백테스팅 기반 최적 사이징
            optimal_size = await self._calculate_optimal_position_size(
                symbol, side, signal_data, use_backtest_feedback
            )

            if optimal_size <= 0:
                self.logger.warning(f"{symbol} 포지션 크기가 0 이하: {optimal_size}")
                return None

            # 2. 리스크 확인
            position_value = optimal_size * signal_data.get('current_price', 0)
            risk_ok, risk_msg = self.risk_manager.check_risk_limits(
                self.portfolio_state, optimal_size, position_value
            )

            if not risk_ok:
                self.logger.warning(f"{symbol} 리스크 한도 초과: {risk_msg}")
                return None

            # 3. 실제 포지션 오픈
            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                self.logger.error(f"{symbol} OrderManager 없음")
                return None

            position = order_manager.open_position(
                side=side,
                total_capital=optimal_size * signal_data.get('current_price', 0),
                risk_percentage=0.02
            )

            if position:
                # 4. 포지션 등록 및 포트폴리오 업데이트
                self.active_positions[symbol] = position
                await self._update_portfolio_state()

                # 5. 백테스팅 피드백 업데이트
                if use_backtest_feedback:
                    await self._update_backtest_feedback(symbol, position, signal_data)

                self.logger.info(f"{symbol} 포지션 오픈 성공: {side.name} {position.quantity}")
                return position

            return None

        except Exception as e:
            self.logger.error(f"{symbol} 포지션 오픈 실패: {e}")
            return None

    async def _calculate_optimal_position_size(self, symbol: str, side: OrderSide,
                                               signal_data: Dict,
                                               use_backtest_feedback: bool) -> float:
        """최적 포지션 크기 계산"""
        try:
            base_size = 0.1  # 기본 10%

            # 1. 백테스팅 기반 사이징
            if use_backtest_feedback and symbol in self.backtest_results_cache:
                backtest_metrics = self.backtest_results_cache[symbol].get('performance', {})
                win_rate = backtest_metrics.get('win_rate', 50) / 100
                avg_win = backtest_metrics.get('avg_win', 0.02)
                avg_loss = backtest_metrics.get('avg_loss', -0.01)

                if win_rate > 0 and avg_win > 0 and avg_loss < 0:
                    kelly_size = self.position_sizer.kelly_criterion_sizing(
                        win_rate, avg_win, abs(avg_loss)
                    )
                    base_size = kelly_size

            # 2. 변동성 조정
            current_volatility = signal_data.get('volatility', 0.02)
            volatility_adjusted_size = self.position_sizer.volatility_adjusted_sizing(
                current_volatility, base_size=base_size
            )

            # 3. 상관관계 조정
            correlation = await self._calculate_symbol_correlation(symbol)
            correlation_adjusted_size = self.position_sizer.correlation_adjusted_sizing(
                correlation, self._get_existing_exposure(), volatility_adjusted_size
            )

            # 4. 신호 강도 조정
            signal_strength = signal_data.get('confidence', 50) / 100
            final_size = correlation_adjusted_size * signal_strength

            # 5. 자본 대비 실제 금액 계산
            current_price = signal_data.get('current_price', 1)
            position_value = self.portfolio_state.available_cash * final_size

            return position_value / current_price

        except Exception as e:
            self.logger.error(f"최적 포지션 크기 계산 오류: {e}")
            return 0.0

    async def _calculate_symbol_correlation(self, symbol: str) -> float:
        """심볼 간 상관관계 계산"""
        try:
            if len(self.active_positions) < 2:
                return 0.0

            # 단순화된 상관관계 계산 (실제로는 더 복잡한 계산 필요)
            # 비슷한 자산군은 높은 상관관계로 가정
            if symbol.startswith('BTC') and any(pos for pos in self.active_positions.keys() if pos.startswith('BTC')):
                return 0.8
            elif symbol.startswith('ETH') and any(pos for pos in self.active_positions.keys() if pos.startswith('ETH')):
                return 0.7

            return 0.3  # 기본 상관관계

        except Exception as e:
            self.logger.error(f"상관관계 계산 오류: {e}")
            return 0.0

    def _get_existing_exposure(self) -> float:
        """기존 포지션 노출도 계산"""
        if not self.active_positions:
            return 0.0

        total_exposure = sum(
            pos.quantity * pos.entry_price
            for pos in self.active_positions.values()
        )

        return total_exposure / self.portfolio_state.total_value if self.portfolio_state.total_value > 0 else 0.0

    async def _update_portfolio_state(self):
        """포트폴리오 상태 업데이트"""
        try:
            total_exposure = 0.0
            total_margin = 0.0
            total_unrealized_pnl = 0.0

            for symbol, position in self.active_positions.items():
                if symbol in self.order_managers:
                    order_manager = self.order_managers[symbol]
                    # 현재 가격으로 포지션 업데이트
                    try:
                        current_price = float(order_manager.api.client.futures_symbol_ticker(symbol=symbol)['price'])
                        position.update(current_price)

                        position_value = position.quantity * position.entry_price
                        total_exposure += position_value
                        total_margin += position_value / position.leverage
                        total_unrealized_pnl += position.unrealized_pnl

                    except Exception as e:
                        self.logger.debug(f"{symbol} 가격 조회 실패: {e}")

            # 포트폴리오 상태 업데이트
            self.portfolio_state.total_exposure = total_exposure
            self.portfolio_state.total_margin_used = total_margin
            self.portfolio_state.positions_count = len(self.active_positions)
            self.portfolio_state.total_value = (
                    self.initial_capital +
                    sum(pos.realized_pnl for pos in self.active_positions.values()) +
                    total_unrealized_pnl
            )
            self.portfolio_state.available_cash = self.portfolio_state.total_value - total_margin

            # 리스크 레벨 판정
            risk_ratio = total_exposure / self.portfolio_state.total_value if self.portfolio_state.total_value > 0 else 0
            if risk_ratio > 0.15:
                self.portfolio_state.risk_level = "HIGH"
            elif risk_ratio > 0.10:
                self.portfolio_state.risk_level = "MEDIUM"
            else:
                self.portfolio_state.risk_level = "LOW"

        except Exception as e:
            self.logger.error(f"포트폴리오 상태 업데이트 오류: {e}")

    async def _update_backtest_feedback(self, symbol: str, position: FuturesPosition, signal_data: Dict):
        """백테스팅 피드백 업데이트"""
        try:
            # 최근 데이터로 빠른 백테스팅 실행
            if symbol not in self.backtest_results_cache:
                # 간단한 전략으로 백테스팅
                def simple_strategy(data, pos):
                    # signal_data의 신호와 유사한 로직
                    if len(data) < 20:
                        return 0

                    current = data.iloc[-1]
                    if signal_data.get('signal') == 'BUY' and pos == 0:
                        return 1
                    elif signal_data.get('signal') == 'SELL' and pos > 0:
                        return -1
                    return 0

                # 백테스팅 실행 (최근 30일)
                self.backtest_engine.add_strategy("feedback_strategy", simple_strategy)

                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)

                results = self.backtest_engine.run_backtest(
                    symbol=symbol,
                    interval="1h",
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    strategy_name="feedback_strategy"
                )

                if results:
                    self.backtest_results_cache[symbol] = results
                    self.logger.info(f"{symbol} 백테스팅 피드백 업데이트 완료")

        except Exception as e:
            self.logger.error(f"백테스팅 피드백 업데이트 오류: {e}")

    async def close_position(self, symbol: str, reason: str = "") -> bool:
        """포지션 청산"""
        try:
            if symbol not in self.active_positions:
                self.logger.warning(f"{symbol} 활성 포지션 없음")
                return False

            order_manager = self.order_managers.get(symbol)
            if not order_manager:
                self.logger.error(f"{symbol} OrderManager 없음")
                return False

            # 포지션 청산
            success = order_manager.close_position(reason)

            if success:
                # 포지션 기록 및 제거
                closed_position = self.active_positions.pop(symbol)
                self.position_history.append(closed_position)

                # 성과 지표 업데이트
                await self._update_metrics(closed_position)
                await self._update_portfolio_state()

                self.logger.info(f"{symbol} 포지션 청산 완료: {reason}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"{symbol} 포지션 청산 실패: {e}")
            return False

    async def _update_metrics(self, closed_position: FuturesPosition):
        """성과 지표 업데이트"""
        try:
            self.metrics.total_trades += 1
            self.metrics.realized_pnl += closed_position.realized_pnl

            if closed_position.realized_pnl > 0:
                self.metrics.winning_trades += 1
                self.metrics.largest_win = max(self.metrics.largest_win, closed_position.realized_pnl)
            else:
                self.metrics.losing_trades += 1
                self.metrics.largest_loss = min(self.metrics.largest_loss, closed_position.realized_pnl)

            # 승률 계산
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades * 100

            # 전체 PnL 업데이트
            self.metrics.total_pnl = self.metrics.realized_pnl + sum(
                pos.unrealized_pnl for pos in self.active_positions.values()
            )

        except Exception as e:
            self.logger.error(f"성과 지표 업데이트 오류: {e}")

    async def monitor_positions(self):
        """포지션 모니터링 (비동기)"""
        """포지션 실시간 모니터링"""
        self.is_running = True

        while self.is_running:
            try:
                # 모든 활성 포지션 확인
                for symbol in list(self.active_positions.keys()):
                    order_manager = self.order_managers.get(symbol)
                    if order_manager:
                        # 현재 가격 조회
                        try:
                            current_price = float(
                                order_manager.api.client.futures_symbol_ticker(symbol=symbol)['price']
                            )

                            # 리스크 관리 확인
                            order_manager.check_risk_management(current_price)

                            # 포지션이 청산되었는지 확인
                            if not order_manager.get_current_position():
                                # 포지션이 청산된 경우 정리
                                if symbol in self.active_positions:
                                    closed_position = self.active_positions.pop(symbol)
                                    self.position_history.append(closed_position)
                                    await self._update_metrics(closed_position)
                                    self.logger.info(f"{symbol} 포지션 자동 청산됨")

                        except Exception as e:
                            self.logger.debug(f"{symbol} 모니터링 오류: {e}")

                # 포트폴리오 상태 업데이트
                await self._update_portfolio_state()

                # 30초마다 확인
                await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"포지션 모니터링 오류: {e}")
                await asyncio.sleep(60)

    def stop_monitoring(self):
        """모니터링 중지"""
        self.is_running = False
        self.logger.info("포지션 모니터링 중지됨")

    def get_portfolio_summary(self) -> Dict:
        """포트폴리오 요약 정보"""
        try:
            return {
                'portfolio_state': {
                    'total_value': self.portfolio_state.total_value,
                    'available_cash': self.portfolio_state.available_cash,
                    'total_exposure': self.portfolio_state.total_exposure,
                    'positions_count': self.portfolio_state.positions_count,
                    'risk_level': self.portfolio_state.risk_level
                },
                'performance_metrics': {
                    'total_pnl': self.metrics.total_pnl,
                    'realized_pnl': self.metrics.realized_pnl,
                    'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.active_positions.values()),
                    'win_rate': self.metrics.win_rate,
                    'total_trades': self.metrics.total_trades,
                    'roi': (self.metrics.total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
                },
                'active_positions': {
                    symbol: {
                        'side': pos.side.name,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'leverage': pos.leverage
                    }
                    for symbol, pos in self.active_positions.items()
                }
            }
        except Exception as e:
            self.logger.error(f"포트폴리오 요약 생성 오류: {e}")
            return {}


# 사용 예시
# 사용 예시에서 고도화된 VaR 계산 테스트 추가
async def main():
    """포지션 관리자 사용 예시"""
    # 포지션 관리자 생성
    position_manager = AdvancedPositionManager(initial_capital=100000)

    # 심볼 초기화
    await position_manager.initialize_symbol("BTCUSDT", leverage=5)
    await position_manager.initialize_symbol("ETHUSDT", leverage=3)

    # 모니터링 시작
    monitor_task = asyncio.create_task(position_manager.monitor_positions())

    try:
        # 포지션 오픈 예시
        signal_data = {
            'signal': 'BUY',
            'confidence': 75,
            'current_price': 50000,
            'volatility': 0.025
        }

        position = await position_manager.open_position_with_analysis(
            symbol="BTCUSDT",
            side=OrderSide.LONG,
            signal_data=signal_data,
            use_backtest_feedback=True
        )

        if position:
            print(f"포지션 오픈 성공: {position}")

        # 고도화된 VaR 계산 테스트 추가
        if position_manager.active_positions:
            var_results = position_manager.risk_manager.calculate_advanced_portfolio_var(
                position_manager.active_positions,
                method='historical'
            )
            print(f"\n고도화된 VaR 계산 결과:")
            print(json.dumps(var_results, indent=2, ensure_ascii=False))

        # 포트폴리오 상태 확인
        summary = position_manager.get_portfolio_summary()
        print(f"\n포트폴리오 요약:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        # 잠시 대기
        await asyncio.sleep(60)

    finally:
        position_manager.stop_monitoring()
        monitor_task.cancel()



if __name__ == "__main__":
    asyncio.run(main())
