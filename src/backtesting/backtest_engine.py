# src/backtesting/backtest_engine.py

import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.data_processing.data_interface import DataInterface
from typing import Callable, Dict

class BacktestEngine:
    """통합 백테스팅 엔진"""
    def __init__(self, initial_capital: float = 10000):
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital
        self.data_interface = DataInterface()
        self.strategies = {}
        self.results = None

    def add_strategy(self, name: str, strategy_func: Callable):
        """전략 추가"""
        self.strategies[name] = strategy_func
        self.logger.info(f"전략 '{name}' 추가완료")

    def run_backtest(self, symbol: str, interval: str, start_date: str, end_date: str, strategy_name: str) -> Dict:
        try:
            self.logger.info(f"백테스트 시작: {symbol} {interval} {start_date} ~ {end_date} ({strategy_name})")
            self.data_interface.set_data_source("historical")

            # 날짜 형식 확인 및 변환
            if not start_date.endswith("00:00:00"):
                start_date = f"{start_date} 00:00:00"
            if not end_date.endswith("23:59:59"):
                end_date = f"{end_date} 23:59:59"

            # 데이터 가져오기
            df = self.data_interface.get_data(
                symbol,
                interval,
                start_str=start_date,
                end_str=end_date
            )

            # 데이터 확인
            if df is None or df.empty:
                self.logger.error(f"데이터를 가져올 수 없습니다. Symbol: {symbol}, Interval: {interval}")
                self.logger.error(f"Period: {start_date} ~ {end_date}")
                return {}

            self.logger.info(f"데이터 로드 완료: {len(df)} 행")

            # 전략 확인
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                self.logger.error(f"전략 '{strategy_name}'를 찾을 수 없습니다.")
                return {}

            # 백테스트 실행
            results = self._simulate_trading(df, strategy, symbol)
            performance = self._calculate_performance(results)

            self.results = {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date,
                'strategy': strategy_name,
                'trades': results,
                'performance': performance
            }

            return self.results

        except Exception as e:
            self.logger.error(f"백테스트 실행 중 오류 발생: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def _simulate_trading(self, df: pd.DataFrame, strategy: Callable, symbol: str) -> pd.DataFrame:
        """거래 시뮬레이션"""
        """거래 시뮬레이션"""
        # 결과 저장용 DataFrame - 데이터 타입을 명시적으로 float로 설정
        results = df.copy()
        results['position'] = 0.0  # float로 초기화
        results['cash'] = float(self.initial_capital)  # float로 초기화
        results['holdings'] = 0.0  # float로 초기화
        results['total_value'] = float(self.initial_capital)  # float로 초기화
        results['signal'] = 0
        results['trades'] = ''

        position = 0.0  # float로 초기화
        cash = float(self.initial_capital)  # float로 초기화

        for i in range(1, len(results)):
            # 현재까지의 데이터로 전략 실행
            current_data = results.iloc[:i + 1].copy()
            signal = strategy(current_data, position)

            results.loc[results.index[i], 'signal'] = signal

            # 매수 신호
            if signal > 0 and position == 0:
                # 전액 매수
                shares = cash / results.iloc[i]['close']
                position = shares
                cash = 0.0
                results.loc[results.index[i], 'trades'] = f'BUY {shares:.4f}@{results.iloc[i]["close"]}'

            # 매도 신호
            elif signal < 0 and position > 0:
                # 전량 매도
                cash = position * results.iloc[i]['close']
                results.loc[results.index[i], 'trades'] = f'SELL {position:.4f}@{results.iloc[i]["close"]}'
                position = 0.0

            # 포지션 업데이트
            results.loc[results.index[i], 'position'] = position
            results.loc[results.index[i], 'cash'] = cash
            results.loc[results.index[i], 'holdings'] = position * results.iloc[i]['close'] if position > 0 else 0.0
            results.loc[results.index[i], 'total_value'] = cash + results.loc[results.index[i], 'holdings']

        return results

    def _calculate_performance(self, results: pd.DataFrame) -> Dict:
        """성과 지표 계산"""
        trades = results[results['trades'] != '']

        # 기본 지표
        total_return = (results.iloc[-1]['total_value'] / self.initial_capital - 1) * 100

        # 거래 통계
        num_trades = len(trades) // 2  # 매수/매도 쌍

        # 일일 수익률
        results['daily_return'] = results['total_value'].pct_change()

        # 샤프 비율 (연간화)
        sharpe_ratio = 0
        if results['daily_return'].std() > 0:
            sharpe_ratio = (results['daily_return'].mean() * 252) / (results['daily_return'].std() * np.sqrt(252))

        # 최대 낙폭
        results['cummax'] = results['total_value'].cummax()
        results['drawdown'] = (results['total_value'] / results['cummax'] - 1) * 100
        max_drawdown = results['drawdown'].min()

        # 승률 계산
        win_trades = 0
        total_trades = 0

        buy_prices = []
        for idx, row in trades.iterrows():
            if 'BUY' in row['trades']:
                buy_prices.append(row['close'])
            elif 'SELL' in row['trades'] and buy_prices:
                buy_price = buy_prices.pop(0)
                if row['close'] > buy_price:
                    win_trades += 1
                total_trades += 1

        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': results.iloc[-1]['total_value']
        }