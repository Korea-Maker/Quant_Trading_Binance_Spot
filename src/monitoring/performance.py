"""
성능 모니터링 모듈

이 모듈은 트레이딩 시스템의 성능을 추적하고 분석합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

from src.utils.logger import get_logger


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Args:
            initial_capital: 초기 자본
        """
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # 거래 기록
        self.trades: List[Dict] = []
        self.equity_curve: deque = deque(maxlen=10000)  # 최대 10000개 포인트
        
        # 통계
        self.total_trades = 0  # 모든 거래 수 (buy + sell)
        self.completed_trades = 0  # 완료된 거래 쌍 수 (buy-sell 쌍)
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        
        # 최대 낙폭 추적
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        
        # Spot/Futures별 성과 추적
        self.spot_performance: Dict[str, any] = {
            'trades': [],
            'total_trades': 0,
            'completed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'equity_curve': deque(maxlen=10000),
            'peak_equity': initial_capital,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
        
        self.futures_performance: Dict[str, any] = {
            'trades': [],
            'total_trades': 0,
            'completed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'equity_curve': deque(maxlen=10000),
            'peak_equity': initial_capital,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
        
        self.logger.info(f"PerformanceMonitor 초기화 완료 (초기 자본: {initial_capital:.2f})")
    
    def record_trade(self, trade: Dict):
        """
        거래 기록
        
        Args:
            trade: 거래 정보 딕셔너리
                - 'timestamp': 거래 시간
                - 'action': 'buy' 또는 'sell'
                - 'price': 거래 가격
                - 'quantity': 거래 수량
                - 'profit': 수익 (매도 시)
                - 'profit_amount': 수익 금액 (매도 시)
                - 'trading_type': 'spot' 또는 'futures' (선택적)
        """
        self.trades.append(trade)
        self.total_trades += 1
        
        # 거래 타입 확인 (기본값: spot)
        trading_type = trade.get('trading_type', 'spot')
        
        # Spot/Futures별 성과 추적
        if trading_type == 'spot':
            perf = self.spot_performance
        else:
            perf = self.futures_performance
        
        perf['trades'].append(trade)
        perf['total_trades'] += 1
        
        # 수익/손실 추적 (매도 시에만)
        if 'profit' in trade and trade['profit'] is not None:
            # 완료된 거래 쌍 (buy-sell)으로 카운트
            self.completed_trades += 1
            perf['completed_trades'] += 1
            
            profit = trade.get('profit_amount', 0.0)
            if profit > 0:
                self.winning_trades += 1
                self.total_profit += profit
                perf['winning_trades'] += 1
                perf['total_profit'] += profit
            elif profit < 0:
                self.losing_trades += 1
                self.total_loss += abs(profit)
                perf['losing_trades'] += 1
                perf['total_loss'] += abs(profit)
            
            # 자본 업데이트
            self.current_capital += profit
            
            # 거래 타입별 자산 곡선 업데이트
            self._update_trading_type_equity_curve(trading_type)
        
        # 전체 자산 곡선 업데이트
        self._update_equity_curve()
    
    def _update_equity_curve(self):
        """자산 곡선 업데이트"""
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': self.current_capital
        })
        
        # 최대 낙폭 계산
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        
        drawdown = self.peak_equity - self.current_capital
        drawdown_pct = (drawdown / self.peak_equity) * 100 if self.peak_equity > 0 else 0
        
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct
    
    def _update_trading_type_equity_curve(self, trading_type: str):
        """거래 타입별 자산 곡선 업데이트"""
        perf = self.spot_performance if trading_type == 'spot' else self.futures_performance
        
        # 현재 타입별 자본 계산 (간단한 추정)
        # 실제로는 각 타입별로 별도 자본을 추적해야 하지만,
        # 여기서는 전체 자본을 기준으로 비율로 추정
        type_capital = self.current_capital  # 간단한 추정
        
        perf['equity_curve'].append({
            'timestamp': datetime.now(),
            'equity': type_capital
        })
        
        # 최대 낙폭 계산
        if type_capital > perf['peak_equity']:
            perf['peak_equity'] = type_capital
        
        drawdown = perf['peak_equity'] - type_capital
        drawdown_pct = (drawdown / perf['peak_equity']) * 100 if perf['peak_equity'] > 0 else 0
        
        if drawdown > perf['max_drawdown']:
            perf['max_drawdown'] = drawdown
            perf['max_drawdown_pct'] = drawdown_pct
    
    def update_current_capital(self, new_capital: float):
        """
        현재 자본 업데이트 (실제 계정 잔액과 동기화)
        
        Args:
            new_capital: 새로운 자본 금액
        """
        old_capital = self.current_capital
        self.current_capital = new_capital
        
        # 자산 곡선 업데이트
        self._update_equity_curve()
        
        if abs(old_capital - new_capital) > 0.01:
            self.logger.debug(f"자본 업데이트: {old_capital:.2f} -> {new_capital:.2f}")
    
    def get_statistics(self, trading_type: Optional[str] = None) -> Dict:
        """
        성능 통계 조회
        
        Args:
            trading_type: 'spot', 'futures', 또는 None (전체)
        
        Returns:
            성능 통계 딕셔너리
        """
        # 거래 타입별 통계 요청
        if trading_type == 'spot':
            return self._get_trading_type_statistics('spot')
        elif trading_type == 'futures':
            return self._get_trading_type_statistics('futures')
        
        # 전체 통계
        # 현재 자본이 0보다 작거나 같으면 초기 자본으로 설정
        if self.current_capital <= 0:
            self.current_capital = self.initial_capital
        
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        
        # 승률은 완료된 거래 쌍(buy-sell) 기준으로 계산
        win_rate = (self.winning_trades / self.completed_trades * 100) if self.completed_trades > 0 else 0.0
        
        avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = self.total_loss / self.losing_trades if self.losing_trades > 0 else 0
        profit_factor = self.total_profit / self.total_loss if self.total_loss > 0 else 0
        
        # 샤프 비율 계산 (간단한 버전)
        sharpe_ratio = 0.0
        if len(self.equity_curve) > 1:
            equity_df = pd.DataFrame(list(self.equity_curve))
            equity_df['returns'] = equity_df['equity'].pct_change()
            if equity_df['returns'].std() > 0:
                sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return': total_return,
            'total_return_amount': self.current_capital - self.initial_capital,
            'total_trades': self.total_trades,  # 모든 거래 (buy + sell)
            'completed_trades': self.completed_trades,  # 완료된 거래 쌍 (buy-sell)
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,  # completed_trades 기준 승률
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.total_profit - self.total_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'peak_equity': self.peak_equity,
            'spot_performance': self._get_trading_type_statistics('spot'),
            'futures_performance': self._get_trading_type_statistics('futures')
        }
    
    def _get_trading_type_statistics(self, trading_type: str) -> Dict:
        """거래 타입별 통계 조회"""
        perf = self.spot_performance if trading_type == 'spot' else self.futures_performance
        
        completed_trades = perf['completed_trades']
        win_rate = (perf['winning_trades'] / completed_trades * 100) if completed_trades > 0 else 0.0
        
        avg_win = perf['total_profit'] / perf['winning_trades'] if perf['winning_trades'] > 0 else 0
        avg_loss = perf['total_loss'] / perf['losing_trades'] if perf['losing_trades'] > 0 else 0
        profit_factor = perf['total_profit'] / perf['total_loss'] if perf['total_loss'] > 0 else 0
        
        # 샤프 비율 계산
        sharpe_ratio = 0.0
        if len(perf['equity_curve']) > 1:
            equity_df = pd.DataFrame(list(perf['equity_curve']))
            equity_df['returns'] = equity_df['equity'].pct_change()
            if equity_df['returns'].std() > 0:
                sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
        
        return {
            'trading_type': trading_type,
            'total_trades': perf['total_trades'],
            'completed_trades': completed_trades,
            'winning_trades': perf['winning_trades'],
            'losing_trades': perf['losing_trades'],
            'win_rate': win_rate,
            'total_profit': perf['total_profit'],
            'total_loss': perf['total_loss'],
            'net_profit': perf['total_profit'] - perf['total_loss'],
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': perf['max_drawdown'],
            'max_drawdown_pct': perf['max_drawdown_pct'],
            'sharpe_ratio': sharpe_ratio,
            'peak_equity': perf['peak_equity']
        }
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """
        최근 성능 조회
        
        Args:
            days: 조회 기간 (일)
            
        Returns:
            최근 성능 통계
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [
            t for t in self.trades 
            if isinstance(t.get('timestamp'), datetime) and t['timestamp'] >= cutoff_date
        ]
        
        if not recent_trades:
            return {
                'period_days': days,
                'trades': 0,
                'profit': 0.0,
                'win_rate': 0.0
            }
        
        recent_profits = [
            t.get('profit_amount', 0.0) 
            for t in recent_trades 
            if 'profit_amount' in t and t.get('profit_amount') is not None
        ]
        
        winning = len([p for p in recent_profits if p > 0])
        total_profit = sum(recent_profits)
        
        # 완료된 거래만 카운트 (profit이 있는 거래)
        completed_recent_trades = len(recent_profits)
        
        return {
            'period_days': days,
            'trades': len(recent_trades),  # 모든 거래
            'completed_trades': completed_recent_trades,  # 완료된 거래 쌍
            'profit': total_profit,
            'win_rate': (winning / completed_recent_trades * 100) if completed_recent_trades > 0 else 0,
            'avg_profit': np.mean(recent_profits) if recent_profits else 0
        }
    
    def get_equity_curve_df(self) -> pd.DataFrame:
        """자산 곡선을 DataFrame으로 반환"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.equity_curve))
        df.set_index('timestamp', inplace=True)
        return df
    
    def reset(self):
        """모니터 초기화"""
        self.trades = []
        self.equity_curve.clear()
        self.current_capital = self.initial_capital
        self.peak_equity = self.initial_capital
        self.total_trades = 0
        self.completed_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
        
        # Spot/Futures별 성과 초기화
        self.spot_performance = {
            'trades': [],
            'total_trades': 0,
            'completed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'equity_curve': deque(maxlen=10000),
            'peak_equity': self.initial_capital,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
        
        self.futures_performance = {
            'trades': [],
            'total_trades': 0,
            'completed_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'equity_curve': deque(maxlen=10000),
            'peak_equity': self.initial_capital,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0
        }
        
        self.logger.info("성능 모니터 초기화 완료")

