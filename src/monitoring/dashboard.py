"""
모니터링 대시보드

이 모듈은 트레이딩 시스템의 실시간 상태를 모니터링하고 표시합니다.
"""

import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from collections import deque

from src.utils.logger import get_logger
from src.monitoring.performance import PerformanceMonitor


class TradingDashboard:
    """트레이딩 대시보드"""
    
    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        """
        Args:
            performance_monitor: 성능 모니터 객체
        """
        self.logger = get_logger(__name__)
        self.performance_monitor = performance_monitor
        
        # 실시간 상태
        self.current_price: float = 0.0
        self.current_signal: Optional[Dict] = None
        self.last_trade: Optional[Dict] = None
        self.system_status: str = "STOPPED"
        
        # 알림 큐
        self.notifications: deque = deque(maxlen=100)
        
        self.logger.info("TradingDashboard 초기화 완료")
    
    def update_price(self, price: float):
        """현재 가격 업데이트"""
        self.current_price = price
    
    def update_signal(self, signal: Dict):
        """현재 신호 업데이트"""
        self.current_signal = signal
        self._check_signal_notification(signal)
    
    def update_trade(self, trade: Dict):
        """거래 정보 업데이트"""
        self.last_trade = trade
        if self.performance_monitor:
            self.performance_monitor.record_trade(trade)
        self._add_notification(f"거래 실행: {trade.get('action', 'unknown')} @ {trade.get('price', 0):.2f}")
    
    def update_status(self, status: str):
        """시스템 상태 업데이트"""
        self.system_status = status
        self._add_notification(f"시스템 상태: {status}")
    
    def _check_signal_notification(self, signal: Dict):
        """신호 알림 확인"""
        primary_signal = signal.get('primary_signal', 'HOLD')
        confidence = signal.get('confidence', 0)
        
        if primary_signal != 'HOLD' and confidence >= 70:
            self._add_notification(
                f"강한 신호: {primary_signal} (신뢰도: {confidence:.1f}%)"
            )
    
    def _add_notification(self, message: str, level: str = "INFO"):
        """알림 추가"""
        notification = {
            'timestamp': datetime.now(),
            'message': message,
            'level': level
        }
        self.notifications.append(notification)
        self.logger.info(f"[{level}] {message}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        대시보드 데이터 조회
        
        Returns:
            대시보드 데이터 딕셔너리
        """
        data = {
            'system_status': self.system_status,
            'current_price': self.current_price,
            'current_signal': self.current_signal,
            'last_trade': self.last_trade,
            'notifications': list(self.notifications)[-10:],  # 최근 10개
        }
        
        # 성능 통계 추가
        if self.performance_monitor:
            data['performance'] = self.performance_monitor.get_statistics()
            data['recent_performance'] = self.performance_monitor.get_recent_performance(days=7)
        
        return data
    
    def print_dashboard(self):
        """대시보드를 콘솔에 출력"""
        print("\n" + "=" * 60)
        print("트레이딩 대시보드")
        print("=" * 60)
        print(f"시스템 상태: {self.system_status}")
        print(f"현재 가격: {self.current_price:.2f}")
        
        if self.current_signal:
            signal = self.current_signal
            print(f"\n현재 신호:")
            print(f"  - 신호: {signal.get('primary_signal', 'HOLD')}")
            print(f"  - 신뢰도: {signal.get('confidence', 0):.1f}%")
            print(f"  - 강도: {signal.get('signal_strength', 0):.2f}")
        
        if self.last_trade:
            trade = self.last_trade
            print(f"\n최근 거래:")
            print(f"  - 액션: {trade.get('action', 'unknown')}")
            print(f"  - 가격: {trade.get('price', 0):.2f}")
            if 'profit' in trade and trade['profit'] is not None:
                print(f"  - 수익률: {trade['profit']*100:.2f}%")
        
        if self.performance_monitor:
            stats = self.performance_monitor.get_statistics()
            print(f"\n성능 통계:")
            print(f"  - 총 수익률: {stats['total_return']:.2f}%")
            print(f"  - 현재 자본: {stats['current_capital']:.2f}")
            print(f"  - 총 거래: {stats['total_trades']}")
            print(f"  - 승률: {stats['win_rate']:.2f}%")
            print(f"  - 최대 낙폭: {stats['max_drawdown_pct']:.2f}%")
        
        print("=" * 60 + "\n")
    
    async def start_monitoring(self, update_interval: int = 5):
        """
        모니터링 시작
        
        Args:
            update_interval: 업데이트 간격 (초)
        """
        self.logger.info("대시보드 모니터링 시작")
        
        while self.system_status == "RUNNING":
            try:
                self.print_dashboard()
                await asyncio.sleep(update_interval)
            except Exception as e:
                self.logger.error(f"모니터링 중 오류: {e}")
                await asyncio.sleep(update_interval)
    
    def get_notifications(self, limit: int = 10) -> list:
        """최근 알림 조회"""
        return list(self.notifications)[-limit:]

