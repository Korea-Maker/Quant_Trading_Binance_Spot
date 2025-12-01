#!/usr/bin/env python3
"""
통합 테스트 스크립트

이 스크립트는 전체 트레이딩 시스템의 통합 테스트를 수행합니다.
실제 API 호출 없이 모듈 초기화 및 기본 기능을 테스트합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from datetime import datetime
from typing import Dict, Any

from src.utils.logger import get_logger
from src.config.settings import TRADING_TYPE, TEST_MODE
from src.integration.realtime_backtest_integration import RealtimeBacktestIntegration
from src.strategy.signals import SignalBasedStrategy
from src.monitoring.performance import PerformanceMonitor
from src.risk_management import (
    create_stop_loss_manager,
    create_position_sizer,
    ExposureManager
)
from src.execution.order_manager import FuturesOrderManager
from src.execution.spot_order_manager import SpotOrderManager
from src.data_processing.unified_processor import UnifiedDataProcessor

logger = get_logger(__name__)


class IntegrationTest:
    """통합 테스트 클래스"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results: Dict[str, bool] = {}
        self.errors: list = []
        
    def log_test(self, test_name: str, passed: bool, error: str = None):
        """테스트 결과 기록"""
        self.test_results[test_name] = passed
        if passed:
            self.logger.info(f"✅ {test_name}: PASSED")
        else:
            self.logger.error(f"❌ {test_name}: FAILED")
            if error:
                self.logger.error(f"   오류: {error}")
                self.errors.append(f"{test_name}: {error}")
    
    def test_module_imports(self) -> bool:
        """모듈 import 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("모듈 Import 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            # 리스크 관리 모듈
            from src.risk_management import (
                create_stop_loss_manager,
                create_position_sizer,
                ExposureManager
            )
            self.log_test("리스크 관리 모듈 import", True)
            
            # 실행 모듈
            from src.execution.order_manager import FuturesOrderManager
            from src.execution.spot_order_manager import SpotOrderManager
            self.log_test("실행 모듈 import", True)
            
            # 데이터 처리 모듈
            from src.data_processing.unified_processor import UnifiedDataProcessor
            self.log_test("데이터 처리 모듈 import", True)
            
            # 통합 모듈
            from src.integration.realtime_backtest_integration import RealtimeBacktestIntegration
            self.log_test("통합 모듈 import", True)
            
            # 전략 모듈
            from src.strategy.signals import SignalBasedStrategy
            self.log_test("전략 모듈 import", True)
            
            # 모니터링 모듈
            from src.monitoring.performance import PerformanceMonitor
            self.log_test("모니터링 모듈 import", True)
            
            return True
            
        except Exception as e:
            self.log_test("모듈 import", False, str(e))
            return False
    
    def test_risk_management_modules(self) -> bool:
        """리스크 관리 모듈 초기화 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("리스크 관리 모듈 초기화 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            # StopLossManager 테스트
            stop_loss_manager = create_stop_loss_manager(
                trading_type=TRADING_TYPE,
                symbol="BTCUSDT",
                leverage=5.0 if TRADING_TYPE == 'futures' else None
            )
            self.log_test(f"{TRADING_TYPE} StopLossManager 초기화", True)
            
            # PositionSizer 테스트
            position_sizer = create_position_sizer(
                trading_type=TRADING_TYPE,
                symbol="BTCUSDT",
                leverage=5.0 if TRADING_TYPE == 'futures' else None
            )
            self.log_test(f"{TRADING_TYPE} PositionSizer 초기화", True)
            
            # ExposureManager 테스트
            exposure_manager = ExposureManager()
            exposure_manager.set_total_capital(10000.0)
            self.log_test("ExposureManager 초기화", True)
            
            return True
            
        except Exception as e:
            self.log_test("리스크 관리 모듈 초기화", False, str(e))
            return False
    
    def test_order_manager_initialization(self) -> bool:
        """OrderManager 초기화 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("OrderManager 초기화 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            exposure_manager = ExposureManager()
            exposure_manager.set_total_capital(10000.0)
            
            if TRADING_TYPE == 'futures':
                order_manager = FuturesOrderManager(
                    symbol="BTCUSDT",
                    initial_leverage=5.0,
                    exposure_manager=exposure_manager
                )
                self.log_test("FuturesOrderManager 초기화 (ExposureManager 연동)", True)
                
                # 리스크 관리 모듈 확인
                if hasattr(order_manager, 'stop_loss_manager'):
                    self.log_test("FuturesOrderManager.stop_loss_manager 존재", True)
                else:
                    self.log_test("FuturesOrderManager.stop_loss_manager 존재", False, "속성이 없음")
                
                if hasattr(order_manager, 'position_sizer'):
                    self.log_test("FuturesOrderManager.position_sizer 존재", True)
                else:
                    self.log_test("FuturesOrderManager.position_sizer 존재", False, "속성이 없음")
                
                if hasattr(order_manager, 'exposure_manager'):
                    self.log_test("FuturesOrderManager.exposure_manager 존재", True)
                else:
                    self.log_test("FuturesOrderManager.exposure_manager 존재", False, "속성이 없음")
            else:
                order_manager = SpotOrderManager(
                    symbol="BTCUSDT",
                    exposure_manager=exposure_manager
                )
                self.log_test("SpotOrderManager 초기화 (ExposureManager 연동)", True)
                
                # 리스크 관리 모듈 확인
                if hasattr(order_manager, 'stop_loss_manager'):
                    self.log_test("SpotOrderManager.stop_loss_manager 존재", True)
                else:
                    self.log_test("SpotOrderManager.stop_loss_manager 존재", False, "속성이 없음")
                
                if hasattr(order_manager, 'position_sizer'):
                    self.log_test("SpotOrderManager.position_sizer 존재", True)
                else:
                    self.log_test("SpotOrderManager.position_sizer 존재", False, "속성이 없음")
                
                if hasattr(order_manager, 'exposure_manager'):
                    self.log_test("SpotOrderManager.exposure_manager 존재", True)
                else:
                    self.log_test("SpotOrderManager.exposure_manager 존재", False, "속성이 없음")
            
            return True
            
        except Exception as e:
            self.log_test("OrderManager 초기화", False, str(e))
            return False
    
    def test_performance_monitor(self) -> bool:
        """PerformanceMonitor 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("PerformanceMonitor 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            monitor = PerformanceMonitor(initial_capital=10000.0)
            self.log_test("PerformanceMonitor 초기화", True)
            
            # Spot/Futures별 성과 추적 확인
            if hasattr(monitor, 'spot_performance'):
                self.log_test("PerformanceMonitor.spot_performance 존재", True)
            else:
                self.log_test("PerformanceMonitor.spot_performance 존재", False, "속성이 없음")
            
            if hasattr(monitor, 'futures_performance'):
                self.log_test("PerformanceMonitor.futures_performance 존재", True)
            else:
                self.log_test("PerformanceMonitor.futures_performance 존재", False, "속성이 없음")
            
            # 거래 기록 테스트
            monitor.record_trade({
                'timestamp': datetime.now(),
                'action': 'buy',
                'price': 50000.0,
                'quantity': 0.001,
                'trading_type': TRADING_TYPE
            })
            self.log_test("PerformanceMonitor.record_trade() (매수)", True)
            
            monitor.record_trade({
                'timestamp': datetime.now(),
                'action': 'sell',
                'price': 51000.0,
                'quantity': 0.001,
                'profit': 0.02,
                'profit_amount': 1.0,
                'trading_type': TRADING_TYPE
            })
            self.log_test("PerformanceMonitor.record_trade() (매도)", True)
            
            # 통계 조회 테스트
            stats = monitor.get_statistics()
            self.log_test("PerformanceMonitor.get_statistics() (전체)", True)
            
            stats_spot = monitor.get_statistics(trading_type='spot')
            self.log_test("PerformanceMonitor.get_statistics() (Spot)", True)
            
            stats_futures = monitor.get_statistics(trading_type='futures')
            self.log_test("PerformanceMonitor.get_statistics() (Futures)", True)
            
            return True
            
        except Exception as e:
            self.log_test("PerformanceMonitor 테스트", False, str(e))
            return False
    
    def test_strategy_adjust_parameters(self) -> bool:
        """전략 파라미터 조정 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("전략 파라미터 조정 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            strategy = SignalBasedStrategy(
                min_confidence=60.0,
                min_signal_strength=1.0
            )
            self.log_test("SignalBasedStrategy 초기화", True)
            
            # adjust_parameters 메서드 확인
            if hasattr(strategy, 'adjust_parameters'):
                self.log_test("SignalBasedStrategy.adjust_parameters() 존재", True)
                
                # 파라미터 조정 테스트
                initial_confidence = strategy.min_confidence
                strategy.adjust_parameters({
                    'win_rate': 30.0,  # 낮은 승률
                    'risk_level': 'HIGH',
                    'trading_type': TRADING_TYPE
                })
                
                # 파라미터가 조정되었는지 확인
                if strategy.min_confidence > initial_confidence:
                    self.log_test("전략 파라미터 조정 (낮은 승률)", True)
                else:
                    self.log_test("전략 파라미터 조정 (낮은 승률)", False, "파라미터가 조정되지 않음")
            else:
                self.log_test("SignalBasedStrategy.adjust_parameters() 존재", False, "메서드가 없음")
            
            return True
            
        except Exception as e:
            self.log_test("전략 파라미터 조정 테스트", False, str(e))
            return False
    
    def test_integration_initialization(self) -> bool:
        """통합 시스템 초기화 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("통합 시스템 초기화 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            integration = RealtimeBacktestIntegration(
                symbol="BTCUSDT",
                interval="1m",
                backtest_lookback_days=7,
                feedback_update_hours=1
            )
            self.log_test("RealtimeBacktestIntegration 초기화", True)
            
            # ExposureManager 확인
            if hasattr(integration, 'exposure_manager'):
                self.log_test("RealtimeBacktestIntegration.exposure_manager 존재", True)
            else:
                self.log_test("RealtimeBacktestIntegration.exposure_manager 존재", False, "속성이 없음")
            
            # 피드백 루프 메서드 확인
            if hasattr(integration, '_performance_based_risk_reassessment'):
                self.log_test("_performance_based_risk_reassessment() 존재", True)
            else:
                self.log_test("_performance_based_risk_reassessment() 존재", False, "메서드가 없음")
            
            if hasattr(integration, '_adjust_data_collection_params'):
                self.log_test("_adjust_data_collection_params() 존재", True)
            else:
                self.log_test("_adjust_data_collection_params() 존재", False, "메서드가 없음")
            
            if hasattr(integration, '_adaptive_adjustment_loop'):
                self.log_test("_adaptive_adjustment_loop() 존재", True)
            else:
                self.log_test("_adaptive_adjustment_loop() 존재", False, "메서드가 없음")
            
            if hasattr(integration, '_adjust_risk_management_params'):
                self.log_test("_adjust_risk_management_params() 존재", True)
            else:
                self.log_test("_adjust_risk_management_params() 존재", False, "메서드가 없음")
            
            return True
            
        except Exception as e:
            self.log_test("통합 시스템 초기화", False, str(e))
            return False
    
    def test_unified_processor_risk_check(self) -> bool:
        """UnifiedDataProcessor 리스크 체크 통합 테스트"""
        self.logger.info("=" * 60)
        self.logger.info("UnifiedDataProcessor 리스크 체크 통합 테스트 시작")
        self.logger.info("=" * 60)
        
        try:
            processor = UnifiedDataProcessor(
                buffer_size=1000,
                min_data_points=50
            )
            self.log_test("UnifiedDataProcessor 초기화", True)
            
            # IntegratedRiskChecker 확인
            if hasattr(processor, 'risk_checker'):
                self.log_test("UnifiedDataProcessor.risk_checker 존재", True)
            else:
                self.log_test("UnifiedDataProcessor.risk_checker 존재", False, "속성이 없음")
            
            # 리스크 체크 메서드 확인
            if hasattr(processor, '_prepare_risk_check_data'):
                self.log_test("_prepare_risk_check_data() 존재", True)
            else:
                self.log_test("_prepare_risk_check_data() 존재", False, "메서드가 없음")
            
            if hasattr(processor, '_handle_risk_check_result'):
                self.log_test("_handle_risk_check_result() 존재", True)
            else:
                self.log_test("_handle_risk_check_result() 존재", False, "메서드가 없음")
            
            return True
            
        except Exception as e:
            self.log_test("UnifiedDataProcessor 리스크 체크 통합", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """모든 테스트 실행"""
        self.logger.info("=" * 60)
        self.logger.info("통합 테스트 시작")
        self.logger.info("=" * 60)
        self.logger.info(f"거래 타입: {TRADING_TYPE}")
        self.logger.info(f"테스트 모드: {TEST_MODE}")
        self.logger.info("=" * 60)
        
        tests = [
            ("모듈 Import", self.test_module_imports),
            ("리스크 관리 모듈 초기화", self.test_risk_management_modules),
            ("OrderManager 초기화", self.test_order_manager_initialization),
            ("PerformanceMonitor", self.test_performance_monitor),
            ("전략 파라미터 조정", self.test_strategy_adjust_parameters),
            ("통합 시스템 초기화", self.test_integration_initialization),
            ("UnifiedDataProcessor 리스크 체크", self.test_unified_processor_risk_check),
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            try:
                result = test_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.log_test(test_name, False, str(e))
                all_passed = False
        
        # 결과 요약
        self.logger.info("=" * 60)
        self.logger.info("테스트 결과 요약")
        self.logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for v in self.test_results.values() if v)
        failed_tests = total_tests - passed_tests
        
        self.logger.info(f"총 테스트: {total_tests}")
        self.logger.info(f"통과: {passed_tests}")
        self.logger.info(f"실패: {failed_tests}")
        
        if self.errors:
            self.logger.error("=" * 60)
            self.logger.error("오류 목록:")
            for error in self.errors:
                self.logger.error(f"  - {error}")
        
        self.logger.info("=" * 60)
        
        if all_passed:
            self.logger.info("✅ 모든 테스트 통과!")
        else:
            self.logger.warning("⚠️ 일부 테스트 실패")
        
        return all_passed


def main():
    """메인 함수"""
    logger.info("통합 테스트 시작")
    
    test = IntegrationTest()
    success = test.run_all_tests()
    
    if success:
        logger.info("통합 테스트 완료: 모든 테스트 통과")
        return 0
    else:
        logger.error("통합 테스트 완료: 일부 테스트 실패")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

