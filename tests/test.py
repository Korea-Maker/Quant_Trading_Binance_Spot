# test_integration.py

import asyncio
import sys
import os
import json
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.unified_processor import UnifiedDataProcessor, BacktestRealtimeAdapter
from src.integration.realtime_backtest_integration import RealtimeBacktestIntegration, adaptive_strategy
from src.data_collection.collectors import DataCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_unified_processor():
    """통합 데이터 프로세서 테스트"""
    print("\n" + "=" * 50)
    print("1. 통합 데이터 프로세서 테스트")
    print("=" * 50)

    try:
        # 프로세서 생성
        processor = UnifiedDataProcessor(enable_ml_features=True)
        logger.info("UnifiedDataProcessor 생성 완료")

        # 데이터 수집
        collector = DataCollector()
        print("\n과거 데이터 수집 중...")
        historical_data = collector.get_historical_data(
            symbol="BTCUSDT",
            interval="1h",
            start_str="3 days ago UTC"
        )
        print(f"수집된 데이터: {historical_data.shape}")

        # 백테스팅 모드 처리
        print("\n백테스팅 모드 처리 중...")
        processed_data = processor.process_data(historical_data)
        print(f"처리된 데이터 shape: {processed_data.shape}")
        print(f"생성된 컬럼 수: {len(processed_data.columns)}")

        # 신호 생성
        signals = processor.generate_signals(processed_data)
        print(f"\n생성된 신호:")
        print(json.dumps(signals, indent=2, default=str))

        # 실시간 모드 시뮬레이션 - 수정된 부분
        print("\n\n실시간 모드 시뮬레이션...")

        # 버퍼 초기화를 위해 일부 데이터를 먼저 처리
        initial_data = historical_data.iloc[:processor.min_data_points]
        processor.process_data(initial_data)

        # 이후 데이터로 실시간 시뮬레이션
        for i in range(processor.min_data_points, min(processor.min_data_points + 5, len(historical_data))):
            simulated_realtime_data = historical_data.iloc[i].to_dict()
            # _process_streaming_data 메서드 직접 호출
            processed_realtime = processor._process_streaming_data(simulated_realtime_data)

            if processed_realtime is not None:
                print(f"\n시뮬레이션 {i - processor.min_data_points + 1}:")
                if 'signals' in processed_realtime:
                    rt_signals = processed_realtime['signals']
                    print(f"신호: {rt_signals.get('primary_signal', 'N/A')}, "
                          f"신뢰도: {rt_signals.get('confidence', 0):.2f}")
                else:
                    print("신호 없음")

        return True, "통합 데이터 프로세서 테스트 성공"

    except Exception as e:
        logger.error(f"프로세서 테스트 실패: {e}")
        return False, str(e)


async def test_backtest_adapter():
    """백테스트 어댑터 테스트"""
    print("\n" + "=" * 50)
    print("2. 백테스트 어댑터 테스트")
    print("=" * 50)

    try:
        processor = UnifiedDataProcessor()
        adapter = BacktestRealtimeAdapter(processor)

        # 모의 백테스팅 결과
        mock_backtest_results = {
            'trades': [
                {'signal_type': 'BUY_SIGNAL', 'profit': 0.02},
                {'signal_type': 'BUY_SIGNAL', 'profit': -0.01},
                {'signal_type': 'SELL_SIGNAL', 'profit': 0.015},
                {'signal_type': 'SELL_SIGNAL', 'profit': 0.03},
            ],
            'pattern_analysis': {
                'PATTERN_BULLISH_ENGULFING': {'success_rate': 0.65},
                'PATTERN_BEARISH_HARAMI': {'success_rate': 0.55}
            },
            'optimization_results': {
                'RSI_14': {'optimal_upper': 75, 'optimal_lower': 25}
            }
        }

        # 피드백 추출
        feedback = adapter.extract_feedback_from_backtest(mock_backtest_results)
        print("\n추출된 피드백:")
        print(json.dumps(feedback, indent=2))

        # 피드백 적용
        processor.update_feedback(feedback)
        print("\n피드백이 프로세서에 적용되었습니다.")

        return True, "백테스트 어댑터 테스트 성공"

    except Exception as e:
        logger.error(f"어댑터 테스트 실패: {e}")
        return False, str(e)


async def test_integration_system():
    """통합 시스템 간단 테스트"""
    print("\n" + "=" * 50)
    print("3. 실시간 백테스팅 통합 시스템 테스트")
    print("=" * 50)

    try:
        # 통합 시스템 생성
        integration = RealtimeBacktestIntegration(
            symbol="BTCUSDT",
            interval="1m",
            backtest_lookback_days=1,  # 빠른 테스트를 위해 1일로 설정
            feedback_update_hours=24
        )

        print("\n통합 시스템 초기화 완료")
        print(f"심볼: {integration.symbol}")
        print(f"간격: {integration.interval}")
        print(f"백테스팅 기간: {integration.backtest_lookback_days}일")

        # 초기 백테스팅만 실행
        print("\n초기 백테스팅 실행 중...")
        await integration._initial_backtest_and_feedback(adaptive_strategy)

        # 성능 요약
        summary = integration.get_performance_summary()
        print("\n초기 성능 요약:")
        print(json.dumps(summary, indent=2, default=str))

        # 실시간 처리 시뮬레이션 (짧은 시간)
        print("\n실시간 처리 시뮬레이션 (10초)...")

        # 시뮬레이션을 위한 태스크
        try:
            await asyncio.wait_for(
                integration.start(adaptive_strategy),
                timeout=10  # 10초만 실행
            )
        except asyncio.TimeoutError:
            print("시뮬레이션 시간 초과 - 정상 종료")

        # 최종 성능 요약
        final_summary = integration.get_performance_summary()
        print("\n최종 성능 요약:")
        print(json.dumps(final_summary, indent=2, default=str))

        # 시스템 중지
        await integration.stop()

        return True, "통합 시스템 테스트 성공"

    except Exception as e:
        logger.error(f"통합 시스템 테스트 실패: {e}")
        return False, str(e)


async def run_all_tests():
    """모든 테스트 실행"""
    print("\n암호화폐 트레이딩 시스템 통합 테스트")
    print("=" * 50)
    print(f"테스트 시작 시간: {datetime.now()}")

    tests = [
        ("통합 데이터 프로세서", test_unified_processor),
        ("백테스트 어댑터", test_backtest_adapter),
        ("통합 시스템", test_integration_system)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success, message = await test_func()
            results.append((test_name, success, message))
        except Exception as e:
            results.append((test_name, False, str(e)))

    # 테스트 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)

    for test_name, success, message in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{test_name}: {status}")
        if not success:
            print(f"  오류: {message}")

    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    print(f"\n총 테스트: {total_tests}, 성공: {passed_tests}, 실패: {total_tests - passed_tests}")

    return all(success for _, success, _ in results)


async def quick_test():
    """빠른 기능 테스트"""
    print("\n빠른 기능 테스트 모드")
    print("=" * 50)

    try:
        # 1. 데이터 수집 테스트
        print("1. 데이터 수집 테스트...")
        collector = DataCollector()
        data = collector.get_historical_data("BTCUSDT", "1h", "1 day ago UTC")
        print(f"   ✅ 데이터 수집 성공: {data.shape}")

        # 2. 프로세서 생성 테스트
        print("\n2. 프로세서 생성 테스트...")
        processor = UnifiedDataProcessor()
        print("   ✅ 프로세서 생성 성공")

        # 3. 데이터 처리 테스트
        print("\n3. 데이터 처리 테스트...")
        processed = processor.process_data(data.head(100))  # 처음 100개만
        print(f"   ✅ 데이터 처리 성공: {processed.shape}")

        # 4. 신호 생성 테스트
        print("\n4. 신호 생성 테스트...")
        signals = processor.generate_signals(processed)
        print(f"   ✅ 신호 생성 성공: {signals['primary_signal']}")

        print("\n모든 빠른 테스트 통과! ✅")

    except Exception as e:
        print(f"\n테스트 실패: {e}")
        logger.error(f"빠른 테스트 실패: {e}", exc_info=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="통합 시스템 테스트")
    parser.add_argument('--quick', action='store_true', help='빠른 테스트만 실행')
    parser.add_argument('--full', action='store_true', help='전체 테스트 실행')

    args = parser.parse_args()

    if args.quick:
        asyncio.run(quick_test())
    else:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
