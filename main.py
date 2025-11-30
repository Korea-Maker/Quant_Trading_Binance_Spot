#!/usr/bin/env python3
"""
바이낸스 비트코인 자동매매 시스템 메인 진입점

이 모듈은 전체 트레이딩 시스템을 초기화하고 실행하는 메인 진입점입니다.
"""

import asyncio
import signal
import sys
from typing import Optional
from datetime import datetime

from src.utils.logger import get_logger
from src.config.settings import (
    DEFAULT_SYMBOL,
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    APP_MODE,
    TEST_MODE,
    MAX_POSITION_SIZE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT
)
from src.integration.realtime_backtest_integration import RealtimeBacktestIntegration
from src.strategy.signals import SignalBasedStrategy


class TradingSystem:
    """트레이딩 시스템 메인 클래스"""
    
    def __init__(self):
        """시스템 초기화"""
        self.logger = get_logger(__name__)
        self.integration: Optional[RealtimeBacktestIntegration] = None
        self.is_running = False
        
        # 설정 검증
        self._validate_config()
        
    def _validate_config(self):
        """설정 검증"""
        errors = []
        
        # API 키 검증
        if not BINANCE_API_KEY:
            errors.append("BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not BINANCE_API_SECRET:
            errors.append("BINANCE_API_SECRET이 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        if errors:
            self.logger.error("설정 오류:")
            for error in errors:
                self.logger.error(f"  - {error}")
            raise ValueError("필수 설정이 누락되었습니다. .env 파일을 확인하세요.")
        
        # 테스트넷/메인넷 모드 확인
        if TEST_MODE:
            self.logger.info("[바이낸스 테스트넷] 테스트넷에서 실제 주문이 실행됩니다 (메인넷에는 영향 없음)")
        else:
            self.logger.warning("[바이낸스 메인넷] 메인넷에서 실제 주문이 실행됩니다. 주의하세요!")
        
        self.logger.info(f"설정 검증 완료 - 모드: {APP_MODE}, 심볼: {DEFAULT_SYMBOL}")
    
    def _create_strategy(self):
        """전략 생성"""
        # SignalBasedStrategy 사용
        strategy = SignalBasedStrategy(
            min_confidence=60.0,
            min_signal_strength=1.0,
            use_pattern_confirmation=True
        )
        return strategy
    
    async def start(self, 
                   symbol: str = DEFAULT_SYMBOL,
                   interval: str = "1h",
                   backtest_lookback_days: int = 30,
                   total_capital: float = 10000.0):
        """시스템 시작"""
        if self.is_running:
            self.logger.warning("시스템이 이미 실행 중입니다.")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("비트코인 자동매매 시스템 시작")
        self.logger.info("=" * 60)
        self.logger.info(f"심볼: {symbol}")
        self.logger.info(f"인터벌: {interval}")
        self.logger.info(f"백테스트 기간: {backtest_lookback_days}일")
        self.logger.info(f"총 자본: {total_capital:.2f} USDT")
        self.logger.info(f"최대 포지션 크기: {MAX_POSITION_SIZE_PCT * 100}%")
        self.logger.info(f"손절: {STOP_LOSS_PCT * 100}%, 익절: {TAKE_PROFIT_PCT * 100}%")
        self.logger.info("=" * 60)
        
        try:
            # 통합 시스템 초기화
            self.integration = RealtimeBacktestIntegration(
                symbol=symbol,
                interval=interval,
                backtest_lookback_days=backtest_lookback_days,
                feedback_update_hours=24
            )
            
            # 전략 생성
            strategy = self._create_strategy()
            
            # 시스템 시작
            self.is_running = True
            await self.integration.start(strategy, total_capital=total_capital)
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
            await self.stop()
        except Exception as e:
            self.logger.error(f"시스템 오류 발생: {e}", exc_info=True)
            await self.stop()
            raise
    
    async def stop(self):
        """시스템 중지"""
        if not self.is_running:
            return
        
        self.logger.info("시스템 종료 중...")
        self.is_running = False
        
        if self.integration:
            try:
                await self.integration.stop()
            except Exception as e:
                self.logger.error(f"시스템 종료 중 오류: {e}")
        
        # 성능 요약 출력
        if self.integration:
            try:
                summary = self.integration.get_performance_summary()
                self.logger.info("=" * 60)
                self.logger.info("성능 요약:")
                self.logger.info(f"  총 거래 수: {summary.get('total_trades', 0)}")
                self.logger.info(f"  활성 포지션: {summary.get('active_positions', 0)}")
                if 'recent_win_rate' in summary:
                    self.logger.info(f"  최근 승률: {summary['recent_win_rate']:.2%}")
                if 'recent_avg_profit' in summary:
                    self.logger.info(f"  최근 평균 수익률: {summary['recent_avg_profit']:.2%}")
                self.logger.info("=" * 60)
            except Exception as e:
                self.logger.error(f"성능 요약 생성 중 오류: {e}")
        
        self.logger.info("시스템 종료 완료")


def setup_signal_handlers(system: TradingSystem, loop: asyncio.AbstractEventLoop):
    """시그널 핸들러 설정 (Ctrl+C 등)
    
    Args:
        system: TradingSystem 인스턴스
        loop: 실행 중인 이벤트 루프
    """
    # 비동기 시그널 핸들러 함수
    async def async_signal_handler(signum: int):
        """비동기 시그널 핸들러"""
        logger = get_logger(__name__)
        logger.info(f"시그널 {signum} 수신 - 시스템 종료 중...")
        await system.stop()
    
    # 동기 시그널 핸들러 (fallback용)
    def sync_signal_handler(signum, frame):
        """동기 시그널 핸들러 - 이벤트 루프에 종료 요청 전달"""
        logger = get_logger(__name__)
        logger.info(f"시그널 {signum} 수신 - 시스템 종료 중...")
        
        # 이벤트 루프가 실행 중인지 확인
        if loop.is_running():
            # 스레드 안전하게 코루틴 스케줄링
            asyncio.run_coroutine_threadsafe(system.stop(), loop)
        else:
            # 이벤트 루프가 없으면 직접 종료 플래그 설정
            system.is_running = False
    
    # Unix 시스템에서는 add_signal_handler 사용 (더 안전)
    if hasattr(signal, 'SIGINT') and hasattr(loop, 'add_signal_handler'):
        try:
            # 비동기 핸들러 직접 등록
            loop.add_signal_handler(
                signal.SIGINT,
                lambda: asyncio.create_task(async_signal_handler(signal.SIGINT))
            )
            if hasattr(signal, 'SIGTERM'):
                loop.add_signal_handler(
                    signal.SIGTERM,
                    lambda: asyncio.create_task(async_signal_handler(signal.SIGTERM))
                )
        except (NotImplementedError, ValueError, OSError):
            # Windows에서는 add_signal_handler가 지원되지 않을 수 있음
            # fallback to signal.signal
            signal.signal(signal.SIGINT, sync_signal_handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, sync_signal_handler)
    else:
        # Windows 또는 add_signal_handler 미지원 환경
        signal.signal(signal.SIGINT, sync_signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, sync_signal_handler)


async def main():
    """메인 함수"""
    system = TradingSystem()
    
    # 현재 실행 중인 이벤트 루프 가져오기
    loop = asyncio.get_running_loop()
    
    # 시그널 핸들러 설정 (이벤트 루프와 함께)
    setup_signal_handlers(system, loop)
    
    try:
        # 시스템 시작
        await system.start()
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"시스템 실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램이 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"치명적 오류: {e}")
        sys.exit(1)
