# 현재 컨텍스트 (Current Context)

이 문서는 현재 작업 중인 컨텍스트와 진행 상황을 기록합니다.
각 AI Agent가 작업을 시작할 때 이 문서를 먼저 확인하세요.

## 현재 Phase

**Phase 2: High Priority (진행 중)**

## 최근 완료 사항

### Phase 1: Critical (완료 ✅)

1. **리스크 관리 모듈 구현**
   - `src/risk_management/stop_loss.py`: Spot/Futures 공통 손절/익절 관리
   - `src/risk_management/position_sizing.py`: Spot/Futures 포지션 사이징
   - `src/risk_management/exposure_manager.py`: 최대 노출 제한 통합 관리
   - `src/risk_management/risk_checker.py`: 단계별 리스크 체크 시스템

2. **데이터 전처리 모듈 명시화**
   - `src/data_processing/preprocessor.py`: 독립 전처리 모듈
   - `src/data_processing/unified_processor.py`: 새 전처리 모듈 사용

## 현재 작업 중인 항목

### 1. 피드백 루프 강화 (Must-have)

**목표:**
- 리스크 관리 → 데이터 수집 피드백 구현
- 실시간 성과 기반 자동 조정
- Spot/Futures별 성과 추적

**진행 상황:**
- 아직 시작 안 함

**다음 단계:**
- `src/integration/realtime_backtest_integration.py` 분석
- 피드백 루프 구현 위치 결정
- 성과 추적 모듈과 리스크 관리 모듈 연동

### 2. 리스크 관리 모듈 통합 (Must-have)

**목표:**
- 기존 OrderManager의 리스크 관리 로직을 새 모듈로 교체
- ExposureManager와 OrderManager 연동

**진행 상황:**
- 아직 시작 안 함

**다음 단계:**
- `src/execution/order_manager.py` 분석
- `src/execution/spot_order_manager.py` 분석
- 리스크 관리 로직 교체 계획 수립

## 다음 작업자를 위한 정보

### 시작하기 전에

1. **문서 읽기 순서:**
   ```
   CONTEXT.md (이 파일) → PRIORITIES.md → DECISIONS.md → ARCHITECTURE.md
   ```

2. **현재 Phase 확인:**
   - Phase 2의 Must-have 항목 우선 처리
   - Should-have는 Must-have 완료 후

3. **레거시 코드 주의:**
   - `src/execution/order_manager.py`의 리스크 관리 로직은 레거시
   - 삭제 전 마이그레이션 필수
   - DECISIONS.md에서 삭제 조건 확인

### 작업 시 주의사항

1. **설계 원칙 준수:**
   - 공통 인터페이스 + 구현체 분리
   - 단일 책임 원칙
   - Spot/Futures 모두 지원

2. **의사결정 기록:**
   - 중요한 결정은 DECISIONS.md에 기록
   - Why, 결정 과정, 우선순위 명시

3. **코드 일관성:**
   - 기존 코드 스타일 유지
   - 타입 힌팅 사용
   - 로깅 포함

### 작업 완료 후

1. **문서 업데이트:**
   - CONTEXT.md: 완료 항목 체크, 다음 작업 명시
   - PRIORITIES.md: 완료 항목 ✅ 표시
   - DECISIONS.md: 새로운 의사결정 기록 (필요 시)

2. **테스트 확인:**
   - 기존 테스트 통과 확인
   - 새 기능 테스트 추가 (가능한 경우)

## 알려진 이슈

### 현재 없음

## 향후 계획

### Phase 2 완료 후 (Phase 3)

1. 이벤트 기반 아키텍처 도입
2. 데이터 품질 관리 강화
3. 백테스팅 기능 확장

### Phase 3 완료 후 (Phase 4)

1. 머신러닝 통합
2. 고급 모니터링 시스템
3. 자동화된 리포트 생성

## 참고 자료

- **의사결정 기록**: `docs/DECISIONS.md`
- **아키텍처 문서**: `docs/ARCHITECTURE.md`
- **우선순위 관리**: `docs/PRIORITIES.md`
- **Agent 역할 분담**: `docs/AGENT_ROLES.md`
- **Agent 작업 명세**: `docs/AGENT_TASKS.md`
- **Agent 시작 가이드**: `docs/AGENT_START_GUIDE.md` ⭐ **새 컨텍스트 시작 시 필수**
- **프로젝트 README**: `README.md`

## 작업 히스토리

### [2024-12-19] Phase 1 완료
- 리스크 관리 모듈 구현 완료
- 단계별 리스크 체크 시스템 구현 완료
- 데이터 전처리 모듈 명시화 완료
- Phase 2 시작

### [2024-12-19] 문서화 프레임워크 구축
- DECISIONS.md 생성
- ARCHITECTURE.md 생성
- PRIORITIES.md 생성
- CONTEXT.md 생성

