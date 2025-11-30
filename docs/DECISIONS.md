# 의사결정 기록 (Decision Log)

이 문서는 프로젝트의 중요한 의사결정과 그 이유를 기록합니다.
각 AI Agent가 이전 결정사항을 이해하고 일관성 있게 작업할 수 있도록 합니다.

**중요**: 이 문서는 인간이 직접 작성하는 문서입니다. AI Agent는 이 문서를 읽고 참고만 하며, 새로운 의사결정을 기록할 때는 아래 형식을 따릅니다.

## 형식

각 의사결정은 다음 형식을 따릅니다:

```markdown
### [YYYY-MM-DD] 결정 제목

**상황 (Context):**
- 어떤 상황에서 이 결정이 필요했는지

**고려사항 (Considerations):**
- 검토한 대안들
- 각 대안의 장단점

**결정 (Decision):**
- 최종 결정 내용

**이유 (Why):**
- 왜 이 결정을 내렸는지
- 어떤 원칙이나 제약사항이 영향을 미쳤는지

**우선순위:**
- Must-have / Should-have / Nice-to-have

**영향 (Impact):**
- 이 결정이 다른 부분에 미치는 영향
- 레거시 코드나 미래 작업에 대한 영향

**관련 파일:**
- 이 결정과 관련된 파일 경로
```

---

## 의사결정 기록

<!-- 인간이 여기에 의사결정을 기록합니다 -->

---

## 레거시 코드 정리 가이드

### 레거시로 간주되는 코드

다음 코드들은 새로운 모듈로 대체되었으나, 하위 호환성을 위해 유지됩니다:

1. **`src/execution/order_manager.py`의 리스크 관리 로직**
   - 대체: `src/risk_management/stop_loss.py`, `src/risk_management/position_sizing.py`
   - 상태: 레거시 (마이그레이션 예정)
   - 삭제 시기: Phase 2 완료 후

2. **`src/execution/spot_order_manager.py`의 리스크 관리 로직**
   - 대체: `src/risk_management/stop_loss.py`, `src/risk_management/position_sizing.py`
   - 상태: 레거시 (마이그레이션 예정)
   - 삭제 시기: Phase 2 완료 후

3. **`src/data_processing/unified_processor.py`의 `_remove_outliers()` 메서드**
   - 대체: `src/data_processing/preprocessor.py`의 `_remove_outliers()`
   - 상태: 레거시 (사용 안 함)
   - 삭제 시기: Phase 2 완료 후

### 유지해야 하는 코드

다음 코드들은 핵심 기능이므로 유지해야 합니다:

1. **`src/execution/order_manager.py`의 주문 실행 로직**
   - 상태: 핵심 기능 (유지)
   - 리스크 관리 로직만 새로운 모듈로 분리

2. **`src/execution/spot_order_manager.py`의 주문 실행 로직**
   - 상태: 핵심 기능 (유지)
   - 리스크 관리 로직만 새로운 모듈로 분리

---

## 다음 작업자를 위한 참고사항

### Must-have (반드시 구현)
- 리스크 관리 모듈과 기존 OrderManager 통합
- 단계별 리스크 체크를 데이터 처리 파이프라인에 통합

### Should-have (구현 권장)
- 레거시 코드 제거 (Phase 2 완료 후)
- 리스크 관리 모듈 단위 테스트 작성

### Nice-to-have (구현 여유 시)
- 리스크 관리 모듈 성능 최적화
- 추가 리스크 체크 규칙 구현
