# 바이낸스 API 키 설정 가이드

## 현재 상태

시스템은 정상적으로 작동하고 있으며, 신호 생성 및 전략 실행은 성공적으로 이루어지고 있습니다.  
하지만 **실제 거래 실행은 API 키 오류로 인해 실패**하고 있습니다.

### 발생한 오류
```
ERROR - 계정 잔액 조회 실패: APIError(code=-2015): Invalid API-key, IP, or permissions for action.
ERROR - 잔액 부족: 0.0 < 450.00000000000006 USDT
ERROR - 매수 주문 실패
```

## 해결 방법

### 1. 바이낸스 테스트넷 API 키 생성

#### 1.1 테스트넷 접속
- URL: https://testnet.binance.vision/
- 또는: https://testnet.binancefuture.com/ (선물 거래용)

#### 1.2 API 키 생성
1. 테스트넷 웹사이트에 로그인
2. **API Management** 메뉴로 이동
3. **Create API** 클릭
4. 다음 권한 활성화:
   - ✅ **Enable Reading** (읽기 권한)
   - ✅ **Enable Spot & Margin Trading** (스팟 거래 권한)
5. **IP Restriction** 설정:
   - 현재 IP 주소 추가, 또는
   - IP 제한 비활성화 (개발/테스트용)
6. **Create** 클릭하여 API 키 생성
7. **API Key**와 **Secret Key** 복사 (한 번만 표시됨!)

### 2. 환경 변수 설정

#### 2.1 `.env` 파일 확인/생성
프로젝트 루트 디렉토리에 `.env` 파일이 있는지 확인하고, 없으면 생성합니다.

#### 2.2 API 키 입력
`.env` 파일에 다음 내용을 추가/수정합니다:

```env
# 바이낸스 API 키 (테스트넷)
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_api_secret_here

# 테스트 모드 (True=테스트넷, False=메인넷)
TEST_MODE=True

# 거래 타입 (spot 또는 futures)
TRADING_TYPE=futures

# SSL 인증서 검증 (Windows에서 SSL 오류 발생 시 False로 설정)
# 주의: 프로덕션 환경에서는 True로 유지하는 것이 안전합니다
SSL_VERIFY=False
```

**⚠️ 주의사항:**
- `.env` 파일은 절대 Git에 커밋하지 마세요 (이미 `.gitignore`에 포함됨)
- API 키는 외부에 노출되지 않도록 주의하세요
- 테스트넷과 메인넷의 API 키는 다릅니다

### 3. 테스트넷 잔액 확인

#### 3.1 테스트 자산 받기
바이낸스 테스트넷은 무료 테스트 자산을 제공합니다:
1. 테스트넷 웹사이트 로그인
2. **Faucet** 또는 **Get Test Funds** 메뉴 확인
3. 테스트 USDT 받기 (일부 테스트넷은 자동으로 제공)

#### 3.2 잔액 확인
- 테스트넷 웹사이트에서 잔액 확인
- 또는 시스템 실행 후 로그에서 확인

### 4. API 키 권한 확인

다음 권한이 활성화되어 있어야 합니다:

- ✅ **Enable Reading**: 계정 정보, 잔액 조회
- ✅ **Enable Spot & Margin Trading**: 스팟 거래 주문 실행
- ❌ **Enable Withdrawals**: 출금 권한 (필요 없음, 보안상 비활성화 권장)

### 5. IP 제한 설정

#### 5.1 IP 제한이 있는 경우
1. 현재 IP 주소 확인: https://www.whatismyip.com/
2. 바이낸스 API 설정에서 IP 추가
3. 변경사항 저장 (최대 5분 소요)

#### 5.2 IP 제한이 없는 경우
- 개발/테스트 환경에서는 IP 제한을 비활성화할 수 있습니다
- **프로덕션 환경에서는 반드시 IP 제한을 활성화하세요**

## 문제 해결 체크리스트

거래가 실행되지 않을 때 다음을 확인하세요:

- [ ] `.env` 파일에 올바른 API 키가 입력되어 있는가?
- [ ] API 키가 테스트넷용인가? (메인넷 키와 혼동하지 않았는가?)
- [ ] API 키에 "Enable Spot & Margin Trading" 권한이 있는가?
- [ ] IP 제한이 설정되어 있다면 현재 IP가 허용 목록에 있는가?
- [ ] 테스트넷 계정에 충분한 테스트 자산이 있는가?
- [ ] `TEST_MODE=True`로 설정되어 있는가?
- [ ] 가상 환경이 활성화되어 있는가?
- [ ] 시스템을 재시작했는가? (환경 변수 변경 후)

## 테스트 방법

### 1. API 키 연결 테스트
```bash
python test_main.py
```

### 2. 시스템 실행
```bash
python main.py
```

### 3. 로그 확인
다음 로그가 나타나면 성공입니다:
```
INFO - 바이낸스 테스트넷에 연결됨
INFO - API 키 권한 확인 성공
INFO - 거래 실행 시작: buy
INFO - 매수 주문 성공
```

## 주의사항

### 테스트넷 vs 메인넷

| 항목 | 테스트넷 | 메인넷 |
|------|---------|--------|
| URL | testnet.binance.vision | binance.com |
| 자금 | 가상 자금 (무료) | 실제 자금 |
| API 키 | 별도 생성 필요 | 별도 생성 필요 |
| 거래 영향 | 없음 | 실제 거래 발생 |

### 보안 권장사항

1. **API 키 보관**
   - `.env` 파일 사용 (Git에 커밋 금지)
   - 환경 변수로 관리
   - 절대 코드에 하드코딩하지 않기

2. **권한 최소화**
   - 필요한 권한만 활성화
   - 출금 권한은 절대 활성화하지 않기

3. **IP 제한**
   - 프로덕션 환경에서는 반드시 IP 제한 활성화
   - 개발 환경에서도 가능하면 IP 제한 사용

4. **정기적 확인**
   - API 키 사용 내역 정기적 확인
   - 의심스러운 활동 즉시 키 삭제

## 메인넷 사용 시 주의사항

**⚠️ 메인넷은 실제 자금이 사용됩니다!**

메인넷으로 전환하려면:
1. `.env` 파일에서 `TEST_MODE=False`로 변경
2. 메인넷 API 키로 교체
3. 충분한 테스트 후 전환
4. 소액으로 시작하여 검증

## 추가 리소스

- 바이낸스 테스트넷: https://testnet.binance.vision/
- 바이낸스 API 문서: https://binance-docs.github.io/apidocs/
- Python-binance 라이브러리: https://python-binance.readthedocs.io/

## 문제 해결 지원

문제가 지속되면 다음 정보를 확인하세요:
1. 전체 에러 로그
2. `.env` 파일 설정 (키 값은 제외)
3. API 키 권한 스크린샷
4. 테스트넷 계정 잔액

---

**마지막 업데이트**: 2025-11-28  
**작성자**: 시스템 자동 생성


