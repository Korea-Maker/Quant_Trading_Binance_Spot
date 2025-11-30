# 설치 가이드

## TA-Lib 설치 방법

TA-Lib은 C 라이브러리 의존성이 있어 Windows에서 설치가 까다로울 수 있습니다.

### 방법 1: Wheel 파일 사용 (권장)

프로젝트 루트에 `ta_lib-0.6.3-cp313-cp313-win_amd64.whl` 파일이 있습니다.

```bash
# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# Wheel 파일 설치
pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl
```

### 방법 2: 공식 Wheel 파일 다운로드

1. https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib 방문
2. Python 버전에 맞는 wheel 파일 다운로드 (예: `TA_Lib‑0.4.28‑cp313‑cp313‑win_amd64.whl`)
3. 설치:
   ```bash
   pip install TA_Lib‑0.4.28‑cp313‑cp313‑win_amd64.whl
   ```

### 방법 3: conda 사용 (가장 쉬움)

```bash
conda install -c conda-forge ta-lib
```

### 설치 확인

```bash
python test_talib.py
```

## 전체 시스템 설치

1. 가상환경 생성 및 활성화:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # 또는
   source .venv/bin/activate  # Linux/Mac
   ```

2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. TA-Lib 설치 (위 방법 중 하나 선택)

4. .env 파일 생성:
   ```bash
   cp .env.example .env
   # .env 파일을 열어 API 키 설정
   ```

5. 테스트 실행:
   ```bash
   python test_talib.py      # TA-Lib 테스트
   python test_system.py     # 전체 시스템 테스트
   ```

## 문제 해결

### "ModuleNotFoundError: No module named 'talib'"

1. 가상환경이 활성화되어 있는지 확인:
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   ```

2. Python 버전 확인:
   ```bash
   python --version
   ```
   wheel 파일의 Python 버전과 일치해야 합니다.

3. 재설치:
   ```bash
   pip uninstall TA-Lib
   pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl
   ```

### "ImportError: DLL load failed"

TA-Lib의 C 라이브러리가 필요합니다. conda를 사용하거나 공식 wheel 파일을 사용하세요.

