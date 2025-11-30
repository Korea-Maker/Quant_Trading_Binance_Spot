#!/usr/bin/env python3
"""
의존성 자동 설치 스크립트 (통합 버전)

이 스크립트는 프로젝트의 모든 의존성을 자동으로 설치합니다.
- TA-Lib (기술적 지표 라이브러리)
- websockets (python-binance 의존성)
- websocket-client (바이낸스 웹소켓 클라이언트)
- requirements.txt의 모든 패키지

사용법:
    python install_dependencies.py                    # 일반 설치
    python install_dependencies.py --reinstall-websocket  # websocket 재설치
    python install_dependencies.py --reinstall-all        # 모든 패키지 재설치
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """명령 실행 및 결과 출력"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"경고: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print(f"출력: {e.stdout}")
        print(f"에러: {e.stderr}")
        return False

def check_package(package_name, import_name=None):
    """패키지 설치 여부 확인"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} 설치됨")
        return True
    except ImportError:
        print(f"✗ {package_name} 미설치")
        return False

def install_talib():
    """TA-Lib 설치"""
    print("\n" + "="*60)
    print("TA-Lib 설치 중...")
    print("="*60)
    
    # 이미 설치되어 있는지 확인
    if check_package("TA-Lib", "talib"):
        return True
    
    # Wheel 파일 경로 확인
    wheel_file = Path(__file__).parent / "ta_lib-0.6.3-cp313-cp313-win_amd64.whl"
    
    if wheel_file.exists():
        print(f"Wheel 파일 발견: {wheel_file}")
        cmd = f"{sys.executable} -m pip install {wheel_file}"
        if run_command(cmd, "TA-Lib wheel 파일 설치 중..."):
            if check_package("TA-Lib", "talib"):
                print("✓ TA-Lib 설치 성공!")
                return True
    
    # Wheel 파일이 없으면 PyPI에서 설치 시도
    print("Wheel 파일이 없습니다. PyPI에서 설치 시도...")
    cmd = f"{sys.executable} -m pip install TA-Lib"
    if run_command(cmd, "TA-Lib PyPI 설치 시도 중..."):
        if check_package("TA-Lib", "talib"):
            print("✓ TA-Lib 설치 성공!")
            return True
    
    print("⚠ TA-Lib 설치 실패. 수동 설치가 필요할 수 있습니다.")
    print("   참고: INSTALL.md 파일을 확인하세요.")
    return False

def uninstall_websocket_packages():
    """websocket 관련 패키지 모두 삭제"""
    print("\n" + "="*60)
    print("websocket 관련 패키지 삭제 중...")
    print("="*60)
    
    packages_to_remove = ['websocket-client', 'websockets', 'websocket']
    
    for package in packages_to_remove:
        cmd = f"{sys.executable} -m pip uninstall -y {package}"
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
            print(f"✓ {package} 삭제 시도 완료")
        except Exception as e:
            print(f"⚠ {package} 삭제 중 오류 (무시): {e}")
    
    print("✓ websocket 관련 패키지 삭제 완료")

def install_websockets():
    """websockets 라이브러리 설치 (python-binance 의존성)"""
    print("\n" + "="*60)
    print("websockets 라이브러리 설치 중...")
    print("="*60)
    
    # 이미 설치되어 있는지 확인
    if check_package("websockets", "websockets"):
        return True
    
    cmd = f"{sys.executable} -m pip install websockets>=15.0.1"
    if run_command(cmd, "websockets 설치 중..."):
        if check_package("websockets", "websockets"):
            print("✓ websockets 설치 성공!")
            return True
    
    print("⚠ websockets 설치 실패.")
    return False

def install_websocket_client(force_reinstall=False):
    """websocket-client 설치
    
    Args:
        force_reinstall: True이면 기존 패키지를 삭제하고 재설치
    """
    print("\n" + "="*60)
    print("websocket-client 설치 중...")
    print("="*60)
    
    if force_reinstall:
        uninstall_websocket_packages()
    
    # 이미 설치되어 있는지 확인
    if not force_reinstall and check_package("websocket-client", "websocket"):
        print("✓ websocket-client가 이미 설치되어 있습니다.")
        return True
    
    # websocket-client 설치
    install_cmd = f"{sys.executable} -m pip install --upgrade --force-reinstall websocket-client>=1.6.0"
    if run_command(install_cmd, "websocket-client 설치 중..."):
        if check_package("websocket-client", "websocket"):
            print("✓ websocket-client 설치 성공!")
            return True
    
    print("⚠ websocket-client 설치 실패.")
    return False

def install_requirements():
    """requirements.txt의 모든 패키지 설치"""
    print("\n" + "="*60)
    print("requirements.txt 패키지 설치 중...")
    print("="*60)
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("⚠ requirements.txt 파일을 찾을 수 없습니다.")
        return False
    
    cmd = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(cmd, "requirements.txt 설치 중...")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='의존성 자동 설치 스크립트')
    parser.add_argument('--reinstall-websocket', action='store_true',
                        help='websocket-client를 삭제하고 재설치')
    parser.add_argument('--reinstall-all', action='store_true',
                        help='모든 패키지를 삭제하고 재설치')
    args = parser.parse_args()
    
    print("="*60)
    print("의존성 자동 설치 스크립트")
    print("="*60)
    
    if args.reinstall_websocket or args.reinstall_all:
        print("\n[주의] websocket 관련 패키지를 삭제하고 재설치합니다.")
    
    # Python 버전 확인
    print(f"\nPython 버전: {sys.version}")
    print(f"Python 실행 경로: {sys.executable}")
    
    # pip 업그레이드
    print("\n" + "="*60)
    print("pip 업그레이드 중...")
    print("="*60)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                   capture_output=True)
    
    # 필수 패키지 설치
    results = {
        "websockets": install_websockets(),  # python-binance 의존성 (우선 설치)
        "websocket-client": install_websocket_client(force_reinstall=args.reinstall_websocket or args.reinstall_all),
        "TA-Lib": install_talib(),
    }
    
    # requirements.txt 설치 (TA-Lib은 이미 설치했으므로 제외)
    print("\n" + "="*60)
    print("기타 의존성 설치 중...")
    print("="*60)
    
    # requirements.txt 설치 (TA-Lib은 이미 설치했으므로 스킵)
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        # requirements.txt를 읽어서 TA-Lib 라인 제외 (인코딩 자동 감지)
        lines = []
        encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'cp949', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(requirements_file, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                print(f"✓ requirements.txt 파일 읽기 성공 (인코딩: {encoding})")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                continue
        
        if not lines:
            print(f"⚠ requirements.txt 파일을 읽을 수 없습니다. 인코딩 문제일 수 있습니다.")
            print(f"   수동으로 설치하세요: pip install -r requirements.txt")
        else:
            # TA-Lib 관련 라인 제외 (주석 제외)
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                # TA-Lib 라인 제외 (주석이 아닌 경우)
                if stripped and not stripped.startswith('#') and 'TA-Lib' in line:
                    continue
                filtered_lines.append(line)
            
            # 임시 파일 생성 (UTF-8로 저장)
            temp_requirements = Path(__file__).parent / "requirements_temp.txt"
            try:
                with open(temp_requirements, 'w', encoding='utf-8', newline='\n') as f:
                    f.writelines(filtered_lines)
            except Exception as e:
                print(f"⚠ 임시 파일 생성 실패: {e}")
                return False
            
            cmd = f"{sys.executable} -m pip install -r {temp_requirements}"
            install_success = run_command(cmd, "기타 의존성 설치 중...")
            
            # 임시 파일 삭제
            try:
                temp_requirements.unlink()
            except:
                pass
            
            if install_success:
                print("✓ 기타 의존성 설치 완료")
    
    # 최종 확인
    print("\n" + "="*60)
    print("설치 확인")
    print("="*60)
    
    all_ok = True
    for package, installed in results.items():
        if installed:
            print(f"✓ {package}: 설치 완료")
        else:
            print(f"✗ {package}: 설치 실패")
            all_ok = False
    
    if all_ok:
        print("\n✓ 모든 필수 패키지가 설치되었습니다!")
    else:
        print("\n⚠ 일부 패키지 설치에 실패했습니다. 수동 설치가 필요할 수 있습니다.")
        print("   참고: INSTALL.md 파일을 확인하세요.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

