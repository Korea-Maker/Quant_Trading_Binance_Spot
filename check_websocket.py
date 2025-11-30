#!/usr/bin/env python3
"""
websocket 라이브러리 설치 상태 확인 스크립트
"""

import sys

def check_package(name, import_name=None):
    """패키지 설치 여부 확인"""
    if import_name is None:
        import_name = name
    
    try:
        __import__(import_name)
        print(f"✓ {name} 설치됨")
        return True
    except ImportError as e:
        print(f"✗ {name} 미설치: {e}")
        return False

def main():
    print("="*60)
    print("websocket 라이브러리 설치 상태 확인")
    print("="*60)
    
    results = {}
    
    # 1. websockets 확인
    print("\n1. websockets 라이브러리:")
    results['websockets'] = check_package("websockets", "websockets")
    
    # 2. python-binance의 BinanceSocketManager 확인
    print("\n2. python-binance의 BinanceSocketManager:")
    try:
        from binance.client import Client
        from binance.websockets import BinanceSocketManager
        print("✓ BinanceSocketManager 사용 가능")
        results['binance_socket_manager'] = True
    except ImportError as e:
        print(f"✗ BinanceSocketManager 사용 불가: {e}")
        results['binance_socket_manager'] = False
    
    # 3. websocket-client 확인
    print("\n3. websocket-client 라이브러리:")
    results['websocket_client'] = check_package("websocket-client", "websocket")
    
    # 요약
    print("\n" + "="*60)
    print("요약")
    print("="*60)
    
    if results.get('binance_socket_manager'):
        print("✓ python-binance의 BinanceSocketManager 사용 가능")
        print("  (websockets가 필요합니다)")
    elif results.get('websocket_client'):
        print("✓ websocket-client 사용 가능")
    else:
        print("✗ 사용 가능한 websocket 라이브러리가 없습니다.")
        print("\n설치 방법:")
        if not results.get('websockets'):
            print("  1. pip install websockets>=15.0.1")
        if not results.get('websocket_client'):
            print("  2. pip install websocket-client>=1.6.0")
        print("  3. 또는 python install_dependencies.py 실행")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

