#!/usr/bin/env python3
"""
테스트 스크립트: 사용자 개입 없이 특정 앱 윈도우 자동 캡처

이 스크립트는 사용자의 개입 없이 지정된 애플리케이션의 윈도우를 자동으로 캡처합니다.
"""

import sys
import os
import time
from macos_window_capture import capture_app_window

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_auto_capture.py [앱 이름]")
        print("예: python test_auto_capture.py RIDI")
        
        # 사용 가능한 앱 이름 표시
        print("\n사용 가능한 앱:")
        from macos_window_capture import get_running_applications
        apps = get_running_applications()
        for app in apps:
            print(f"- {app['name']} (PID: {app['pid']})")
        
        sys.exit(1)
        
    app_name = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    print(f"'{app_name}' 앱 윈도우 자동 캡처 시작...")
    
    # 캡처 시도
    img = capture_app_window(app_name)
    
    if img:
        # 결과 저장
        output_path = os.path.join(output_dir, f"{app_name.replace(' ', '_')}_auto_capture.png")
        img.save(output_path)
        print(f"캡처 성공! 이미지가 다음 위치에 저장되었습니다: {output_path}")
        
        # 이미지 크기 표시
        print(f"이미지 크기: {img.size[0]}x{img.size[1]} 픽셀")
    else:
        print(f"'{app_name}' 앱 윈도우 캡처 실패")
        sys.exit(1) 