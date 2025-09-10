# Complete GPU Acceleration Setup for MediaPipe

## 🚀 실행 순서

### 1단계: 필수 패키지 설치 (한 번만 실행)
```bash
./install_gpu_support.sh
```

### 2단계: 가상 디스플레이 시작
```bash
./start_virtual_display.sh
```

### 3단계: GPU 환경 설정
```bash
source setup_gpu_env.sh
```

### 4단계: Jupyter 노트북에서 GPU 설정 셀 실행
노트북의 "Complete GPU acceleration setup for MediaPipe" 셀을 실행

### 5단계: MediaPipe 실행
이제 MediaPipe가 GPU 가속으로 실행됩니다!

## 🔧 설치되는 패키지들

- **Xvfb**: 가상 디스플레이 서버
- **Mesa OpenGL**: GPU 렌더링 라이브러리
- **ROCm OpenCL**: AMD GPU 컴퓨팅
- **Vulkan**: 최신 GPU API 지원

## 🎯 GPU 가속 확인 방법

MediaPipe 로그에서 다음 메시지들을 확인:
- ❌ `GPU suport is not available` → GPU 미사용
- ✅ `GPU acceleration enabled` → GPU 가속 성공

## 🛠️ 문제해결

### GPU 가속이 안 되는 경우:
1. `sudo apt install xvfb mesa-utils` 실행
2. `./start_virtual_display.sh` 다시 실행
3. `source setup_gpu_env.sh` 다시 실행

### 성능 모니터링:
```bash
# GPU 상태 확인
rocm-smi

# 프로세스 모니터링
watch -n 1 rocm-smi
```

## 🧹 정리

작업 완료 후:
```bash
./stop_virtual_display.sh
```

## 📊 예상 성능 향상

- **CPU 모드**: ~100-200ms per frame
- **GPU 가속**: ~10-50ms per frame (2-10x 빠름)