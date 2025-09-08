# Technical Guideline — Portrait Outline Generator

## 0) 목표 정의
- 입력: 인물 사진(JPG/PNG)  
- 출력: 
  - `*_construction.png` (세로 중심선 + 눈높이선 가이드)  
  - `*_outline.png` (단순·깨끗한 윤곽선)  
  - `*_outline.svg` (선택, 벡터)  
- 제약: **원본 형태 충실**, **상업적 사용 라이선스 안전**

---

# 1) 시스템 아키텍처

```mermaid
flowchart LR
  A[Input Portrait Image] --> B[Preprocessing\n(OpenCV: resize, color convert, edge-friendly filtering)]
  B --> C[Person Segmentation\n(Mediapipe SelfieSeg)]
  C -->|Mask| D[Edge Detection\n(PiDiNet/DexiNed or Canny)]
  D --> E[Morphology + Small Component Removal]
  E --> F[Contourization\n(findContours)]
  F --> G[Polyline Simplification\n(Douglas–Peucker)]
  G --> H[Curve Smoothing\n(Chaikin)]
  H --> I1[Raster Outline\n(*_outline.png)]
  B --> J[Face Detection & Eye Keypoints\n(Mediapipe FaceDetection)]
  J --> K[Guide Composer\n(center vertical + eye line)]
  K --> I2[Construction Sheet\n(*_construction.png)]
  H --> I3[SVG Export\n(*_outline.svg)]
```

---

# 2) 설치 & 환경

### 2.1. Conda(권장)
```bash
# 새 환경
conda create -n portrait_outline python=3.10 -y
conda activate portrait_outline

# 필수
pip install opencv-python numpy mediapipe

# 선택(스켈레톤/고급 후처리, SVG 생산에 도움)
pip install scikit-image

# (선택) 딥엣지 모델
# PiDiNet/DexiNed는 보통 torch 모델로 배포되므로 필요 시:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA환경 예시
# CPU라면 일반 pip 설치
```

### 2.2. Windows 주의
- `mediapipe`가 Visual C++ Redistributable 요구할 수 있음(보통 자동 해결).  
- GPU가 없다면 Canny + Mediapipe로도 충분히 고품질 결과 가능.

---

# 3) 디렉토리 구조 (권장)

```
portrait_outline/
  ├─ requirements.txt
  ├─ src/
  │   ├─ pipeline.py                # CLI 엔트리
  │   ├─ stages/
  │   │   ├─ preprocess.py
  │   │   ├─ segmentation.py
  │   │   ├─ edge_detection.py
  │   │   ├─ postprocess.py
  │   │   ├─ vectorize.py
  │   │   └─ guides.py
  │   └─ utils/
  │       ├─ io.py
  │       └─ geometry.py
  ├─ models/                        # (선택) PiDiNet/DexiNed 체크포인트
  ├─ configs/
  │   └─ default.yaml
  └─ out/                           # 결과물
```

---

# 4) 파이프라인 상세
...(truncated for brevity in this code block, assume full guideline text included from assistant's prior answer)...
