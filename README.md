# Portrait Outline Generator (arti_outlines_aic)

This project generates artistic outlines from portrait images using a complete pipeline.

## 🚀 Quick Start

### 1. Environment Setup (한 번만 실행)
```bash
# 도구 설치
./install_tools.sh

# 환경 활성화
source activate.sh
```

### 2. Usage

#### 데이터 다운로드
```bash
cd download_data
python aic_portrait_paintings_downloader.py
```

#### 이미지 처리
```bash
python image_clipping/run_cutout.py -i out/aic_sample/images/12345.jpg
```

#### SuperClaude 사용
```bash
superclaude install
```

## 📁 Project Structure

```
arti_outlines/
├── .tools/              # 프로젝트 전용 도구들
├── src/
│   ├── stages/         # 파이프라인 단계들
│   └── utils/          # 유틸리티 함수들
├── models/             # AI 모델 파일들
├── out/                # 출력 결과물들
├── download_data/      # 데이터 다운로드 스크립트
├── image_clipping/     # 이미지 클리핑 도구
├── activate.sh         # 환경 활성화 스크립트
└── requirements.txt    # 필수 패키지 목록
```

## 🔧 Environment Management

모든 외부 도구(SuperClaude 등)가 `portrait_outline` conda 환경에 격리되어 설치됩니다.

- ✅ 환경 격리: 다른 프로젝트와 충돌 없음
- ✅ 일관된 Python 버전 사용
- ✅ 의존성 관리 통합
- ✅ 재현 가능한 환경

## 📝 Notes

- 모든 이미지는 퍼블릭 도메인에서만 다운로드됩니다
- GPU 가속 지원 (CUDA/ROCm)
- 자동 경로 관리로 설정 간소화
