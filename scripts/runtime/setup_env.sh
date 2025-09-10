#!/bin/bash
# Portrait Outline Project Environment Setup

echo "=== Portrait Outline 프로젝트 환경 설정 ==="

# Conda 환경 활성화
echo "1. portrait_outline 환경 활성화 중..."
conda activate portrait_outline

# 필요한 패키지 설치
echo "2. 필수 패키지 설치 중..."
pip install -r requirements.txt

# PATH 설정 확인
echo "3. PATH 설정 확인 중..."
export PATH="$HOME/.local/bin:$PATH"

# 디렉토리 구조 확인
echo "4. 프로젝트 디렉토리 구조:"
tree -L 2 . || ls -la

echo "=== 환경 설정 완료 ==="
echo "사용법:"
echo "  - 데이터 다운로드: cd download_data && python aic_portrait_paintings_downloader.py"
echo "  - 이미지 처리: python image_processing/run_cutout.py -i <image_path>"
echo "  - SuperClaude: superclaude install"