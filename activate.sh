#!/bin/bash

# Portrait Outline Generator Environment Activation Script (Conda Version)
# Usage: source activate.sh

echo "🎨 Activating Portrait Outline Generator Environment..."

# Initialize conda
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
else
    echo "⚠️ Conda 초기화 실패. 수동으로 conda를 초기화하세요."
    eval "$(/opt/anaconda3/bin/conda shell.bash hook)" 2>/dev/null
fi

# Activate conda environment
if conda activate portrait_outline 2>/dev/null; then
    echo "✅ Conda 환경 활성화됨: $(python --version)"
    echo "📦 Conda 환경: $(conda info --envs | grep '*' | awk '{print $1}')"
else
    echo "❌ Conda 환경 'portrait_outline'을 찾을 수 없습니다!"
    echo "다음 명령어로 환경을 생성하세요:"
    echo "  conda create -n portrait_outline python=3.11 -y"
    echo "  conda activate portrait_outline"
    echo "  pip install -r requirements.txt"
    return 1
fi

# Check GPU acceleration capabilities
echo "🚀 GPU 가속 상태:"
python -c "
import onnxruntime as ort
providers = ort.get_available_providers()
if 'CoreMLExecutionProvider' in providers:
    print('✓ CoreML 가속 사용 가능 (Apple Silicon/Mac)')
elif 'CUDAExecutionProvider' in providers:
    print('✓ CUDA 가속 사용 가능 (NVIDIA GPU)')
elif 'ROCMExecutionProvider' in providers:
    print('✓ ROCm 가속 사용 가능 (AMD GPU)')
else:
    print('ℹ CPU 실행 사용 중 (GPU 가속 없음)')
"

# Set environment variables for optimal performance
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=4

echo ""
echo "📍 현재 디렉토리: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Conda 환경: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""
echo "🎯 사용 가능한 명령어:"
echo ""
echo "📍 이미지 처리 디렉토리로 이동:"
echo "  cd image_processing"
echo ""
echo "🖼️ 와이어프레임 포트레이트 생성 (초보자 프리셋):"
echo "  python wireframe_portrait_processor.py input.jpg --preset beginner -o output.png"
echo ""
echo "📊 샘플 이미지로 테스트:"
echo "  ls out_sample/clipped_images_fg/  # 사용 가능한 샘플 이미지 확인"
echo "  python wireframe_portrait_processor.py out_sample/clipped_images_fg/864_fg.png --preset beginner -o test_output.png"
echo ""
echo "📚 전체 옵션 보기:"
echo "  python wireframe_portrait_processor.py --help"
echo ""
echo "✨ Portrait Outline Generator 환경 준비 완료!"
