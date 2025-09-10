#!/bin/bash
# Portrait Outline 프로젝트 전용 도구 설치 스크립트

# 스크립트 위치를 기준으로 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
TOOLS_DIR="$PROJECT_ROOT/.tools"
ENV_NAME="portrait_outline"

echo "=== Portrait Outline 프로젝트 도구 설치 ==="

# 1. 프로젝트 내 도구 디렉토리 생성
echo "1. 프로젝트 도구 디렉토리 생성 중..."
mkdir -p "$TOOLS_DIR/bin"
mkdir -p "$TOOLS_DIR/venvs"

# 2. 현재 conda 환경에 SuperClaude 설치
echo "2. portrait_outline 환경에 SuperClaude 설치 중..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# SuperClaude를 현재 환경에 설치
pip install superclaude

# 3. 프로젝트 전용 PATH 설정
echo "3. 프로젝트 전용 환경 설정 생성 중..."
cat > "$PROJECT_ROOT/.env" << 'EOF'
#!/bin/bash
# Portrait Outline 프로젝트 환경 설정

# Conda 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate portrait_outline

# 프로젝트 루트 디렉토리 (동적으로 설정)
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export PORTRAIT_OUTLINE_ROOT="\$SCRIPT_DIR"

# 프로젝트 도구 PATH 추가 (conda 환경 내 도구 우선)
export PATH="\$PORTRAIT_OUTLINE_ROOT/.tools/bin:\$PATH"

# 프로젝트 작업 디렉토리로 이동
cd "\$PORTRAIT_OUTLINE_ROOT"

echo "✅ Portrait Outline 환경 활성화됨"
echo "📍 현재 디렉토리: $(pwd)"
echo "🐍 Python: $(which python)"
echo "📦 Conda 환경: $CONDA_DEFAULT_ENV"
EOF

# 4. 편의 스크립트 생성
echo "4. 편의 스크립트 생성 중..."
cat > "$PROJECT_ROOT/activate.sh" << 'EOF'
#!/bin/bash
# 프로젝트 환경 활성화
source .env
EOF

chmod +x "$PROJECT_ROOT/activate.sh"
chmod +x "$PROJECT_ROOT/.env"

echo "=== 설치 완료 ==="
echo ""
echo "🚀 사용법:"
echo "  source activate.sh       # 환경 활성화"
echo "  superclaude install       # SuperClaude 설치"
echo "  python <script.py>        # 스크립트 실행"
echo ""
echo "📁 모든 도구가 portrait_outline 환경에 격리됨"