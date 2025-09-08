#!/bin/bash
# 프로젝트 전용 도구 설치 스크립트

PROJECT_ROOT="/home/joonhyoung-lee/바탕화면/arti_outlines"
TOOLS_DIR="$PROJECT_ROOT/tools"

# 환경 활성화
source "$PROJECT_ROOT/.env"

if [ $# -eq 0 ]; then
    echo "사용법: ./install_tool.sh <package_type> <package_name>"
    echo ""
    echo "예시:"
    echo "  ./install_tool.sh python requests"
    echo "  ./install_tool.sh node task-master-ai"  
    echo "  ./install_tool.sh pipx superclaude"
    echo "  ./install_tool.sh conda opencv"
    exit 1
fi

PACKAGE_TYPE="$1"
PACKAGE_NAME="$2"

case "$PACKAGE_TYPE" in
    "python"|"pip")
        echo "🐍 Python 패키지 '$PACKAGE_NAME'을 portrait_outline 환경에 설치..."
        pip install "$PACKAGE_NAME"
        echo "$PACKAGE_NAME" >> requirements.txt
        ;;
    "node"|"npm")
        echo "📦 Node.js 패키지 '$PACKAGE_NAME'을 프로젝트에 설치..."
        cd "$TOOLS_DIR"
        npm install "$PACKAGE_NAME"
        ;;
    "pipx")
        echo "🔧 pipx 도구 '$PACKAGE_NAME'을 conda 환경에 설치..."
        pip install "$PACKAGE_NAME"
        echo "$PACKAGE_NAME" >> requirements.txt
        ;;
    "conda")
        echo "🐍 Conda 패키지 '$PACKAGE_NAME'을 portrait_outline 환경에 설치..."
        conda install -c conda-forge "$PACKAGE_NAME" -y
        ;;
    *)
        echo "❌ 지원하지 않는 패키지 타입: $PACKAGE_TYPE"
        echo "지원: python, node, pipx, conda"
        exit 1
        ;;
esac

echo "✅ '$PACKAGE_NAME' 설치 완료!"