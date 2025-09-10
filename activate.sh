#!/bin/bash
# 프로젝트 환경 활성화 스크립트 (루트에서 스크립트 호출)

# 스크립트 실제 위치로 이동
cd "$(dirname "${BASH_SOURCE[0]}")"

# 실제 스크립트 실행 (exec 대신 source 사용)
source ./scripts/runtime/activate.sh "$@"