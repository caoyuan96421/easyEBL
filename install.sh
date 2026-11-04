#!/usr/bin/env bash
# Create .venv next to this script and install easyEBL (Linux / macOS).
#   ./install.sh [--recreate] [--tests] [--extras all,test] [--python /path/to/python3]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
EXTRAS="all,test"; RECREATE=0; TESTS=0; PY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) RECREATE=1 ;;
    --tests) TESTS=1 ;;
    --extras) EXTRAS="$2"; shift ;;
    --python) PY="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac; shift
done
if [[ -z "$PY" ]]; then
  for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,13)))'; then PY="$c"; break; fi
  done
fi
[[ -n "$PY" ]] || { echo "Need Python 3.10-3.13 (or pass --python)"; exit 1; }
echo "Using $("$PY" --version) ($PY)"
[[ $RECREATE -eq 1 ]] && rm -rf "$VENV"
[[ -x "$VENV/bin/python" ]] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -e "$ROOT[$EXTRAS]"
"$VENV/bin/python" -c "import easyebl, sys; print('easyEBL', easyebl.__version__, 'on Python', sys.version.split()[0])"
if [[ $TESTS -eq 1 ]]; then QT_QPA_PLATFORM=offscreen "$VENV/bin/python" -m pytest -q; fi
echo "Activate: source .venv/bin/activate   Demo: .venv/bin/easyebl-demo"
