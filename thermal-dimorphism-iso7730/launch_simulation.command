#!/bin/bash
# Double-click launcher (macOS)
# Works from any location — resolves path relative to this script
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
/usr/local/bin/python3 run_pipeline.py --simulate
echo ""
echo "Press Enter to close..."
read
