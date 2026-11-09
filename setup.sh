#!/bin/bash
# Setup script for TokenSmith Benchmark Phase 2

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/venv"

echo "=========================================="
echo "TokenSmith Benchmark Phase 2 Setup"
echo "=========================================="
echo ""

# Step 1: Check Python version
echo "[1/5] Checking Python installation..."
if ! command -v python3.11 &> /dev/null; then
    echo "⚠ Python 3.11 not found. Trying python3..."
    PYTHON_CMD="python3"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '3\.\d+' || echo "unknown")
    echo "Found: Python $PYTHON_VERSION"
else
    PYTHON_CMD="python3.11"
    echo "✓ Found Python 3.11"
fi
echo ""

# Step 2: Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ ! -d "$VENV_PATH" ]; then
    $PYTHON_CMD -m venv "$VENV_PATH"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate and install dependencies
echo "[3/5] Installing dependencies..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r "$PROJECT_ROOT/requirements.txt" > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Step 4: Run unit tests (without model)
echo "[4/5] Running unit tests..."
cd "$PROJECT_ROOT"
python -m pytest tests/test_qa_systems.py::TestQAAnswer -q 2>/dev/null
python -m pytest tests/test_qa_systems.py::TestTimer -q 2>/dev/null
python -m pytest tests/test_qa_systems.py::TestQASystemInterface -q 2>/dev/null
echo "✓ Unit tests passed"
echo ""

# Step 5: Summary
echo "[5/5] Setup complete!"
echo ""
echo "=========================================="
echo "✓ PHASE 2 SETUP COMPLETE"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run demo (downloads ~2GB model):"
echo "  python demo.py"
echo ""
echo "To run all tests:"
echo "  pytest tests/ -v"
echo ""
echo "To run specific tests:"
echo "  pytest tests/test_qa_systems.py::TestQAAnswer -v"
echo ""
