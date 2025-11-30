#!/bin/bash
# Fast installation script for macOS
# hoping to avoid the xformers compilation error

set -e  # Exit on error

echo "========================================================================"
echo "Installing EHR Bias Analysis Environment"
echo "========================================================================"
echo ""

# Check Python version
echo "Step 1: Checking Python version..."
python3 --version
echo ""

# Create virtual environment
echo "Step 2: Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated"
echo ""

# Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"
echo ""

# Install core packages (fast, pre-built wheels)
echo "Step 5: Installing core packages..."
pip install -r requirements-core.txt
echo "✓ Core packages installed"
echo ""

# Install gensim (for Word2Vec in PCA analysis)
echo "Step 6: Installing gensim..."
pip install gensim --no-deps  # Skip dependencies to avoid issues
pip install smart-open  # Required by gensim
echo "✓ Gensim installed"
echo ""

# Try to install convokit (for Fighting Words)
echo "Step 7: Installing convokit..."
echo "Note: This might take a minute..."
pip install convokit || {
    echo "⚠ ConvoKit installation failed"
    echo "You can install it later manually if needed:"
    echo "  pip install convokit"
    echo ""
}
echo ""

# Download NLTK data
echo "Step 8: Downloading NLTK data..."
python3 << 'PYTHON_SCRIPT'
import nltk
print("Downloading punkt...")
nltk.download('punkt', quiet=True)
print("Downloading punkt_tab...")
nltk.download('punkt_tab', quiet=True)
print("Downloading stopwords...")
nltk.download('stopwords', quiet=True)
print("Downloading wordnet...")
nltk.download('wordnet', quiet=True)
print("✓ NLTK data downloaded")
PYTHON_SCRIPT
echo ""

# Test installation
echo "Step 9: Testing installation..."
python3 << 'PYTHON_SCRIPT'
import sys
success = True

packages = [
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('scipy', 'SciPy'),
    ('sklearn', 'Scikit-learn'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('statsmodels', 'Statsmodels'),
    ('nltk', 'NLTK'),
    ('gensim', 'Gensim'),
    ('openai', 'OpenAI'),
    ('dotenv', 'python-dotenv'),
]

for module, name in packages:
    try:
        __import__(module)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - FAILED")
        success = False

# Test ConvoKit separately (optional)
try:
    import convokit
    print(f"✓ ConvoKit")
except ImportError:
    print(f"⚠ ConvoKit - Not installed (optional)")

if not success:
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Installation completed successfully!"
    echo "========================================================================"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: source venv/bin/activate"
    echo "2. Copy .env file: cp .env.example .env"
    echo "3. Test data loader: python test_data_loader.py"
    echo "4. Start Jupyter: jupyter notebook"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "✗ Installation had errors"
    echo "========================================================================"
    echo ""
    echo "Some packages failed to install."
    echo "See errors above for details."
    echo ""
    exit 1
fi
