# Quick Start Guide


## 🚀 Setup (3 minutes)

### Automated Installation (Recommended)

```bash
# Run the installation script
./install.sh
```

This handles everything automatically and avoids compilation errors!

### Manual Installation (If script fails)

**Quick version:**
```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install core packages (no compilation!)
pip install -r requirements-core.txt

# Install optional packages
pip install gensim --no-deps
pip install smart-open
pip install convokit  # Or skip if it fails

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Set Up Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env (only need to add OpenAI key if doing summarization)
nano .env
```

**Minimal `.env` (for most analysis):**
```
OPENAI_API_KEY=your_key_here  # Only needed for GPT summarization
RANDOM_SEED=42
```

### 3. Test the Data Loader

```bash
python test_data_loader.py
```

**Expected output:**
```
✓ Loaded 100,000 records
✓ Data loaded successfully!
✓ SUCCESS!
```

---