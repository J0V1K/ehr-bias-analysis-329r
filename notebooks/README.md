# Analysis Notebooks

This directory contains the cleaned and documented Jupyter notebooks for the EHR bias analysis project.

## Notebooks

### 01_fighting_words_analysis.ipynb (12KB)
**Fighting Words Analysis with updated FDR**

Identifies statistically significant differences in word usage across racial groups in discharge instructions.

**Key Features:**
- Fighting Words algorithm (Monroe et al., 2008)
- Benjamini-Hochberg FDR correction
- Effect size calculation
- Comprehensive interpretation with caveats

**What's New:**
- Added FDR correction for multiple comparisons
- Integrated `statistical_analysis.py` module
- No Google Colab dependencies
- Uses `src/data_loader.py`

---

### 02_pca_visualization.ipynb (8KB)
**Word2Vec PCA Visualization**

Trains Word2Vec embeddings on discharge instructions and visualizes word distributions by race using PCA.

**Key Features:**
- Word2Vec skip-gram models
- PCA dimensionality reduction
- Visual comparison across racial groups

**What's New:**
- Fixed critical lambda bug (would crash)
- Added markdown documentation
- Cleaned text processing

---

### 03_sentiment_analysis.ipynb (15KB)
**Sentiment Analysis with DistilBERT**

Applies pre-trained sentiment analysis to assess tone differences in discharge instructions across racial groups.

**Key Features:**
- DistilBERT sentiment classification
- Statistical significance testing (chi-square, Mann-Whitney U)
- Comprehensive limitations discussion

**What's Fixed:**
- Completely rewritten from 42-cell mess to focused 16-cell analysis
- Removed Google Colab code
- Added statistical testing
- Discusses medical context limitations


---

### 04_gpt_summarization.ipynb (11KB)
**GPT-3.5 Discharge Summary Generation**

Generates discharge summaries using GPT-3.5 for comparison with clinician-written instructions.

**Key Features:**
- GPT-3.5-turbo API integration
- Prompt engineering for medical summarization
- Cost estimation and ethical considerations

**What's Fixed:**
- Added proper error handling
- No Google Colab dependencies

---

## Original Notebooks

Backups of the original notebooks are stored in `originals/` for reference.

---

## Running the Notebooks

### Prerequisites:
```bash
# Install environment
./install.sh
source venv/bin/activate

# Create .env file (for notebook 04)
cp .env.example .env
# Edit .env to add OPENAI_API_KEY
```

### Launch Jupyter:
```bash
jupyter notebook
```

### Or run from command line:
```bash
# Execute a notebook
jupyter nbconvert --execute --to notebook notebooks/01_fighting_words_analysis.ipynb

# Clear outputs
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
```

---

## For Reviewers

1. Start with **01_fighting_words_analysis.ipynb** - the core statistical analysis
2. The FDR correction (section 6) addresses the main methodological critique
3. Each notebook includes interpretation and limitations sections
4. All code should be ready to run on local machine. Originally, it ran on Colab.

---
