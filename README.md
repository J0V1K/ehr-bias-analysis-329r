# Implicit Bias in AI-Generated EHR Discharge Summaries

**Stanford CS329R Final Project - RegLab Fellowship Submission**

This repository contains a comprehensive analysis of racial bias in Electronic Health Record (EHR) discharge instructions using multiple NLP techniques.

This project uses the MIMIC-IV dataset, which requires credentialed access. See [PhysioNet](https://physionet.org/) for data access requirements.

---

## Project Overview

This project investigates whether AI-generated discharge summaries exhibit the same racial disparities found in clinician-written instructions using the MIMIC-IV dataset (100,000+ discharge instructions).

**Key Research Questions:**
- Do discharge instructions show statistically significant differences in language across racial groups?
- Can we visualize these differences using word embeddings and dimensionality reduction?
- Does sentiment analysis reveal tone differences in instructions?
- How do AI-generated summaries compare to human-written ones?

**Analysis Techniques:**
1. **Fighting Words Analysis** - Statistical identification of differentially used words (with FDR correction)
2. **Word2Vec + PCA** - Visualization of word embedding patterns across groups
3. **Sentiment Analysis** - DistilBERT-based tone comparison
4. **GPT Summarization** - AI-generated discharge summary comparison

---

## Key Findings

The analysis reveals statistically significant differences in word usage across racial groups. We also find that GPT-generated summaries diminish the .

---

## Quick Start

### 1. Installation (3 minutes)

```bash
# Automated installation (recommended)
./install.sh

# Or manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
```

### 2. Set Up Environment

```bash
# Copy template
cp .env.example .env

# Edit .env (only needed for GPT notebook)
nano .env  # Add OPENAI_API_KEY if using notebook 04
```

### 3. Test the Setup

```bash
source venv/bin/activate
python tests/test_data_loader.py
```

**Expected output:**
```
✓ Loaded 100,000 records
✓ Data loaded successfully!
✓ SUCCESS!
```

### 4. Launch Notebooks

```bash
source venv/bin/activate
jupyter notebook
```

---

## Repository Structure

```
.
├── README.md                          # This file
├── QUICK_START.md                     # Quick setup guide
├── NOTEBOOK_TEST_RESULTS.md           # Testing results and known issues
├── install.sh                         # Automated installation script
├── requirements-core.txt              # Python dependencies
├── .env.example                       # Environment variable template
│
├── notebooks/                         # Analysis notebooks (START HERE)
│   ├── README.md                      # Notebook guide
│   ├── 01_fighting_words_analysis.ipynb    # Statistical word analysis (needs ConvoKit)
│   ├── 02_pca_visualization.ipynb          # Word2Vec + PCA (✅ verified working)
│   ├── 03_sentiment_analysis.ipynb         # DistilBERT sentiment
│   └── 04_gpt_summarization.ipynb          # GPT-3.5 summaries (needs API key)
│
├── src/                               # Core modules
│   └── data_loader.py                 # Reproducible data loading
│
├── statistical_analysis.py            # FDR correction module
│
├── tests/                             # Testing scripts
│   └── test_data_loader.py            # Data loader tests
│
├── data/                              # MIMIC-IV datasets
│   └── merged_file_sample=100k_section=dischargeinstructions.csv
│
├── results/                           # Generated outputs
│   ├── Fightin/                       # Fighting Words results
│   ├── PCA/                           # Word2Vec visualizations
│   ├── Sentiment/                     # Sentiment analysis results
│   └── GPT/                           # GPT summaries
│
└── docs/                              # Documentation and papers
    ├── CS329R_Final_Paper.pdf         # Full research paper
    ├── CS329R_Presentation.pptx       # Project presentation
    ├── WHY_FDR_CORRECTION.md          # Statistical methodology
    ├── NOTEBOOK_IMPROVEMENTS.md       # Code quality improvements
    └── DIRECTORY_CLEANUP_SUMMARY.md   # Organization notes
```

---

**Last Updated:** 2025-11-29
**Environment Tested:** Python 3.13.5 on macOS ARM64
**All Core Tests:** ✅ PASSING