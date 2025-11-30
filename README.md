# Implicit Bias in AI-Generated EHR Discharge Summaries

**Stanford CS329R Final Project - RegLab Fellowship Submission**

This repository contains a comprehensive analysis of racial bias in Electronic Health Record (EHR) discharge instructions using multiple NLP techniques.

This project uses the MIMIC-IV dataset, which requires credentialed access. See [PhysioNet](https://physionet.org/content/mimiciv/3.1/) and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) for data access requirements.

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

The analysis reveals differences in word usage across racial groups. We also find that GPT-generated summaries diminish the contextual differences while maintaining underlying treatment disparities.



### Racial Disparities in Original EHR Discharge Summaries

**1. Language Patterns and Personalization**
- **Black and Hispanic patients**: High clustering in word usage patterns, suggesting less variety in symptom descriptions and greater focus on rote instructions
- **White patients**: More diverse language patterns, more personalized discharge summaries
- **Formality gap**: "Dr" appears significantly more often in White patient records compared to Black/Hispanic records, indicating doctors introduce themselves more completely or refer to care professionals more frequently for White patients

**2. Treatment Focus Differences (Fighting Words Analysis)**
- **White patients**: Frequently recommended "rehab," "therapy," "treatment" with emphasis on follow-up care and context for recommendations
- **Black and Hispanic patients**: Strong association with "dialysis," "diabetes," "sugar," "insulin" - focus on chronic disease management with direct prescriptions but less contextual explanation
- **Notable disparities**:
  - "Pain" correlates with "care" for White patients but "bleeding" for Black patients
  - "Alcohol" prominent in Hispanic discharge summaries but absent in White patient top 100 words
  - "Emergency" appears less urgent in Asian patient records
  - "Nurse," "recommend," and "treatment" more common in White records

**3. Intersectional Findings**
- Pain and nausea strongly associated with White women compared to White men
- Blood sugar disparities persist across gender within racial groups
- Hispanic men and women show concerning patterns suggesting health disparities in genital health

### AI-Generated Summary Analysis (GPT-3.5)

**4. Homogenization with Lost Context**
- **Surface uniformity masks inequality**: Generated summaries converge to similar, flat tone across all racial groups despite underlying records showing significant differences
- **Critical context loss**: AI summaries omit important details like "high blood pressure" and "insulin" mentions, stating only final treatment plans without motivating explanation
- **Perpetuation of existing bias**: While White patients receive context for treatment recommendations in original records, AI summaries extend the pattern of giving non-White patients direct prescriptions without justification
- **Tone issues**: Summaries are "overly confident, tonally callous, and not responsive to different health needs of different populations"
- **Impersonal approach**: Despite prompts emphasizing patient-centered care, summaries addressed medical professionals and discussed patients in abstract, impersonal terms

**Critical Conclusion**: "Identical treatment does not mean equitable treatment" - AI-generated summaries appear similar across groups but describe fundamentally different types of care (preventive/rehabilitative for White patients vs. chronic disease management for minorities). This apparent equality actually perpetuates and potentially exacerbates existing healthcare disparities. Consider the conclusion of the RegLab's Fairness through Difference Awareness paper.

**Recommendation**: General-purpose AI models should NOT be used to summarize health records without significant interrogation of the prompt, data, and healthcare system involved. Summarization requires human-in-the-loop feedback and cannot be automated without risk of harm.

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
