# IMDB Sentiment Analysis: Comparative Study of Text Classification Models and Embeddings

**Group 2 - Text Classification Assignment**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models and Embeddings](#models-and-embeddings)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Experimental Results](#experimental-results)
- [Methodology](#methodology)
- [Team Contributions](#team-contributions)
- [Key Findings](#key-findings)
- [Usage Guide](#usage-guide)
- [References](#references)

## 🎯 Project Overview

This project presents a comprehensive comparative analysis of text classification systems for sentiment analysis using the IMDB movie review dataset. Our team investigated the performance of **four different model architectures** across **three word embedding techniques**, resulting in 12 unique model-embedding combinations.

### **Objectives**
1. Evaluate the effectiveness of traditional machine learning vs. deep learning approaches for sentiment classification
2. Compare the impact of different word embedding strategies (TF-IDF, Word2Vec Skip-gram, Word2Vec CBOW) on model performance
3. Identify optimal model-embedding combinations for sentiment analysis tasks
4. Provide reproducible research with comprehensive documentation and code

### **Problem Statement**
Sentiment analysis is a critical NLP task with applications in customer feedback analysis, social media monitoring, and market research. This project addresses the question: **"How do different word embedding techniques affect the performance of various neural network and traditional machine learning models on binary sentiment classification?"**

## 📊 Dataset

### **IMDB Movie Reviews Dataset**
- **Source**: [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Task**: Binary sentiment classification (Positive/Negative)
- **Total Samples**: 50,000 movie reviews
- **Distribution**: Balanced dataset (25,000 positive, 25,000 negative reviews)
- **Average Review Length**: ~250 words
- **Vocabulary Size**: ~88,000 unique words

### **Dataset Characteristics**
- **Class Balance**: 50% positive, 50% negative (perfectly balanced)
- **Train/Test Split**: 80/20 (40,000 training, 10,000 testing samples)
- **Language**: English
- **Domain**: Movie reviews from IMDB platform

### **Why IMDB Dataset?**
- Industry-standard benchmark for sentiment analysis research
- Large enough for deep learning models to learn meaningful representations
- Real-world text with diverse vocabulary and expression styles
- Enables direct comparison with existing literature

## 🔬 Models and Embeddings

### **Model Architectures**

| Model                   | Team Member                | Architecture Details                               | Parameters                        |
| ----------------------- | -------------------------- | -------------------------------------------------- | --------------------------------- |
| **Logistic Regression** | Liliane Umwanankabandi     | Traditional ML classifier with L2 regularization   | max_iter=1000, solver='liblinear' |
| **Simple RNN**          | Ntakirutimana Pretty Diane | Vanilla RNN with 32-64 units                       | 20 epochs, batch_size=64          |
| **LSTM**                | Jade ISIMBI TUZINDE        | Bidirectional LSTM (128 units) + Dropout (0.3-0.4) | 15 epochs, batch_size=64          |
| **GRU**                 | Oreste Abizera             | Bidirectional GRU (128 units) + Dropout (0.3-0.4)  | 15 epochs, batch_size=64          |

### **Embedding Techniques**

| Embedding              | Type        | Dimensions        | Key Parameters                               |
| ---------------------- | ----------- | ----------------- | -------------------------------------------- |
| **TF-IDF**             | Statistical | 500-5000 features | n-grams: (1,2), max_features varied by model |
| **Word2Vec Skip-gram** | Predictive  | 100               | window=5, sg=1, min_count=2                  |
| **Word2Vec CBOW**      | Predictive  | 100               | window=5, sg=0, min_count=2                  |

### **Model-Embedding Combinations**
Each team member evaluated their assigned model with all three embedding techniques:
- **Logistic Regression**: TF-IDF, Skip-gram, CBOW
- **Simple RNN**: TF-IDF, Skip-gram, CBOW
- **LSTM**: TF-IDF, Skip-gram, CBOW
- **GRU**: TF-IDF, Skip-gram, CBOW

**Total Experiments**: 12 unique model-embedding combinations

## 📁 Repository Structure

```
ml-draft/
├── README.md                          # This file - comprehensive project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   └── IMDB Dataset.csv              # IMDB movie reviews dataset
├── notebooks/
│   ├── Logistic_Regression.ipynb     # LR experiments (TF-IDF, Skip-gram, CBOW)
│   ├── RNN_model.ipynb               # Simple RNN experiments with all embeddings
│   ├── LSTM.ipynb                    # LSTM experiments with all embeddings
│   ├── GRU.ipynb                     # GRU experiments with all embeddings
│   └── template_experiment.ipynb     # Template for new experiments
└── .gitignore                        # Git ignore file
```

## 🚀 Installation and Setup

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU optional but recommended for deep learning models

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/oreste-abizera/Group2-IMDB-Text-Classification-Analysis.git
cd Group2-IMDB-Text-Classification-Analysis
```

### **Step 2: Create Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Download NLTK Data**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

### **Step 5: Verify Installation**
```bash
python -c "import tensorflow; import sklearn; import gensim; print('All dependencies installed successfully!')"
```

### **Dependencies Overview**
- **pandas** (2.x): Data manipulation and analysis
- **numpy** (1.24+): Numerical computing
- **scikit-learn** (1.3+): Traditional ML algorithms and metrics
- **matplotlib** & **seaborn**: Data visualization
- **nltk** (3.8+): Natural language processing utilities
- **gensim** (4.x): Word2Vec implementation
- **tensorflow** (2.x): Deep learning framework
- **jupyter**: Interactive notebook environment
- **wordcloud**: Text visualization


## 🔬 Methodology

### **1. Data Preprocessing Pipeline**

All experiments follow a consistent preprocessing strategy to ensure fair comparison:

```python
def clean_text(text):
    """
    Standardized text cleaning function used across all experiments
    """
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 3. Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenization
    tokens = word_tokenize(text)
    
    # 5. Remove stopwords (optional based on embedding)
    # Note: Stopword removal is SKIPPED for Word2Vec embeddings
    # to preserve context for better word representations
    
    return tokens
```

### **2. Embedding-Specific Preprocessing**

| Embedding | Stopword Removal | Vectorization Method         | Output Format                          |
| --------- | ---------------- | ---------------------------- | -------------------------------------- |
| TF-IDF    | ✅ Yes            | TfidfVectorizer              | Sparse matrix (n_samples × n_features) |
| Skip-gram | ❌ No             | Mean pooling of word vectors | Dense array (n_samples × 100)          |
| CBOW      | ❌ No             | Mean pooling of word vectors | Dense array (n_samples × 100)          |

### **3. Model Training Configuration**

#### **Logistic Regression**
- Solver: liblinear
- Regularization: L2 (C=1.0)
- Max iterations: 1000
- No early stopping

#### **Deep Learning Models (RNN, LSTM, GRU)**
- **Common architecture**:
  - Embedding layer (for Word2Vec) or direct input (for TF-IDF)
  - Bidirectional RNN/LSTM/GRU layer (128 units)
  - Dropout layer (0.3-0.4)
  - Dense output layer with sigmoid activation
- **Training parameters**:
  - Optimizer: Adam
  - Loss: Binary crossentropy
  - Epochs: 15-20
  - Batch size: 64
  - Validation split: 20% of training data
  - Early stopping: Monitor val_loss with patience=3
  - Max sequence length: 200 tokens

### **4. Evaluation Metrics**

All models are evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: Positive predictive value
- **Recall**: True positive rate (Sensitivity)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### **5. Cross-Model Comparison Framework**

To ensure fair comparison:
1. **Same random seed** (42) for all train-test splits
2. **Identical preprocessing** for each embedding type
3. **Standardized evaluation** metrics across all models
4. **Consistent hyperparameter tuning** approach
5. **Multiple runs** to account for randomness (where applicable)

## 👥 Team Contributions

### **Team Members and Responsibilities**

| Team Member                    | Model               | Embeddings Implemented  | Key Contributions                                   |
| ------------------------------ | ------------------- | ----------------------- | --------------------------------------------------- |
| **Oreste Abizera**             | GRU                 | TF-IDF, Skip-gram, CBOW | Repository setup, GRU implementation, documentation |
| **Jade ISIMBI TUZINDE**        | LSTM                | TF-IDF, Skip-gram, CBOW | LSTM architecture, bidirectional implementation     |
| **Liliane Umwanankabandi**     | Logistic Regression | TF-IDF, Skip-gram, CBOW | Traditional ML baseline, statistical analysis       |
| **Pretty Diane Ntakirutimana** | Simple RNN          | TF-IDF, Skip-gram, CBOW | RNN implementation, performance benchmarking        |

### **Shared Responsibilities**
- **Data preprocessing pipeline design**: All members
- **Exploratory data analysis**: All members
- **Literature review**: All members
- **Report writing**: All members
- **Code review and testing**: All members
- **Results compilation and visualization**: All members

### **Individual Workload Distribution**
Each team member contributed approximately **25%** of the total project work, with individual focus on:
- Model architecture design and implementation (40%)
- Embedding integration and experimentation (30%)
- Result analysis and documentation (20%)
- Peer review and collaboration (10%)


### **Lessons Learned**

- **Preprocessing consistency is crucial** for fair comparison
- **Hyperparameter tuning** significantly impacts deep learning model performance
- **Baseline models** (Logistic Regression) should never be underestimated
- **Context preservation** in Word2Vec requires careful stopword handling

## 📖 Usage Guide

### **Running Individual Experiments**

#### **1. Logistic Regression Experiments**
```bash
jupyter notebook notebooks/Logistic_Regression.ipynb
```
This notebook contains:
- TF-IDF vectorization with n-grams
- Word2Vec mean pooling aggregation
- Classification reports and confusion matrices

#### **2. RNN Experiments**
```bash
jupyter notebook notebooks/RNN_model.ipynb
```
Features:
- Vanilla RNN implementation
- All three embedding comparisons
- Training history visualization

#### **3. LSTM Experiments**
```bash
jupyter notebook notebooks/LSTM.ipynb
```
Includes:
- Bidirectional LSTM architecture
- Dropout regularization
- Performance metrics across embeddings

#### **4. GRU Experiments**
```bash
jupyter notebook notebooks/GRU.ipynb
```
Contains:
- Bidirectional GRU implementation
- Comparative analysis with LSTM
- Embedding performance evaluation

### **Creating New Experiments**

Use the provided template to add new models or embeddings:

```bash
# Copy template
cp notebooks/template_experiment.ipynb notebooks/your_experiment.ipynb

# Open in Jupyter
jupyter notebook notebooks/your_experiment.ipynb
```

The template includes:
- Pre-configured data loading
- Exploratory data analysis (EDA) visualizations
- Preprocessing pipeline
- Embedding preparation sections
- Model training and evaluation framework
- Results saving functionality

### **Reproducing Results**

To reproduce all experiments:

```bash
# Ensure data is in place
ls data/IMDB\ Dataset.csv

# Run all notebooks in order
jupyter nbconvert --execute --to notebook \
  notebooks/Logistic_Regression.ipynb \
  notebooks/RNN_model.ipynb \
  notebooks/LSTM.ipynb \
  notebooks/GRU.ipynb
```

## 📝 License

This project is part of an academic assignment and is intended for educational purposes.

## 🤝 Acknowledgments

- **Course Instructor**: Samiratu Ntohsi
- **Institution**: African Leadership University (ALU)
- **Course**: Machine Learning Techniques 1
- **Assignment**: Comparative Analysis of Text Classification with Multiple Embeddings

---

## 📧 Contact

For questions or collaboration:
- **Repository**: [GitHub Link](https://github.com/oreste-abizera/Group2-IMDB-Text-Classification-Analysis)
- **Team Lead**: Oreste Abizera

---

**Last Updated**: February 8, 2026
