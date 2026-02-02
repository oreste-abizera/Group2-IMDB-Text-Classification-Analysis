# Group 2: IMDB Text Classification Analysis

## Objective
Comparative analysis of text classification systems using various embeddings (TF-IDF, Skip-gram (Word2Vec), CBOW (Word2Vec)) and models (Traditional ML, RNN, LSTM, GRU).

## Repository Structure
- `data/`: Place your dataset file (`IMDB Dataset.csv`) here.
- `notebooks/`: Individual experiment notebooks. **Start by copying `template_experiment.ipynb`.**
- `models/`: Save your trained model weights here.
- `results/`: CSV files containing experiment metrics.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/oreste-abizera/Group2-IMDB-Text-Classification-Analysis.git
   cd Group2-IMDB-Text-Classification-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Start Your Experiment

1. **Copy the Template**:
   Duplicate `notebooks/template_experiment.ipynb` and rename it:
   `firstname_model_embedding.ipynb` (e.g., `oreste_gru_tfidf.ipynb`).

2. **Run the Notebook**:
   - The template includes all necessary steps:
     - **Loads Data**: Automatically loads `data/IMDB Dataset.csv`.
     - **EDA**: Visualizes class distribution and text lengths.
     - **Preprocessing**: Cleans text (lowercase, HTML removal, etc.) and tokenizes.
   
3. **Implement Your Model**:
   - Fill in the **Embedding Layer** and **Model Architecture** sections.
   - Train and Evaluate your model.
   - Uncomment the save results block to store your metrics.

## Preprocessing Strategy
The notebook includes a `clean_text` function that:
- Lowercases text
- Removes HTML tags
- Removes special characters
- Tokenizes and removes stopwords (can be toggled for embeddings like Word2Vec/GloVe if needed)

## Contribution Tracker
Ensure you fill out the group contribution tracker as required by the assignment.
**[Link to your tracker here (Make a copy of the official one)]**


## Contribution Guidelines
- Ensure your notebook runs from top to bottom before committing.
- Update the `results/` folder with your findings.
