# 🏦 Bank Transaction Fraud Detection

A comprehensive fraud detection system using **unsupervised clustering** to generate labels from unlabeled bank transaction data, followed by **supervised learning** with both Machine Learning and Deep Learning models.

## 📋 Table of Contents

- [Business Background](#business-background)
- [Problem Statement](#problem-statement)
- [Research Questions](#research-questions)
- [Goals & Objectives](#goals--objectives)
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Clustering for Label Generation](#3-clustering-for-label-generation)
  - [4. Model Training](#4-model-training)
- [Models](#models)
  - [Machine Learning Models](#machine-learning-models)
  - [Deep Learning Models](#deep-learning-models)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This project tackles the challenge of **fraud detection in unlabeled bank transaction data**. Since the dataset lacks fraud labels, we employ an **unsupervised-supervised hybrid approach**:

1. **Unsupervised Learning**: Use clustering algorithms (K-Means, Isolation Forest) to identify anomalous transactions
2. **Supervised Learning**: Train ML and DL models using the generated labels
3. **Evaluation**: Compare performance across multiple models with proper validation

## Business Background
Perusahaan perbankan menghadapi tantangan dalam mendeteksi transaksi mencurigakan yang dapat mengakibatkan kerugian finansial dan menurunkan kepercayaan nasabah. Dengan volume transaksi digital yang terus meningkat, bank membutuhkan sistem deteksi penipuan yang akurat dan efisien. Dataset yang tersedia merupakan data transaksi perbankan tanpa label fraud, sehingga diperlukan pendekatan unsupervised learning untuk mengidentifikasi pola penipuan yang belum diketahui.

**Key Business Aspects:**
- **Sektor**: Perbankan dan Financial Services
- **Fokus**: Fraud detection pada transaksi digital
- **Data**: Transaksi perbankan real (unlabeled)
- **Tantangan Bisnis**: Mencegah financial loss, menjaga reputasi, compliance regulation

## Problem Statement
Bank saat ini **tidak memiliki sistem deteksi penipuan yang efektif** untuk mengidentifikasi transaksi mencurigakan secara real-time. Sistem existing mungkin mengandalkan rule-based approach yang mudah diakali oleh penipu canggih. Dengan data yang tidak memiliki label fraud, bank kesulitan untuk:

1. **Membedakan transaksi normal dan penipuan** secara akurat
2. **Mengidentifikasi pola penipuan baru** yang berkembang
3. **Melakukan pencegahan secara proaktif** sebelum kerugian terjadi
4. **Mengoptimalkan alokasi sumber daya** untuk investigasi fraud

**Konsekuensi Bisnis:**
- Potensi kerugian finansial hingga **miliaran rupiah**
- Menurunnya kepercayaan nasabah dan reputasi bank
- Risiko regulatory penalties dan compliance issues
- Inefisiensi dalam investigasi manual

## Research Questions

#### 1. Clustering & Anomaly Detection:
- Dapatkah metode clustering (K-Means, Isolation Forest) secara efektif mengidentifikasi transaksi mencurigakan dalam data unlabeled?
- Cluster manakah yang merepresentasikan pola fraud?

#### 2. Model Performance:
- Model machine learning manakah (Random Forest, Logistic Regression, ANN, DNN) yang paling akurat dalam mendeteksi fraud?
- Apakah deep learning models mengungguli traditional ML models untuk use case ini?

#### 3. Feature Importance:
- Fitur-fitur apa yang paling berpengaruh dalam mendeteksi transaksi penipuan?
- Variabel behavioral seperti apa yang paling indicative of fraudulent activity?

#### 4. Practical Implementation:
- Bagaimana membangun sistem fraud detection yang scalable dan real-time?
- Apa metrics yang paling relevan untuk mengevaluasi model fraud detection?

## Goals & Objectives

#### Primary Goals:
1. **Membangun sistem fraud detection** yang akurat menggunakan pendekatan hybrid (unsupervised + supervised learning)
2. **Mengidentifikasi pola transaksi mencurigakan** dari data historis
3. **Membandingkan performa multiple ML/DL models** untuk menentukan approach terbaik

#### Technical Objectives:
1. Mencapai **accuracy > 90%** pada test set
2. Mendapatkan **precision tinggi** untuk meminimalkan false positives
3. Mengidentifikasi **10+ feature penting** yang mendeteksi fraud
4. Membangun model yang dapat **berjalan secara real-time**

#### Business Objectives:
1. **Reduce financial losses** akibat fraud hingga **30%** dalam 6 bulan pertama
2. **Decrease false positive rate** untuk improve customer experience
3. **Enable proactive fraud detection** daripada reactive response
4. **Provide actionable insights** untuk tim risk management

## Features

- ✅ **Unsupervised Label Generation** using K-Means and Isolation Forest
- ✅ **2 Machine Learning Models**: Random Forest, Logistic Regression
- ✅ **2 Deep Learning Models**: ANN (4 layers), DNN (5 layers)
- ✅ **Comprehensive Feature Engineering**: Temporal, behavioral, and statistical features
- ✅ **Proper Evaluation**: Train/Validation/Test split (70/15/15)
- ✅ **Cross-Validation**: 5-fold CV for ML models
- ✅ **Visualization**: EDA, clustering results, training history, confusion matrices
- ✅ **Feature Importance Analysis**: Understanding key fraud indicators

## Dataset

**Source**: [Kaggle - Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)

### Dataset Statistics
- **Samples**: 2,512 transactions
- **Features**: 16 columns
- **Transaction Types**: Credit (23%), Debit (77%)
- **Channels**: Online, ATM, Branch
- **Time Period**: January 2023 - January 2024

### Key Features
| Feature | Description |
|---------|-------------|
| `TransactionID` | Unique transaction identifier |
| `AccountID` | Unique account identifier |
| `TransactionAmount` | Monetary value (0.26 - 1919.11) |
| `TransactionDate` | Timestamp of transaction |
| `TransactionType` | Credit or Debit |
| `Location` | U.S. city of transaction |
| `DeviceID` | Device used for transaction |
| `IP Address` | IPv4 address |
| `MerchantID` | Merchant identifier |
| `AccountBalance` | Post-transaction balance |
| `PreviousTransactionDate` | Last transaction timestamp |
| `Channel` | Transaction channel |
| `CustomerAge` | Account holder age |
| `CustomerOccupation` | Account holder occupation |
| `TransactionDuration` | Duration in seconds |
| `LoginAttempts` | Number of login attempts |

## Project Structure

```
fraud-detection/
│
├── data/
│   └── bank_transactions_data.csv       # Dataset file
│
├── notebooks/
│   └── fraud_detection.ipynb            # Main Jupyter notebook
│
├── models/
│   ├── random_forest_model.pkl          # Saved RF model
│   ├── logistic_regression_model.pkl    # Saved LR model
│   ├── ann_model.h5                     # Saved ANN model
│   └── dnn_model.h5                     # Saved DNN model
│
├── outputs/
│   ├── visualizations/                  # Generated plots
│   └── results/                         # Model performance results
│
├── src/
│   ├── preprocessing.py                 # Data preprocessing functions
│   ├── feature_engineering.py           # Feature creation functions
│   ├── clustering.py                    # Clustering algorithms
│   └── evaluation.py                    # Evaluation metrics
│
├── requirements.txt                      # Python dependencies
├── README.md                            # Project documentation
└── LICENSE                              # License file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tensorflow>=2.10.0
jupyter>=1.0.0
```

## Usage

### Option 1: Jupyter Notebook
```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

### Option 2: Python Script
```python
import pandas as pd
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.clustering import generate_labels
from src.models import train_all_models

# Load data
df = pd.read_csv('data/bank_transactions_data.csv')

# Preprocess and engineer features
X, feature_columns = create_features(df)

# Generate labels using clustering
y = generate_labels(X)

# Train models
results = train_all_models(X, y)

# Print results
print(results)
```

### Quick Start
1. Download dataset from Kaggle and place in `data/` folder
2. Open and run the Jupyter notebook
3. All cells will execute sequentially with visualizations
4. Results will be displayed at the end

## Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Median imputation for numerical features
- **Date Conversion**: Parse transaction timestamps
- **Data Validation**: Check for duplicates and outliers

### 2. Feature Engineering

#### Created Features:
- **Temporal Features**:
  - `Transaction_Hour`: Hour of day (0-23)
  - `Transaction_Day`: Day of month (1-31)
  - `Transaction_Month`: Month of year (1-12)
  - `Transaction_DayOfWeek`: Day of week (0-6)

- **Behavioral Features**:
  - `Time_Since_Last_Transaction`: Hours since last transaction
  - `Transaction_Frequency`: Number of transactions per account
  - `Balance_to_Amount_Ratio`: Balance / Transaction amount

- **Encoded Features**:
  - Label encoding for categorical variables (TransactionType, Location, Channel, CustomerOccupation)

### 3. Clustering for Label Generation

#### K-Means Clustering
```python
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```
- Divides data into 2 clusters (normal vs fraud)
- Smaller cluster assumed to be fraudulent
- Silhouette score used for quality assessment

#### Isolation Forest
```python
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)
```
- Specialized anomaly detection algorithm
- Identifies 10% of data as potential fraud
- Based on isolation principle (anomalies are easier to isolate)

### 4. Model Training

**Data Split**:
- Training: 70%
- Validation: 15%
- Test: 15%
- Stratified sampling to maintain class distribution

## Models

### Machine Learning Models

#### 1. Random Forest Classifier
```python
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```
- **Ensemble method** using 100 decision trees
- **Advantages**: Robust to overfitting, handles non-linear relationships
- **Feature importance**: Provides interpretability
- **Cross-Validation**: 5-fold CV

#### 2. Logistic Regression
```python
LogisticRegression(max_iter=1000, random_state=42)
```
- **Linear classifier** for binary classification
- **Advantages**: Fast, interpretable, good baseline
- **Regularization**: L2 regularization (default)
- **Cross-Validation**: 5-fold CV

### Deep Learning Models

#### 3. Feedforward Neural Network (ANN)
**Architecture**:
```
Input Layer (16 features)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Dense(16) + ReLU
    ↓
Dense(1) + Sigmoid
```
- **Optimizer**: Adam
- **Loss**: Binary Cross-Entropy
- **Early Stopping**: Patience = 10 epochs
- **Batch Size**: 32

#### 4. Deep Neural Network (DNN)
**Architecture**:
```
Input Layer (16 features)
    ↓
Dense(256) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + ReLU + BatchNorm + Dropout(0.4)
    ↓
Dense(64) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense(32) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Dense(16) + ReLU + Dropout(0.2)
    ↓
Dense(1) + Sigmoid
```
- **Deeper architecture** for complex pattern recognition
- **Higher dropout rates** in initial layers
- **Regularization**: Batch Normalization throughout

## Results

### Model Performance Comparison

| Model | Validation Accuracy | Test Accuracy | CV Mean Accuracy | Training Time |
|-------|-------------------|---------------|------------------|---------------|
| Random Forest | 0.9973 | 0.9920 | 0.9892 ± 0.0049 | ~10-15 sec |
| Logistic Regression | 1.0000 | 0.9920 | 0.9949 ± 0.0052 | ~2-5 sec |
| ANN | 0.9947 | 0.9761 | N/A | ~30-45 sec |
| DNN | 0.9947 | 0.9841 | N/A | ~45-60 sec |

*Note: Actual values depend on clustering results and random seed*

### Feature Importance (Top 10)

Based on Random Forest analysis:

1. `LoginAttempts` - High login attempts indicate suspicious behavior
2. `TransactionAmount` - Unusual transaction amounts
3. `AccountBalance` - Account balance patterns
4. `Balance_to_Amount_Ratio` - Ratio of balance to transaction
5. `Time_Since_Last_Transaction` - Transaction frequency patterns
6. `Transaction_Hour` - Unusual transaction timing
7. `Transaction_Frequency` - Account activity level
8. `TransactionDuration` - Transaction processing time
9. `CustomerAge` - Age-related patterns
10. `Channel_Encoded` - Transaction channel preference

## 📊 Evaluation Metrics

### Confusion Matrix Components
- **True Positive (TP)**: Correctly identified fraud
- **True Negative (TN)**: Correctly identified normal transactions
- **False Positive (FP)**: Normal transactions flagged as fraud
- **False Negative (FN)**: Fraud not detected (most critical!)

### Metrics Calculated
```python
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
```

### Why Multiple Metrics?
- **Accuracy**: Overall correctness
- **Precision**: Minimize false alarms
- **Recall**: Catch all fraud cases (critical for fraud detection!)
- **F1-Score**: Balance between precision and recall

## 💡 Key Insights

### 1. Clustering Quality
- K-Means Silhouette Score indicates cluster separation quality
- Isolation Forest provides alternative perspective on anomalies
- PCA visualization shows cluster boundaries

### 2. Feature Engineering Impact
- Temporal features capture time-based patterns
- Behavioral features (frequency, timing) are strong fraud indicators
- Ratio features normalize for account-specific characteristics

### 3. Model Comparison
- **Random Forest**: Best for interpretability and feature importance
- **Logistic Regression**: Fast baseline with decent performance
- **ANN/DNN**: Capture complex non-linear patterns
- **Ensemble potential**: Combining models could improve results

### 4. Fraud Indicators
- **High login attempts**: Strong fraud signal
- **Unusual transaction timing**: Late night/early morning
- **Transaction amount outliers**: Significantly higher than normal
- **Rapid successive transactions**: Short time between transactions

## 🔮 Future Improvements

### Model Enhancements
- [ ] **Hyperparameter Tuning**: Grid Search / Random Search / Bayesian Optimization
- [ ] **Class Imbalance Handling**: SMOTE, ADASYN, class weights
- [ ] **Ensemble Methods**: Stacking, Voting Classifier, Blending
- [ ] **Advanced Architectures**: LSTM for sequential patterns, Autoencoders

### Feature Engineering
- [ ] **Network Analysis**: Graph-based features from account/merchant networks
- [ ] **Aggregation Features**: Rolling statistics, moving averages
- [ ] **Interaction Features**: Feature combinations and polynomial features
- [ ] **Text Features**: NLP on location/merchant data if available

### Production Deployment
- [ ] **Model Serialization**: Save and load trained models
- [ ] **API Development**: FastAPI/Flask for real-time predictions
- [ ] **Monitoring Dashboard**: Track model performance over time
- [ ] **A/B Testing**: Compare model versions in production
- [ ] **Drift Detection**: Monitor for data/concept drift

### Scalability
- [ ] **Batch Processing**: Handle large transaction volumes
- [ ] **Distributed Training**: Use Spark MLlib for big data
- [ ] **Online Learning**: Update models with new data incrementally
- [ ] **GPU Acceleration**: Optimize deep learning training

## 📞 Creator

**Defrizal Yahdiyan Risyad** - defrijay@gmail.com
