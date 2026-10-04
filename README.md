# 🏦 Bank Transaction Fraud Detection

![Cover](assets/cover.jpg)

**In short:** I built a way to spot suspicious bank transactions, even though the bank has no confirmed fraud examples. First, two methods (K-Means and Isolation Forest) mark the unusual transactions. This gives us a stand-in ("proxy") fraud label. Then four models learn from that label: two standard Machine Learning models and two Deep Learning models. I compare them on accuracy, cost of mistakes, key warning signs, and speed.

## Table of Contents

1. [Background](#1-background)
2. [Business Questions](#2-business-questions)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
   - 4.1 [Models](#41-models)
5. [Tools](#5-tools)
6. [Insight](#6-insight)
   - 6.1 [Model Performance](#61-model-performance)
   - 6.2 [Top Fraud-Indicating Features](#62-top-fraud-indicating-features)
   - 6.3 [Cost of Getting It Wrong](#63-cost-of-getting-it-wrong)
   - 6.4 [Real-Time Feasibility](#64-real-time-feasibility)
   - 6.5 [Key Takeaways (Answers to the Business Questions)](#65-key-takeaways-answers-to-the-business-questions)
7. [Recommendations](#7-recommendations)
8. [Conclusion](#8-conclusion)
9. [Limitations](#9-limitations)
10. [Creator](#10-creator)

---

## 1. Background

Digital banking made payments faster and easier, but it also gave fraudsters more ways in. Every fraud that slips through means lost money for the bank. Repeated cases also hurt something harder to win back: customer trust.

Today the bank mostly uses fixed rules. A transaction is flagged only if it crosses a set limit. That catches known tricks, but fraudsters learn to stay under the limit.

There is also a bigger problem: the bank has no list of past transactions confirmed as fraud. So there is no "answer key" to teach a smarter system. We first have to work out which transactions look suspicious, and only then can we train a system to catch them.

If this stays unsolved, the bank risks lost money, lost customer trust, trouble with regulators, and investigators spending time on reviews that a smarter system could sort automatically.

---

## 2. Business Questions

Four questions guide the whole project.

| # | Question | Why it matters |
|---|---|---|
| **BQ1** | Can spotting unusual behavior find fraud when we have no labels, and does the result make business sense? | If the fraud label is not trustworthy, every later result is built on sand. |
| **BQ2** | Which model gives the best balance between catching fraud (recall) and not bothering good customers (precision)? | This decides which model gets used. |
| **BQ3** | What fraud warning signs can the risk team act on? | It turns a model into alert rules and checklists that investigators can use. |
| **BQ4** | Is it practical to run in production, and which measure should drive the decision? | It links model quality to real-world limits and the real cost of mistakes. |

---

## 3. Dataset

**Source:** [Kaggle – Bank Transaction Dataset for Fraud Detection](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection)

| Stat | Value |
|---|---|
| Samples | 2,512 transactions |
| Features | 16 columns, no missing values |
| Transaction amount | $0.26 – $1,919.11 (average ≈ $298) |
| Customer age | 18 – 80 years (average ≈ 45) |
| Login attempts | 1 – 5 per transaction (95 transactions with 3 or more, 3.8% of the data) |
| Channels | Online, ATM, Branch |
| Time period | January 2023 – January 2024 |

| Feature | Description |
|---|---|
| `TransactionID` | Unique transaction ID |
| `AccountID` | Unique account ID |
| `TransactionAmount` | Money amount of the transaction |
| `TransactionDate` | Date and time of the transaction |
| `TransactionType` | Credit or Debit |
| `Location` | U.S. city of the transaction |
| `DeviceID` | Device used |
| `IP Address` | IPv4 address |
| `MerchantID` | Merchant ID |
| `AccountBalance` | Balance after the transaction |
| `PreviousTransactionDate` | Date and time of the previous transaction |
| `Channel` | Online, ATM, or Branch |
| `CustomerAge` | Age of the account holder |
| `CustomerOccupation` | Job of the account holder |
| `TransactionDuration` | How long the transaction took, in seconds |
| `LoginAttempts` | Number of login attempts |

> **Data quality note:** `PreviousTransactionDate` did not make sense. In all 2,512 rows, the "previous" date was *later* than the transaction itself, and every value fell within a 6-minute window on 2024-11-04. It looks like an export timestamp, not a real history. So I replaced it with `Time_Since_Last_Transaction`, calculated from each account's own sorted transactions.

![Exploratory data overview](assets/eda_overview.png)

---

## 4. Methodology

**Step 1 – Prepare the data.** Parse the dates, check for missing values (none found) and outliers, and recalculate the time since each account's last transaction.

**Step 2 – Build new features**

| Category | Features |
|---|---|
| Time | `Transaction_Hour`, `Transaction_Day`, `Transaction_Month`, `Transaction_DayOfWeek` |
| Behavior | `Time_Since_Last_Transaction`, `Transaction_Frequency`, `Balance_to_Amount_Ratio` |
| Encoded | Categories turned into numbers (`TransactionType`, `Location`, `Channel`, `CustomerOccupation`) |

Two feature sets are used for two jobs:
- **Behavior only (8 features)** is used to find unusual transactions.
- **All features (16)** is used to train the models.

**Step 3 – Create a working fraud label**
- **K-Means on all features (including age and job) failed.** The groups just followed `CustomerOccupation` and `CustomerAge`, not fraud (clarity score, called silhouette score, only 0.099).
- **K-Means on behavior features only worked better.** The score rose to 0.439, and the split was driven almost entirely by `LoginAttempts`. It flagged 95 transactions.
- **Isolation Forest** (`contamination=0.1`) was then run on the same behavior features. It flagged 252 transactions (10.0% of the data).
- **Cross-check:** the agreement score between the two methods (Cohen's kappa) is 0.448, which is moderate. **87% of the transactions flagged by K-Means were also flagged by Isolation Forest.** That is real evidence the label is based on behavior and is not random.
- **Isolation Forest on behavior features became the working label.** It is a stand-in, not confirmed fraud.

![Isolation Forest flagged-vs-normal profile](assets/isolation_forest_profile_ratio.png)
![PCA view of flagged vs. normal transactions](assets/pca_visualization.png)

**Step 4 – Train the models.** The data is split into 70% train (1,759), 15% validation (376), and 15% test (377). Each split keeps about 10% flagged transactions.

### 4.1 Models

| Model | Type | Notes |
|---|---|---|
| Random Forest (200 trees) | Machine Learning | Handles complex patterns and shows which features matter. Classes balanced; checked with 5-fold cross-validation. |
| Logistic Regression | Machine Learning | Fast and easy to explain. Classes balanced; checked with 5-fold cross-validation. |
| ANN (4 hidden layers: 128→64→32→16) | Deep Learning | ReLU + BatchNorm + Dropout, sigmoid output, Adam optimizer, binary cross-entropy loss, about 9x weight on the flagged class, early stopping (patience 10), batch size 32. |
| DNN (5 hidden layers: 256→128→64→32→16) | Deep Learning | Deeper than the ANN, with more dropout in the early layers. |

---

## 5. Tools

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| Data handling | pandas, numpy |
| Charts | matplotlib, seaborn |
| Standard ML | scikit-learn (KMeans, IsolationForest, RandomForestClassifier, LogisticRegression) |
| Deep Learning | TensorFlow/Keras |
| Environment | Jupyter Notebook |

---

## 6. Insight

### 6.1 Model Performance

Results on the test set (377 transactions):

| Model | Accuracy | Precision (Flagged) | Recall (Flagged) | F1 (Flagged) | ROC-AUC | Time per transaction |
|---|---|---|---|---|---|---|
| Random Forest | 0.950 | 0.913 | 0.553 | 0.689 | 0.980 | 0.045 ms |
| Logistic Regression | 0.934 | 0.618 | 0.895 | 0.731 | 0.975 | 0.002 ms |
| ANN | 0.958 | 0.775 | 0.816 | 0.795 | 0.982 | 0.916 ms |
| DNN | 0.968 | 0.810 | 0.895 | 0.850 | 0.988 | 0.915 ms |

**How to read this:** *Precision* = of the transactions a model flags, how many are right. *Recall* = of all the transactions that should be flagged, how many it catches. *F1* = one score that balances the two.

*These numbers show how well each model learned the stand-in label, not confirmed real-world fraud. See [Limitations](#9-limitations).*

![Precision/recall/F1 and training time by model](assets/model_metrics_comparison.png)
![Confusion matrices for all four models](assets/confusion_matrices.png)

Each model makes a different trade-off:
- **Random Forest** has the highest precision: few false alarms, but it misses more fraud.
- **Logistic Regression** has high recall: it catches more fraud, but raises more false alarms.
- **DNN** has the best balance. It has the best F1 (0.850), the best ROC-AUC (0.988), and the best accuracy (0.968). It does *not* win on every metric: Random Forest has higher precision, and Logistic Regression ties it on recall.
- **ANN** is close behind DNN.

Deep learning gives a real gain here: F1 is +15% for ANN and +23% for DNN compared with Random Forest.

### 6.2 Top Fraud-Indicating Features

Ranked by Random Forest feature importance:

| Rank | Feature | Category |
|---|---|---|
| 1 | `LoginAttempts` | Behavioral |
| 2 | `Balance_to_Amount_Ratio` | Behavioral |
| 3 | `TransactionAmount` | Transaction |
| 4 | `Time_Since_Last_Transaction` | Behavioral |
| 5 | `AccountBalance` | Account |
| 6 | `TransactionDuration` | Transaction |
| 7 | `Transaction_Frequency` | Behavioral |
| 8 | `Transaction_Hour` | Temporal |
| 9 | `CustomerAge` | Demographic |
| 10 | `Location_Encoded` | Transaction |

![Top 10 fraud indicators](assets/feature_importance.png)

Grouped another way, **account-level behavior (how the account has acted over time) makes up 66.8% of the signal, and transaction-level traits (what one transaction looks like) make up 33.2%.** Fraud here looks less like "this one transaction is odd" and more like "this account is acting differently than usual."

![Account-level vs. transaction-level importance share](assets/transaction_vs_account_level.png)

**A simple rule that needs no model:** `LoginAttempts ≥ 1` **and** `Balance_to_Amount_Ratio ≥ 156` (both are 90th-percentile cutoffs). It catches 88 of the 252 flagged transactions (35%). Note: every transaction in this data has at least 1 login attempt, so in practice the rule depends on the balance ratio.

### 6.3 Cost of Getting It Wrong

A missed fraud costs far more than a false alarm. These figures are examples: $500 per missed fraud and $15 per false alarm. When we count cost, the model ranking changes:

| Model | Missed fraud | False alarms | Estimated cost ($) |
|---|---|---|---|
| DNN | 4 | 8 | 2,120 |
| Logistic Regression | 4 | 21 | 2,315 |
| ANN | 7 | 9 | 3,635 |
| Random Forest | 17 | 2 | 8,530 |

![Estimated cost per model](assets/cost_comparison.png)

Random Forest's high precision backfires. It avoids false alarms, but the 17 frauds it misses cost far more. So it is the **most expensive** model, even with a decent accuracy score. Accuracy alone is the wrong yardstick for fraud. **DNN has the lowest cost**, and Logistic Regression is a close second and much cheaper to train.

### 6.4 Real-Time Feasibility

Every model scores one transaction in well under a millisecond on an ordinary CPU. Logistic Regression can score about 433,000 transactions per second on one core. Even the slowest models (ANN and DNN) handle over 1,000 per second. That fits comfortably inside a typical real-time approval window.

So speed is not the problem. The real gate to production is checking that the working label is right and setting the alert level from real costs.

### 6.5 Key Takeaways (Answers to the Business Questions)

- **BQ1 – Can we find fraud without labels?** Yes, but only the right way. Using every column, including age and job, misleads: it finds customer groups, not fraud. Using only behavior columns and checking with two methods (87% overlap) gives a label we can build on.
- **BQ2 – Which model works best?** DNN has the best overall balance and the lowest cost, with ANN close behind. Random Forest looks good on precision but costs the most because it misses the most fraud. Logistic Regression is a cheap, strong alternative.
- **BQ3 – What should investigators watch?** Behavior signals: login attempts, balance-to-amount ratio, and time since the last transaction. They matter far more than transaction size or customer profile. Account-level behavior carries about two-thirds of the signal.
- **BQ4 – Can it run in real life?** Yes. Speed is not an issue. The cost of mistakes, not accuracy, should decide which model and which alert level to use.

---

## 7. Recommendations

**Check the label first**
- Take a sample of the transactions Isolation Forest flagged and ask the fraud team to confirm or reject them. This turns the stand-in label into partial real ground truth.

**Choose the model and alert level**
- Set the alert level from the bank's real cost of a missed fraud and of a false alarm, not from a default 0.5 cutoff or from accuracy.
- Logistic Regression and DNN cost the least here. Be careful with Random Forest. Start with a pilot that only watches (no blocking), and move to deep learning only if its gain holds up on real labels.
- Try tuning and other ways to handle the small fraud class, such as SMOTE.

**Use two layers of checks**
- Keep a running risk score for each account (login attempts, balances, and long quiet periods change it over time), plus a fast check on each transaction at approval time.
- Use the simple two-part rule as a backup and as a sanity check on the model.

**Get ready for real use**
- Save and version the trained models, and run the chosen one behind a simple scoring service during approval.
- Send flagged transactions to an investigation list instead of blocking them right away.
- Monitor for changes in fraud patterns and retrain on a schedule.
- Leave out the age and job columns in production, or watch them closely, to avoid unfair treatment of customer groups.

---

## 8. Conclusion

Even without fraud examples, we can build a useful fraud detection system. Grouping on every column fails because it finds customer age and job, not fraud. Using only behavior columns and checking K-Means against Isolation Forest (87% overlap, moderate agreement) gives a label based on real behavior. It matches known fraud patterns such as account takeover and quiet accounts that suddenly become active.

With that label, **DNN performs best overall (F1 = 0.850, ROC-AUC = 0.988) and has the lowest estimated cost**, with ANN close behind. Logistic Regression is a strong, almost free alternative. Random Forest was the most expensive once missed fraud is counted, which shows why accuracy or precision alone should not pick a fraud model.

The strongest warning signs come from account behavior, not from transaction size or customer profile. This points to a running account risk score plus a fast check on each transaction. Speed is not an issue. Before going live, the label must be checked against real investigated cases, and the alert level must be set from real costs.

---

## 9. Limitations

- The fraud label is a **stand-in made by an unsupervised method**, not confirmed by a human investigator. Every number shows how well the models learned that stand-in, not necessarily real-world fraud.
- The dataset is small and probably synthetic (2,512 transactions). The patterns may not hold for a real bank's volume and fraud mix.
- There is no plan yet for changing fraud patterns or monitoring. A fixed model will get worse over time without retraining.
- The cost figures ($500 per missed fraud, $15 per false alarm) are examples. They must be replaced with the bank's real numbers before setting an alert level.

---

## 10. Creator

**Defrizal Yahdiyan Risyad** – defrijay@gmail.com
