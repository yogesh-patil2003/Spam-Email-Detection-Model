# 📧 Spam Email Detection using Machine Learning

This project aims to classify emails as **Spam** or **Not Spam** using a machine learning approach. It applies a **Logistic Regression** classifier on text-based features extracted from email content to provide accurate predictions.

---

## 📌 Project Overview

Spam emails are a widespread issue, posing risks like phishing attacks, malware, and data theft. This project leverages machine learning techniques to automate the classification of email messages, helping to filter unwanted content efficiently.

The system uses a cleaned and labeled dataset of email messages and builds a supervised learning model for binary classification.

---

## 🚀 Features

* Classifies email messages as **Spam (0)** or **Not Spam (1)**
* Utilizes **TF-IDF Vectorization** for feature extraction from text
* Employs **Logistic Regression** for its simplicity and effectiveness
* Easy-to-interpret output for real-world applicability
* Can serve as the foundation for email filtering tools

---

## 📂 Dataset

* **Source**: Custom Email Spam Dataset (CSV format)
* **Instances**: \~5,000 (typical)
* **Features**:

  * Text message content
  * Label: `"spam"` or `"ham"` converted to `0` and `1`

---

## 🛠️ Tech Stack

* **Python 3.x**
* **Jupyter Notebook**

### Libraries Used:

* NumPy
* Pandas
* Scikit-learn (TF-IDF, Logistic Regression, train-test split, metrics)

---

## 🧠 Model Training & Evaluation

* Text data vectorized using `TfidfVectorizer`
* Logistic Regression trained on **80%** of the dataset
* Evaluated on **20%** test split

### Accuracy:

* **Training Accuracy**: \~97.2%
* **Testing Accuracy**: \~95.8% (depending on dataset size and quality)

---

## 🧪 Sample Prediction

```python
input_message = "Congratulations! You've won a free iPhone. Click here to claim now."
# Output: SPAM (0)

input_message = "Please find the minutes of today’s meeting attached."
# Output: NOT SPAM (1)
```

---

## 💡 Future Improvements

* Implement advanced models like **Naive Bayes**, **Random Forest**, or **BERT**
* Add support for real-time spam filtering
* Deploy as a **web or desktop app** for general use
* Enable **multilingual** spam detection


