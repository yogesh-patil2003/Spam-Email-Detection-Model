# 📧 Spam Email Detection using Machine Learning

This project classifies email messages as either Spam or Not Spam using a machine learning pipeline built in Python. It uses text preprocessing, TF-IDF feature extraction, and a Logistic Regression classifier to predict whether an incoming message is spam.

---

## 📌 Project Overview

Spam emails are a common problem and can lead to phishing attempts, scams, unwanted promotions, and security risks. This project aims to automate the detection of spam messages using a supervised machine learning approach.

The project reads a labeled dataset of email messages, cleans the data, transforms the message text into numerical features, trains a Logistic Regression model, and evaluates its performance on a test set.

---

## 🚀 Features

- Classifies messages as Spam (0) or Ham/Not Spam (1)
- Uses TF-IDF vectorization for text feature extraction
- Trains a Logistic Regression model for binary classification
- Evaluates the model using accuracy metrics
- Provides a simple prediction workflow for new email texts
- Works directly from a CSV dataset and a Jupyter notebook

---

## 📂 Dataset

- Dataset file: `mail_data.csv`
- Total rows: 5,572
- Columns:
  - `Category`: label for the message (`spam` or `ham`)
  - `Message`: the actual email/text content

The dataset is loaded into a Pandas DataFrame, null values are handled, and the labels are converted to numeric values:

- `spam` → `0`
- `ham` → `1`

---

## 🛠️ Tech Stack

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn

### Libraries Used

- `pandas`
- `numpy`
- `sklearn.model_selection.train_test_split`
- `sklearn.feature_extraction.text.TfidfVectorizer`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.metrics.accuracy_score`

---

## 🧠 Model Workflow

The project follows this pipeline:

1. Load the dataset from `mail_data.csv`
2. Replace missing values with empty strings
3. Map labels (`spam` / `ham`) to numeric values (`0` / `1`)
4. Split the dataset into training and testing sets
5. Convert text messages into TF-IDF features
6. Train a Logistic Regression model
7. Evaluate model accuracy on training and testing data
8. Predict the class for a new sample message

---

## 📊 Model Performance

The notebook reports the following training and testing accuracies:

- Training Accuracy: `0.9676912721561588` (~96.77%)
- Test Accuracy: `0.9668161434977578` (~96.68%)

These values show that the model performs very well in classifying spam vs. non-spam messages.

---

## 🧪 Example Prediction

```python
input_mail = ["I've been searching for the right words to thank you for this breather."]

input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
```

Sample output:

```python
[1]
Ham mail
```

---

## 📁 Project Structure

```text
Spam-Email-Detection-Model/
├── README.md
├── Spam E-mail Detection.ipynb
├── mail_data.csv
└── .gitignore (if present)
```

---

## ▶️ How to Run

1. Open the notebook `Spam E-mail Detection.ipynb` in Jupyter Notebook or VS Code.
2. Ensure the required Python libraries are installed.
3. Run all cells in order.
4. The notebook will train the model and allow you to test custom messages.

Install dependencies using:

```bash
pip install pandas numpy scikit-learn jupyter
```

---

## 💡 Future Improvements

- Try alternative models such as Naive Bayes, Random Forest, or SVM
- Add a real-time spam detection interface
- Deploy as a web app or API
- Improve preprocessing for multilingual or noisy messages
- Add model saving and prediction utilities for production use

---

## ✅ Summary

This project is a practical machine learning solution for spam detection using a classic and effective text classification pipeline. It demonstrates end-to-end text classification using TF-IDF and Logistic Regression, with strong accuracy on the provided dataset.

