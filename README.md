# 🍴 Online Food Feedback

This project analyzes online food delivery feedback and predicts customer satisfaction using **Random Forest**.  
The goal is to understand the key factors influencing customer experience and provide insights to improve food delivery services.  

---

## 📌 Features
- Preprocessing of customer feedback data (cleaning, encoding, feature selection).
- Model training using **Random Forest Classifier**.
- Evaluation with accuracy, precision, recall, and F1-score.
- Data versioning handled with **DVC** for reproducibility.
- CI/CD pipeline integration with **GitHub Actions**.

---

## 🛠 Tech Stack
- **Language:** Python 3.11
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Model:** RandomForestClassifier
- **Tools:** DVC, pytest, GitHub Actions

---

## 📂 Project Structure
ONLINE-FOOD-FEEDBACK/
│-- data/ # Raw and processed datasets
│-- models/ # Trained Random Forest model
│-- notebooks/ # Jupyter notebooks for EDA & training
│-- src/ # Source code (preprocessing, training, evaluation)
│-- dvc.yaml # DVC pipeline definition
│-- requirements.txt # Project dependencies
│-- _test.py # Unit tests
│-- README.md # Project documentation