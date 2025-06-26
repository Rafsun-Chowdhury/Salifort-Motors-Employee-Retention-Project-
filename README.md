
# 🏢 Salifort Motors: Predicting Employee Attrition with Machine Learning

This project analyzes employee attrition at Salifort Motors to identify patterns and predict which employees are at risk of leaving. Using classification models and real-world-style HR data, it helps HR professionals make proactive, data-informed decisions to reduce turnover.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rafsun-Chowdhury/Salifort-Motors-Employee-Retention-Project-/blob/main/Salifort_Motors_Attrition_Model.ipynb)

---

## 📊 Tools & Technologies

- **Python, Pandas, NumPy** — data manipulation
- **scikit-learn** — model training and evaluation
- **Matplotlib & Seaborn** — visual analysis
- **Random Forest Classifier** — primary prediction model

---

## 🎯 Project Objectives

- Analyze key drivers behind employee attrition
- Predict which employees are at high risk of leaving
- Provide HR with insights to support intervention and retention strategies
- Build a simplified, HR-friendly prediction tool using only 3 input variables

---

## 📈 Model Performance

| Metric    | Class: Stayed | Class: Left | Overall Accuracy |
|-----------|----------------|--------------|------------------|
| Precision | 0.99           | 0.99         | **99%**          |
| Recall    | 1.00           | 0.92         |                  |
| F1-score  | 0.99           | 0.96         |                  |

> ✅ The model performs exceptionally well, especially in identifying employees at risk of leaving, which is the most critical use case for HR.

---

## 📊 Visual Insights

- 📌 **Attrition by Salary Level** — reveals high churn among low-salary groups
- 📌 **Satisfaction vs. Attrition** — strong link between dissatisfaction and attrition
- 📌 **Feature Importance** — shows top predictors of employee turnover

---

## 🤖 HR-Friendly Prediction Tool

Using just:
- **Satisfaction Level**
- **Monthly Working Hours**
- **Salary Level (Low / Medium / High)**

…HR can predict the probability of an employee leaving with a single function call.

```python
simplified_attrition_risk(model, satisfaction_level=0.4, monthly_hours=170, salary_level=0)
```

---

## ▶️ How to Run

### Option 1: Run in Google Colab  
Click the badge at the top of this README.

### Option 2: Run Locally
```bash
git clone https://github.com/Rafsun-Chowdhury/Salifort-Motors-Employee-Retention-Project-.git
cd Salifort-Motors-Employee-Retention-Project-
pip install -r requirements.txt
jupyter notebook Salifort_Motors_Attrition_Model.ipynb
```

---

## 👤 Author

**Rafsun Chowdhury**  
📧 Email: rafsunrf@gmail.com  
🔗 [GitHub](https://github.com/Rafsun-Chowdhury)  
🌐 [Portfolio](https://rafsun-chowdhury.github.io/portfolio/)  
💼 [LinkedIn](https://www.linkedin.com/in/rafsun-chowdhury/)
