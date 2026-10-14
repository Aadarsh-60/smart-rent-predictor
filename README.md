# 🏠 Smart Rent Predictor & Advisory

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning web application designed to predict residential house rents across major Indian metropolitan cities (Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata) and provide personal financial affordability advisory with localized neighborhood recommendations.

---

## 🌟 Key Highlights

* 🎯 **Ensemble Rent Prediction**: Random Forest Regressor yielding $R^2 \approx 0.70$ and $\text{MAE} \approx ₹6,687$.
* 📊 **Confidence Intervals**: 80% prediction interval derived from individual decision tree variance to communicate market uncertainty.
* 💡 **Financial Health Advisory**: Implements the personal finance 30%–40% affordability rule with interactive Plotly gauge visualization.
* 📈 **Interactive Market Analytics**: City-wise comparisons, BHK distributions, and price-per-square-foot analysis powered by Plotly.
* 🏆 **Multi-Model Benchmarking**: Evaluates and compares Linear Regression, Ridge Regularization, Decision Trees, and Random Forest.
* 🧭 **Neighborhood Locality Explorer**: Filter and search through real-world listings across metro localities by budget and space.
* 🎓 **Interview Ready**: Complete [INTERVIEW_PREP.md](INTERVIEW_PREP.md) guide with architecture, design rationale, and 10+ technical Q&A pairs.

---

## 📊 Model Benchmarking & Results

Candidate algorithms evaluated on an 80/20 train-test split (Pan-India Dataset across 22+ States):

| Algorithm | MAE (₹ INR) | RMSE (₹ INR) | $R^2$ Score | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor (Champion)** | **₹2,875.36** | **₹4,236.84** | **0.9421** | **0.494s** |
| Decision Tree Regressor | ₹4,207.33 | ₹6,414.22 | 0.8673 | 0.038s |
| Ridge Regression (L2) | ₹4,629.67 | ₹6,520.44 | 0.8629 | 0.047s |
| Linear Regression (OLS) | ₹4,631.83 | ₹6,531.99 | 0.8624 | 0.241s |

> **Key Finding**: Incorporating hierarchical geographical features (`State`, `City_Tier`, and `Locality_Zone`) enabled Random Forest to model sub-market pricing variance with exceptional fidelity, boosting $R^2$ from ~0.70 to **0.9421** and cutting prediction error (MAE) down to **₹2,875**.

---

## 🔍 Top Predictive Features

1. **Property Size (Sq. Ft.)** — 34.4%
2. **City (Mumbai Premium)** — 31.5%
3. **Bathroom Count** — 12.7% *(Engineered)*
4. **Room Density (`Size_per_BHK`)** — 7.9% *(Engineered)*
5. **City (Delhi)** — 2.3%
6. **Furnishing Status (Unfurnished)** — 2.1%

---

## 🗂️ Project Structure

```
smart-rent-predictor/
├── app.py                      # Root Streamlit entry point
├── dashboard/
│   └── app.py                  # Full multi-tab Streamlit & Plotly UI
├── data/
│   └── raw/
│       ├── house_rent.csv      # Raw dataset (~4,746 property listings)
│       └── cleaned_data.csv    # Cleaned dataset (outliers removed)
├── models/
│   ├── random_forest_model.pkl # Trained Random Forest Regressor
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── training_columns.pkl    # One-hot encoded feature column schema
│   ├── model_comparison.json   # Benchmark metrics for all candidate models
│   ├── feature_importance.json # Gini impurity feature importances
│   └── test_predictions.json   # Actual vs predicted validation sample
├── notebooks/
│   └── eda_modeling.ipynb      # Jupyter notebook for exploratory analysis
├── plots/                      # Visualizations (EDA, distributions)
├── src/
│   ├── data_cleaning.py        # Outlier filtering logic
│   ├── utils.py                # Preprocessing, feature engineering & scaling
│   └── train_model.py          # Model training & benchmarking pipeline
├── INTERVIEW_PREP.md           # 🎓 Technical Interview Master Guide
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview & documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Aadarsh-60/smart-rent-predictor.git
cd smart-rent-predictor
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate       # On Linux/macOS
# or: .\venv\Scripts\activate  # On Windows

pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. (Optional) Re-train & Benchmark Models
```bash
python src/train_model.py
```

---

## 👨‍💻 Author
**Aadarsh Vikram**  
* GitHub: [@Aadarsh-60](https://github.com/Aadarsh-60)
* Email: [vikramaadarsh2001@gmail.com](mailto:vikramaadarsh2001@gmail.com)

---

## 📄 License
This project is open-source and licensed under the [MIT License](LICENSE).