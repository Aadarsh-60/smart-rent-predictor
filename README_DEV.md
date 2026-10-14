# 🏠 Smart Rent Advisor - Developer Guide

Welcome to the **Smart Rent Advisor** developer workspace! This document helps you set up, run, and maintain the project like a pro. It's modular, reliable, and restart-safe. 💪

---

## 🗂️ Project Structure
```
smart-rent-advisor/
├── app.py # 🔹 Streamlit dashboard
├── data/
│ ├── raw/ # 📂 Cleaned & raw datasets
│ └── processed/ # 📂 (Optional) Transformed sets
├── models/ # 📁 Trained models and scalers
│ ├── random_forest_model.pkl
│ └── scaler.pkl
├── notebooks/ # 📒 Jupyter Notebooks (EDA, modeling)
├── plots/ # 📊 Visualizations
├── src/
│ ├── data_cleaning.py # 🔧 Cleaning logic
│ ├── utils.py # 🧠 Preprocessing functions
│ └── train_model.py # 🏋️ Model training script
└── README.md # 🧾 Project overview
```


---

## 🚀 Quick Start

### ✅ 1. Create & Activate Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
.\venv\Scripts\activate      # Windows
# or
source venv/bin/activate     # macOS/Linux
```

### ✅ 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have one, generate with:

```bash
pip freeze > requirements.txt
```

### ✅ 3. Clean Dataset (Optional if already done)

Use data_cleaning.py from src/ or notebook logic. Save cleaned data to:

```bash
data/raw/cleaned_data.csv
```

### ✅ 4. Train the Model

```bash
python src/train_model.py
```
Outputs:

Model: models/random_forest_model.pkl

Scaler: models/scaler.pkl

### ✅ 5. Launch the Streamlit App

```bash
streamlit run app.py
```

🧠 Common Recovery Steps (After Restart)<br>
If your kernel/runtime disconnects:

# Reload cleaned data
```bash
import pandas as pd
df = pd.read_csv("data/raw/cleaned_data.csv")
```

# Reload model and scaler
```bash
import joblib
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
```

# Ready to predict!
```bash
pred = model.predict(...)
```

🛠️ Developer Tips<br>

🔁 Don't re-train model unless you change data.

📦 Use joblib to load/save large objects efficiently.

✅ Always test new features via notebooks before migrating to src/

---

📬 Contribute<br>
Want to help improve the Smart Rent Advisor?<br>
Feel free to create issues or send pull requests!

---

👨‍💻 Author
Aadarsh Vikram

---