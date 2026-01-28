# Student Academic Performance Predictor 🎓

Predict and understand student academic performance using machine learning and data‑driven insights. This project not only predicts grades but also provides personalized recommendations based on psychology research.

---

## 🛠️ Project Overview

This project is a complete machine learning pipeline that analyzes student data to:

- Identify patterns in academic performance  
- Segment students into performance clusters  
- Predict final grades  
- Provide personalized, psychology‑backed recommendations  

It includes an **interactive web application** where students or educators can input data and receive actionable insights.

---

## ⚡ Key Features

- **Comprehensive EDA:** Clean data, detect outliers, analyze distributions  
- **Student Segmentation:** KMeans clustering into 3 distinct performance groups  
- **Grade Prediction:** RandomForest model predicting A/B/C/D final grades  
- **Psychology‑Based Tips:** Personalized recommendations for study habits, sleep, and stress management  
- **Interactive Web App:** User‑friendly interface with real‑time predictions  
- **Feature Analysis:** Visualize the most influential factors driving student success

---

## 🛠️ How I Built This Project

### 1. Data Exploration & Cleaning (EDA)
- Loaded raw CSV student datasets  
- Handled missing values, duplicates, and outliers  
- Analyzed distributions with histograms and boxplots  
- Cleaned data saved as `processed_data.csv`

### 2. Student Clustering Analysis
- Normalized features using `StandardScaler`  
- Applied KMeans clustering (k=3)  
- Segmented students into performance groups  
- Clustered data saved as `clustered_data.csv`

### 3. Predictive Modeling
- Prepared features and performed scaling  
- Trained a `RandomForestClassifier` to predict final grades  
- Implemented feature engineering and stratified train‑test split  
- Models serialized using `joblib` and `pickle`

### 4. Web Application Development
- Built using **Streamlit** with a professional UI  
- Predicts grades and provides personalized recommendations  
- Interactive features like sliders, dropdowns, and dynamic result cards  
- Feature importance visualizations included

---

## 📊 Data Insights

- **Study Hours:** Avg ~20 hrs/week (range: 5–44 hrs)  
- **Attendance:** Generally high, avg ~80% (min ~60%)  
- **Student Segments:** 3 distinct clusters based on engagement  
- **Key Predictors:** Study habits, attendance, and stress levels

---

## 🚀 Technologies Used

- Python 3.x  
- Pandas & NumPy – Data manipulation  
- Scikit‑learn – Machine learning  
- Matplotlib & Seaborn – Data visualization  
- Streamlit – Web application  
- Joblib & Pickle – Model serialization  
- Jupyter Notebooks – Interactive analysis

---

## 📂 Project Structure

```
students_academic_project/
│
├── app/
│   ├── streamlit_app.py          # Main web application
│   ├── trained_classifier.pkl    # Saved RandomForest model
│   ├── trained_regressor.pkl     # Saved regression model (if used)
│   └── scaler.pkl                # Feature scaler
│
├── data/
│   ├── merged_dataset.csv        # Original dataset
│   ├── processed_data.csv        # Cleaned data
│   └── clustered_data.csv        # Clustered data
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_clustering.ipynb      # KMeans clustering analysis
│   └── 03_prediction.ipynb      # Model training & evaluation
│
├── src/                         # Additional source code
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

---

## 📦 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/smiley645/student_performance_predictor.git
cd student_performance_predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage

### Run the Web App
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

### Explore the Analysis
Open notebooks in the `notebooks/` folder:
- `01_eda.ipynb` – Data cleaning & exploration  
- `02_clustering.ipynb` – Student segmentation  
- `03_prediction.ipynb` – Model training & evaluation

---

## 🎓 How the App Works

1. Input your details: Study hours, attendance, age, motivation, etc.  
2. Get predictions: Expected final grade (A/B/C/D) and score range  
3. Receive recommendations: Psychology‑backed tips for improvement  
4. See key factors: Which aspects most influence your success

---

## 📈 Model Performance

- **Algorithm:** RandomForestClassifier with 200 estimators  
- **Features:** 14 student attributes including demographics and habits  
- **Target:** Final grade prediction  
- **Scaler:** StandardScaler for numerical normalization

---

## 🔍 What I Learned

- Building a complete ML pipeline from data cleaning to deployment  
- Importance of preprocessing and feature engineering  
- Student segmentation with clustering techniques  
- Developing interactive web apps with actionable insights  
- Model serialization for production‑ready applications
