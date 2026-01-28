# Student Academic Performance Predictor

A machine learning project for analyzing and predicting student academic performance using data science techniques.

## 📋 Project Overview

This project performs:
- **Exploratory Data Analysis (EDA)** - Understanding data patterns and distributions
- **Clustering Analysis** - Grouping students into performance segments
- **Predictive Modeling** - Forecasting student performance outcomes

The project includes interactive visualization through a Streamlit web application.

## 🗂️ Project Structure

```
├── app/
│   └── streamlit_app.py          # Main web application
├── data/
│   ├── merged_dataset.csv         # Combined dataset
│   ├── processed_data.csv         # Cleaned and preprocessed data
│   └── clustered_data.csv         # Data with cluster assignments
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_clustering.ipynb       # Clustering Analysis
│   └── 03_prediction.ipynb       # Predictive Modeling
├── src/                           # Source code modules
└── requirements.txt               # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone or download the project
2. Navigate to the project directory:
   ```bash
   cd students_academic_project
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Running the Application

### Launch the Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
The app will open in your browser at `http://localhost:8501`

### Run Jupyter Notebooks
```bash
jupyter notebook
```
Then navigate to the `notebooks/` folder and open the desired notebook.

## 📦 Dependencies

Key libraries used:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **streamlit** - Web application framework
- **matplotlib & seaborn** - Data visualization
- **plotly** - Interactive charts
- **joblib** - Model serialization

## 🔍 Project Workflow

1. **Data Preparation** - Load and merge datasets
2. **EDA** - Analyze distributions, correlations, and patterns
3. **Clustering** - Segment students into groups
4. **Modeling** - Train and evaluate prediction models
5. **Visualization** - Display insights in web app

## 📁 Data Files

- `merged_dataset.csv` - Raw combined student data
- `processed_data.csv` - Cleaned and preprocessed dataset
- `clustered_data.csv` - Data with cluster labels

## 🛠️ Technologies

- Python 3.x
- Streamlit
- scikit-learn
- Jupyter Notebooks
- Pandas & NumPy

## 📝 Notes

- Ensure all data files are in the `data/` directory
- Models should be saved in pickle/joblib format
- Update `requirements.txt` if adding new dependencies

## 📧 Contact

For questions or issues, please refer to the project documentation or contact the development team.
