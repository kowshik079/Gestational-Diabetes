# Gestational Diabetes Prediction

A machine learning project for predicting gestational diabetes using the Pima Indians Diabetes Dataset. This project implements exploratory data analysis (EDA) and K-Nearest Neighbors (KNN) classification to predict diabetes outcomes based on various health indicators.

## :clipboard: Overview

This project analyzes health data to predict the likelihood of gestational diabetes in patients. The analysis includes comprehensive data preprocessing, visualization, and machine learning model implementation with performance evaluation.

## :card_index_dividers: Dataset

The dataset contains health information from Pima Indian women with the following features:

- *Pregnancies*: Number of pregnancies
- *Glucose*: Plasma glucose concentration (mg/dL)
- *BloodPressure*: Diastolic blood pressure (mmHg)
- *SkinThickness*: Triceps skin fold thickness (mm)
- *Insulin*: 2-Hour serum insulin (μU/mL)
- *BMI*: Body Mass Index (kg/m²)
- *DiabetesPedigreeFunction*: Diabetes pedigree function score
- *Age*: Age in years
- *Outcome*: Target variable (0: No diabetes, 1: Diabetes)

*Dataset Size*: 768 records

## :hammer_and_wrench: Technologies Used

- *Python 3.x*
- *Libraries*:
  - pandas - Data manipulation and analysis
  - numpy - Numerical computations
  - matplotlib - Data visualization
  - seaborn - Statistical data visualization
  - scikit-learn - Machine learning algorithms and metrics

## :bar_chart: Project Structure


Gestational-Diabetes/
├── dataset.csv              # Raw dataset
├── Untitled.ipynb          # Main analysis notebook
└── README.md               # Project documentation


## :mag: Analysis Pipeline

### 1. Data Exploration & Preprocessing

- Loading and inspecting dataset structure
- Handling missing values (replaced 0s with NaN for medical impossibilities)
- Statistical summary and data distribution analysis
- Missing data visualization

### 2. Exploratory Data Analysis (EDA)

- Correlation analysis with heatmap visualization
- Distribution plots for all features
- Target variable analysis (diabetes outcomes)
- Pregnancy count analysis by outcome
- BMI analysis for normal weight patients

### 3. Data Preprocessing

- Feature scaling using StandardScaler
- Train-test split (70-30 ratio)

### 4. Machine Learning Implementation

- *Algorithm*: K-Nearest Neighbors (KNN) Classification
- *Model Optimization*:
  - K-value optimization (tested K=1 to K=39)
  - Cross-validation for best performance
- *Final Model*: KNN with K=20 (optimized value)

### 5. Model Evaluation

- Confusion Matrix analysis
- Classification Report (Precision, Recall, F1-Score)
- Accuracy Score calculation
- Visualization of results with heatmaps

## :chart_with_upwards_trend: Key Results

- *Model Performance*: Achieved optimal results with K=20
- *Evaluation Metrics*:
  - Detailed classification report with precision, recall, and F1-scores
  - Confusion matrix visualization
  - Model accuracy assessment

## :rocket: Getting Started

### Prerequisites

bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter


### Running the Analysis

1. Clone or download this repository
2. Ensure dataset.csv is in the project directory
3. Open Untitled.ipynb in Jupyter Notebook or VS Code
4. Run all cells sequentially to reproduce the analysis

### Usage

python
# Load the main notebook
jupyter notebook Untitled.ipynb

# Or use VS Code with Jupyter extension
code Untitled.ipynb


## :clipboard: Features Implemented

- :white_check_mark: Comprehensive data cleaning and preprocessing
- :white_check_mark: Missing value analysis and handling
- :white_check_mark: Statistical analysis and visualization
- :white_check_mark: Correlation analysis
- :white_check_mark: Feature distribution analysis
- :white_check_mark: KNN model implementation
- :white_check_mark: Hyperparameter tuning (K-value optimization)
- :white_check_mark: Model performance evaluation
- :white_check_mark: Confusion matrix visualization
- :white_check_mark: Classification metrics reporting

## :dart: Future Improvements

- [ ] Implement additional ML algorithms (Random Forest, SVM, Logistic Regression)
- [ ] Advanced feature engineering and selection
- [ ] Cross-validation for more robust evaluation
- [ ] ROC curve analysis
- [ ] Feature importance analysis
- [ ] Model deployment pipeline
- [ ] Interactive dashboard for predictions

## :memo: Notes

- The dataset contains some zero values that are medically impossible (e.g., 0 glucose, 0 BMI), which have been handled as missing values
- The analysis focuses on the Pima Indian population, so results may not generalize to other populations
- Model performance should be validated on additional datasets for broader applicability

## :handshake: Contributing

Feel free to fork this project and submit pull requests for any improvements or additional features.

## :page_facing_up: License

This project is open source and available under the [MIT License](LICENSE).

## :telephone_receiver: Contact

For questions or suggestions regarding this project, please open an issue in the repository.

---

*Note*: This project is for educational and research purposes. Always consult with healthcare professionals for medical decisions.
