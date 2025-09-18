# Statistical Theory - Final Project:  
**Gym Members Exercise**

This project applies **statistical analysis**, **unsupervised learning**, and **supervised learning** to gym members’ exercise tracking data. It explores **gender-related patterns** in behavioral and physiological features, evaluates predictive models, and validates clustering results with statistical tests. All outputs are exported as **publication-ready PDF figures** into the `figures_pdf/` directory.

---

## Project Structure
```
gym_members_project/
├── anomaly_detection.py                 # IQR-based outlier detection and cleaning
├── analysis.py                           # Cluster–gender statistical tests and p-value table export
├── clustering.py                         # Elbow + heatmaps for KMeans/DBSCAN/Agglomerative
├── cluster_analysis.py                   # Gender-dominated clusters, MWU/Kruskal, feature-direction summary
├── data_loader.py                        # Load CSV, encode labels, numeric encoding, scaling
├── dimensionality_reduction.py           # PCA→KMeans + UMAP visualizations
├── gender_feature_stats.py               # Mann–Whitney tests by gender
├── gender_model_selection.py             # Benchmark 6 classifiers; ROC comparison (fig6D)
├── gender_prediction_model_analysis.py   # RF: Behavioral vs Combined → confusions, importances, ROC CIs
├── normality_tests.py                    # Shapiro–Wilk + Levene (standalone)
├── PCA_loadings.py                       # PCA component loadings plots (fig5A/B)
├── visualizations.py                     # PDF saving utility (creates figures_pdf/)
├── pca and umap selection.py             # Exploratory DR comparison (not used by main.py)
├── main.py                               # Orchestrates the complete pipeline
├── figures_pdf/                          # Auto-created; all PDF outputs
└── gym_members_exercise_tracking.csv     # Input dataset (user-provided; not committed)
```



---

## Methods Used

### Preprocessing & EDA
- Dropped missing values and inconsistent rows.  
- Standardized continuous variables (**Age**, **BMI**, **Avg_Glucose**).  
- One-hot encoded categorical variables (**Gender**, **SES**, **Smoking_Status**).  
- **EDA outputs**:  
  - **fig1.pdf**: feature distributions.  
  - **fig2.pdf**: correlation heatmap & target associations.  

### Data Splitting & Scaling
- **Stratified split**: 70% train, 15% validation, 15% test.  
- **StandardScaler** fit only on train set, applied to val/test.  
- Validation used for **early stopping** (when supported).  

### Models
Evaluated **14 supervised models**, including:  
- Logistic Regression (L1, L2)  
- Decision Tree, Random Forest  
- Gradient Boosting, XGBoost, LightGBM, CatBoost  
- SVM (linear, RBF)  
- KNN, Naive Bayes  
- Ensemble methods (Voting, Bagging, Boosting)  

### Hyperparameter Tuning
- Used **GridSearchCV** and **Optuna** (5-fold time-aware CV).  
- Optimized depth, learning rate, regularization, number of estimators, and class weights.  

### Evaluation & Selection
- Metrics: **Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC**.  
- Figures:  
  - **fig3.pdf**: model comparison & ranking.  
  - **fig4.pdf**: confusion matrices, ROC & PR curves.  

### Statistical Validation
- Applied **significance tests** (paired t-test, McNemar’s test) to compare classifiers.  
- Highlighted differences in recall/precision for minority class (**stroke cases**).  

### Ensemble Methods
- Implemented **Soft Voting Classifier** combining best 3 models.  
- Ensemble improved overall AUC and recall compared to individual models.  
- **fig5.pdf**: ensemble performance.  

---

## Setup

### 1) Create a virtual environment (optional)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


### 2) **Install dependencies**
```bash
pip install -r requirements.txt
```

### 3) **Place your dataset**
Ensure `gym_members_exercise_tracking.csv` is located at the project root.



---

## How to Run

Run the full analysis pipeline:
```bash
python main.py
```

This will:
1) Load and preprocess the dataset.
2) Remove outliers via IQR (per gender).
3) Print assumption checks (Shapiro–Wilk, Levene) and Mann–Whitney results.
4) Train and evaluate supervised models (RF detailed; 6-model benchmark).
5) Evaluate multiple clustering algorithms and parameter grids.
6) Export all figures/tables to `figures_pdf/`.

---

## Outputs (PDFs in `figures_pdf/`)

- **fig1A** – KMeans loss heatmap vs. PCA dimensions & cluster counts  
- **fig1B** – KMeans elbow curve (PCA=2D)  
- **fig1C** – KMeans silhouette heatmap  
- **fig1D** – DBSCAN “coerced” silhouette heatmap  
- **fig1E** – Agglomerative silhouette heatmap  
- **fig3B** – PCA (2D) + KMeans scatter  
- **fig3C** – Gender distribution by cluster (counts and %)  
- **fig5A**, **fig5B** – PCA loadings (PC1, PC2)  
- **fig5C** – Feature-direction summary table  
- **fig6A** – RF confusion (Behavioral)  
- **fig6B** – RF feature importances (Combined)  
- **fig6C** – RF ROC with 95% bootstrap CIs (Behavioral vs Combined)  
- **fig6D** – ROC comparison (RF, LR, SVM, KNN, NB, MLP)  
- **Table_1** – Cluster–gender p-values (ANOVA, Kruskal) summary

---

## Contact

For questions or feedback: **hoshenmn@gmail.com**

---

## requirements.txt
```
numpy
pandas
matplotlib
seaborn
scikit-learn
umap-learn
scipy
```

