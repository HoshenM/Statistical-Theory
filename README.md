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
Preprocessing

Label encoding of categorical variables (including Gender).
Standardization of numeric features.
IQR-based outlier detection/removal (per-gender bounds by default).
Statistical Tests

Shapiro–Wilk (normality per gender)
Levene’s test (variance equality across genders)
Mann–Whitney U (gender differences, nonparametric)
Kruskal–Wallis H (differences across clusters, nonparametric)
ANOVA (where assumptions permit)
NEW: Spearman rank correlation matrix for numeric features, visualized as a diverging heatmap (fig2)
Supervised Learning

Random Forest (detailed analysis: Behavioral vs Combined features)
Logistic Regression, SVM, KNN, Naive Bayes, MLP
Unsupervised Learning

Dimensionality reduction: PCA, UMAP
Clustering: KMeans, DBSCAN, Agglomerative
Evaluation & Visualization

ROC, AUC, and bootstrap CIs (95%) for RF comparison
Silhouette-based heatmaps across PCA dimensions and cluster counts
Gender distribution by cluster table
PCA loadings and feature-direction summaries
```
---
##  Setup
### 1) Create a virtual environment (recommended)
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate


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

