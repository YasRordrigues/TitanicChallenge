# Titanic Challenger
The Titanic Challenger refers to the famous Titanic shipwreck, where most of the ~1,500 people lost their lives, and it still raises the question: **is it possible to predict whether a passenger would survive or not?**  
To answer this question, this repository includes:
- A Titanic dataset from Kaggle;
- EDA and ETL on the data;
- A pipeline of models, with a best model reaching **80.7% accuracy** at predicting whether a passenger would survive.

## Data
- **Kaggle Titanic** dataset (single CSV).  
- Target: `sobreviveu`.  
- Columns used/derived: `idade`, `genero`, `classe_bilhete`, `local_embarque`, `preco_bilhete`, `cabine_numero` → `cabine_setor`, `irmaos_conjuges_abordo`, `pais_filhos_abordo`, `tamanho_familia`, `estava_sozinho`, `mulher_ou_crianca`, `log_preco_bilhete`.

## Approach
1. **EDA**: distribution of age, class, gender, and survival rate; correlations; analysis of **cabin sectors** and their relationship with survival.  
2. **Cleaning & ETL**  
   - `cabine_setor`: extract the **deck letter** from `cabine_numero`; missing values become `Unknown`.  
   - `preco_bilhete`: zeros treated as missing; **median imputation by (classe_bilhete × local_embarque)**; `log_preco_bilhete` to reduce skew.  
   - `idade`: **median imputation** by (classe_bilhete × genero).  
   - Family: `tamanho_familia = irmãos/cônjuges + pais/filhos + 1`; `estava_sozinho` (binary).  
   - Historical rule: `mulher_ou_crianca = (genero == feminino) ∪ (idade < 12)`.
3. **Preprocessing (ColumnTransformer)**  
   - Numerical: `SimpleImputer(median)` + (when applicable) `StandardScaler`.  
   - Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder` (fallback to `infrequent_if_exist` when available).  
4. **Modeling (scikit-learn pipelines)**  
   - Models tested: **LogisticRegression**, **KNN**, **DecisionTree**, **RandomForest**, **GradientBoosting**, **HistGradientBoosting**, **AdaBoost**.  
   - **Validation**: `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`.  
   - **Selection metric**: **AUC-ROC** (robust to imbalance).  
   - **Class imbalance**: use of `class_weight` (e.g., `balanced`, `balanced_subsample`) instead of oversampling/SMOTE to avoid leakage in CV.  
   - **Hyperparameter search** via `GridSearchCV` (lean, impact-focused grids).
5. **Explainability**  
   - Extraction of **post-OHE feature names**.  
   - **Feature importance** per model (trees: `feature_importances_`; linear: `|coef|`; fallback: **permutation importance**).  
   - **Aggregated ranking** across models to highlight consistent signals (e.g., `mulher_ou_crianca`, `classe_bilhete`, `cabine_setor`, `idade`, `log_preco_bilhete`, `tamanho_familia`).

## What we included (and why)
- **Boosted Trees** (HGB, GB) and **RF**: capture non-linearities and interactions in tabular data; stable performance.  
- **LogReg/KNN/DT**: useful baselines to compare bias/variance and interpretability.  
- **`class_weight`** instead of oversampling: simpler, less prone to leakage, and sufficient for the dataset’s imbalance.

## What we didn’t include (and why)
- **Deep Learning/Stacking**: high complexity on a small dataset.  
- **Full cabin number**: many missing values; using **sector/letter** alone proved more robust.

## Results
- Selection by **AUC-ROC in 3-fold CV** and final validation on **25% stratified holdout**.  
- Typical performance: **AUC-ROC around 0.86** and **accuracy around 80–82%** (may vary with split/seed).  
- **Confusion matrix** and **feature importances** help balance *performance* and *model understanding*.

## How to run
```bash
# 1) Clone and create environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip

# 2) Minimal dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

# 3) Data
# Adjust the CSV path in the notebook as needed.

# 4) Execute
# Open notebooks/Desafio_Ascan_Titanic.ipynb and run all cells
