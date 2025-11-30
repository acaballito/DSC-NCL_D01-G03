# 📘 RECOMMENDED NOTEBOOK STRUCTURE
## Professional Data Science Notebook Organization

---

## **PART 0: SETUP & INTRODUCTION** 
*(Keep this minimal and clean)*

```
## 📊 BMW Pricing - Data Preparation & EDA
### Project Information
- **Professor:** Alberto Vacas  
- **Project:** BMW Pricing  
- **Group:** 03  
- **Objective:** Prepare BMW used car dataset for price prediction modeling

---

### 0.1. Environment Setup
```python
# Imports
# Data loading
# Initial configs
```

### 0.2. Dataset Overview
```python
# df.head(), df.shape, df.info()
# Brief 2-3 line summary
```

---

## **PART 1: DATA CLEANING & PREPARATION**
*(Execution-focused: Code + brief markdown. NO detailed justifications here)*

### 1.1. Initial Data Quality Assessment
- Check nulls (summary table)
- Check duplicates
- Check data types
- **Action:** Brief 1-liner per issue found

### 1.2. Column Elimination
```python
# Drop marca, asientos_traseros_plegables
```
**Why (1 sentence):** High nulls (70%) or no variance

### 1.3. Data Type Corrections
```python
# Convert dates to datetime
```

### 1.4. Duplicate Removal
```python
# Remove duplicates
```

### 1.5. Feature Engineering
```python
# Create EDAD_VEHICULO
# Create fecha_registro_disponible
```

### 1.6. Missing Value Treatment
**Strategy by variable type:**

#### 1.6.1. Critical Variables (Eliminate Records)
- precio, modelo, km, potencia, fecha_venta → **Drop records with nulls**

#### 1.6.2. Categorical Variables
- tipo_coche, color → **Impute "desconocido"**
- tipo_gasolina → **Impute moda (diesel)**

#### 1.6.3. Boolean Variables
- bluetooth, alerta_lim_velocidad, aire_acondicionado, etc. → **Impute False**

#### 1.6.4. fecha_registro (Advanced Imputation)
```python
# Step 1: Create indicator variable
# Step 2: Calculate median km/año from non-null records
# Step 3: Estimate EDAD_VEHICULO = km / km_per_year
# Step 4: Back-calculate fecha_registro
```
**Brief justification (2-3 lines):** 50% nulls too high to delete. Use km/año relationship to preserve data.

**Result:** 0 nulls in final dataset ✅

### 1.7. Outlier Analysis
```python
# IQR method for precio
# Decision: Keep superior outliers (luxury cars valid)
# Keep other variable outliers (km, potencia)
```
**Brief justification (1-2 lines):** 311 outliers are genuine luxury BMW (M4, i8, X6 M), not errors.

### 1.8. Save Cleaned Dataset
```python
df_bmw_clean.to_pickle("dataset_bmw_clean.pkl")
```
**Checkpoint:** Clean dataset ready for EDA (4,827 records, 0 nulls)

---

## **PART 2: EXPLORATORY DATA ANALYSIS (EDA)**
*(Focus on insights, not decisions)*

### 2.1. Load Cleaned Data
```python
df_bmw_clean = pd.read_pickle("dataset_bmw_clean.pkl")
```

### 2.2. Univariate Analysis

#### 2.2.1. Numerical Variables
- **precio:** Mean 15,827€, median 14,200€, skewed right
- **km:** Mean 140,931 km, symmetric distribution
- **potencia:** Mean 129 CV, range 0-423 CV
- **EDAD_VEHICULO:** Mean 5.38 years

**Visualizations:** Histograms + boxplots for each

#### 2.2.2. Categorical Variables
- **modelo:** 76 categories, 320 most frequent (15.5%)
- **tipo_gasolina:** 95.7% diesel
- **color:** black/grey/white dominate
- **tipo_coche:** sedan/estate most common

**Visualizations:** Bar charts with frequencies

#### 2.2.3. Boolean Variables
- GPS: 93.2% have it (almost standard)
- Bluetooth: only 20.6%
- Camera: 11.7%

**Key Insight:** GPS ubiquitous, bluetooth rare

### 2.3. Correlation Analysis

**Correlation Matrix (Heatmap)**

**Key Findings:**
- ⚠️ **HIGH:** km ↔ EDAD_VEHICULO (r=0.730) → Multicollinearity risk
- ✅ **Moderate with price:**
  - potencia → precio (r=0.639) strongest positive
  - EDAD_VEHICULO → precio (r=-0.423) 
  - km → precio (r=-0.409)
- ✅ fecha_registro_disponible → precio (r=0.020) → Imputation unbiased ✅

**Implication:** May need to drop km or EDAD_VEHICULO for linear models

### 2.4. Bivariate Analysis (Variable vs Target)

#### 2.4.1. Numerical vs Precio
- **potencia:** Strong positive (r=0.639) ⭐ Best predictor
- **km:** Moderate negative (r=-0.409)
- **EDAD_VEHICULO:** Moderate negative (r=-0.423)

**Visualizations:** Scatterplots with regression lines

#### 2.4.2. Categorical vs Precio
- **modelo:** i8 (95,200€) vs 116 (4,500€) → 90,700€ difference! ⭐⭐⭐
- **tipo_gasolina:** hybrid_petrol (+135% vs diesel)
- **color:** Minor impact
- **tipo_coche:** Moderate impact

**Visualizations:** Boxplots by category

#### 2.4.3. Boolean vs Precio
- **alerta_lim_velocidad:** +7,032€ (55.9% increase!) ⭐
- **camara_trasera:** +5,815€ (39.7%)
- **gps:** -247€ (no impact, too common)

**Key Insight:** Premium features signal luxury vehicles, not just feature value

**Hierarchy of Importance:**
```
Tier S: modelo, potencia
Tier A: km, EDAD_VEHICULO, tipo_gasolina  
Tier B: Premium features (camera, alert)
Tier C: tipo_coche, color
Tier D: gps, fecha_registro_disponible
```

---

## **PART 3: FEATURE TRANSFORMATION & PREPARATION FOR MODELING**

### 3.1. Categorical Encoding

**Technique:** One-Hot Encoding (OHE)

```python
df_bmw_prep = pd.get_dummies(df_bmw_clean, 
                              columns=['modelo', 'tipo_gasolina', 'color', 'tipo_coche'],
                              drop_first=False)
```

**Result:** 4 categorical → 101 binary columns
- modelo: 76 columns
- tipo_gasolina: 5 columns
- color: 11 columns
- tipo_coche: 9 columns

**Why OHE? (Brief):** Variables are nominal (no order), 101 categories manageable, preserves all information.

### 3.2. Numerical Scaling

**Technique:** MinMaxScaler [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_bmw_prep[['km', 'potencia', 'EDAD_VEHICULO']] = scaler.fit_transform(...)
```

**Why normalize?:** Puts all variables on same scale for algorithms sensitive to magnitude (LinearReg, SVM, NN)

### 3.3. Final Dataset Verification

```python
print(df_bmw_prep.shape)  # (4,827, 115)
print(df_bmw_prep.isnull().sum().sum())  # 0
print(df_bmw_prep.dtypes.value_counts())
```

**Final Stats:**
- **Dimensions:** 4,827 records × 115 features
- **Nulls:** 0 ✅
- **Data types:** All numeric (ready for ML)
- **Records lost:** Only 16 from original 4,843 (0.3%)

### 3.4. Save Preprocessed Dataset

```python
df_bmw_prep.to_pickle("dataset_bmw_preprocesado.pkl")
```

**Checkpoint:** Dataset 100% ready for modeling ✅

---

## **PART 4: ANSWERS TO PROJECT QUESTIONS** 
*(DETAILED 5-step justifications HERE ONLY)*

> **Note:** This section provides detailed justifications for all preprocessing decisions made in Parts 1-3. For understanding the work flow, refer to the execution sections above. For understanding the reasoning, read below.

---

### ❓ Question 1: ¿Qué columnas eliminaron inicialmente del dataset y por qué?

#### **Columna 1: `marca`**

**1. IDENTIFICACIÓN DEL PROBLEMA:**
...
(Your existing detailed 5-step content)

#### **Columna 2: `asientos_traseros_plegables`**

**1. IDENTIFICACIÓN DEL PROBLEMA:**
...
(Your existing detailed 5-step content)

---

### ❓ Question 2: Manejo de nulos, explicar qué se hizo con los nulos por cada columna

**Summary Table:**

| Variable | Nulos | Estrategia | Justificación |
|----------|-------|------------|---------------|
| precio | 6 (0.12%) | Eliminación | Target variable, can't train without it |
| modelo, km, potencia | 6 | Eliminación | Critical predictors, few nulls |
| fecha_venta | 1 | Eliminación | Needed for EDAD_VEHICULO |
| **fecha_registro** | **2,414 (50%)** | **Imputation (km/año)** | **See detailed justification below** |
| tipo_gasolina | 5 | Moda (diesel) | 95.7% are diesel |
| tipo_coche | 1,455 | "desconocido" | Preserve missingness info |
| color | 444 | "desconocido" | Preserve missingness info |
| Boolean vars | 1,947 | False | Absence = no feature |

#### **⭐ Detailed Case: `fecha_registro` (50% nulls)**

**1. IDENTIFICACIÓN DEL PROBLEMA:**
...
(Your existing detailed 5-step content for fecha_registro)

---

### ❓ Question 2.1: Tratamiento de outliers

**1. IDENTIFICACIÓN DEL PROBLEMA:**
...
(Your existing detailed outlier content)

---

### ❓ Question 3: Análisis univariable - Información interesante encontrada

**Key Insights:**

**Numerical Variables:**
- **precio:** Skewed right (mean > median), luxury cars pull mean up
- **km:** Symmetric distribution, 140k avg suggests moderate use
- **potencia:** Range 0-423 CV, M-series dominates high end
- **EDAD_VEHICULO:** Average 5.4 years, validates imputation

**Categorical Variables:**
- **modelo:** 76 types, 320 most common (15.5%) but i8 most expensive (95k€)
- **tipo_gasolina:** 95.7% diesel (typical European market)
- **color:** Neutral colors dominate (black, grey, white)

**Boolean Variables:**
- **GPS:** 93% penetration → no price discriminator
- **Bluetooth:** Only 21% but adds +4,200€ when present → luxury indicator
- **Alerta velocidad:** Adds +7,032€ → strongest boolean predictor

**💡 Surprising Finding:** GPS doesn't add value because it's standard. Rare premium features (bluetooth, camera) signal luxury tier.

*(Keep this section as insights summary, not detailed analysis)*

---

### ❓ Question 4: Análisis de correlación - Variables correlacionadas

**Correlation Findings:**

**With Target (precio):**
- potencia: r=0.639 (strong positive) ⭐ Best numerical predictor
- EDAD_VEHICULO: r=-0.423 (moderate negative)
- km: r=-0.409 (moderate negative)
- fecha_registro_disponible: r=0.020 (no correlation → imputation unbiased ✅)

**Between Predictors:**
- ⚠️ **km ↔ EDAD_VEHICULO: r=0.730** (HIGH) → Multicollinearity risk for linear models

**Implication for Modeling:**
- For Linear Regression: Consider dropping km or EDAD_VEHICULO
- For tree-based (RF, XGBoost): Keep both, they handle multicollinearity
- Imputation validation: fecha_registro_disponible uncorrelated with price → no systematic bias ✅

*(Brief summary, no need for 5-step here since it's exploratory)*

---

### ❓ Question 5: Análisis variable vs target - Insights interesantes

**Most Determinant Factors of BMW Price:**

**1. MODELO** (Categorical - 90,700€ range)
   - i8: 95,200€ (hybrid sports)
   - M-series: 40,000-60,000€ (sports sedans)
   - Serie 1: 4,500-10,000€ (economy)
   - **Insight:** Model encapsulates brand, power, luxury → strongest overall predictor

**2. POTENCIA** (Numerical - r=0.639)
   - 300+ CV → 40,000€+
   - 100-150 CV → 10,000-15,000€
   - **Insight:** Technical spec with clearest linear relationship

**3. TIPO_GASOLINA** (Categorical - +135% for hybrid)
   - Hybrid: 37,575€ (but only 8 vehicles)
   - Diesel/petrol: ~15,000€ (majority)
   - **Insight:** More about identifying special models (i8, i3) than fuel type

**4. USAGE (km & EDAD_VEHICULO)**
   - Depreciation: -0.409 (km) and -0.423 (age)
   - **Insight:** Age slightly stronger than mileage for depreciation

**5. PREMIUM FEATURES** (Boolean - up to +7,032€)
   - Alerta velocidad: +55.9%
   - Cámara trasera: +39.7%
   - **Insight:** Features are INDICATORS of luxury trim, not just feature value

**Counterintuitive Finding:**
- GPS has NEGATIVE impact on price (-247€, not significant)
- Reason: 93% of cars have it → standard equipment, no differentiation

**Hierarchy:**
```
Tier S (Critical): modelo, potencia
Tier A (Very Important): km, EDAD_VEHICULO, tipo_gasolina
Tier B (Important): Premium features (camera, alert, bluetooth)
Tier C (Relevant): tipo_coche, color
Tier D (Minimal impact): gps, fecha_registro_disponible
```

*(Summary format, not 5-step since exploratory)*

---

### ❓ Question 6: Transformación de categóricas - Técnica usada

**1. IDENTIFICACIÓN DEL PROBLEMA:**
...
(Your existing detailed 5-step OHE content)

---

## **PART 5: CONCLUSIONS & NEXT STEPS** *(Optional but professional)*

### 5.1. Data Preparation Summary

**Achieved:**
- ✅ 0 null values (from 4,800+ nulls initially)
- ✅ Advanced imputation for 50% missing fecha_registro
- ✅ 311 luxury outliers preserved (validated as genuine)
- ✅ All categorical variables encoded
- ✅ Numerical variables scaled
- ✅ Only 16 records lost (0.3% of original dataset)

**Final Dataset:**
- **Records:** 4,827
- **Features:** 115
- **Quality:** Ready for ML modeling ✅

### 5.2. Key Insights for Modeling

**Strong Predictors:**
- modelo (categorical, 76 levels)
- potencia (continuous, r=0.639)
- EDAD_VEHICULO (continuous, r=-0.423)
- tipo_gasolina (categorical, identifies special models)

**Considerations:**
- ⚠️ Multicollinearity: km ↔ EDAD_VEHICULO (r=0.730)
- Strategy: Use regularization (Ridge/Lasso) OR tree-based models
- Alternative: Create km_per_year feature, drop originals

**Expected Model Performance:**
- Good predictability (strong correlations exist)
- Non-linear relationships likely (luxury tier effects)
- Recommended algorithms: Random Forest, XGBoost, Gradient Boosting

### 5.3. Next Steps

1. **Split data:** Train/validation/test sets (70/15/15)
2. **Baseline model:** Linear regression with regularization
3. **Advanced models:** Random Forest, XGBoost
4. **Feature importance:** Validate our EDA insights
5. **Hyperparameter tuning:** Grid search / Bayesian optimization
6. **Model evaluation:** RMSE, MAE, R² on test set
7. **Business interpretation:** Feature coefficients analysis

---

**End of Notebook** 🎉

---

# 📏 STRUCTURE PRINCIPLES APPLIED:

1. **Separation of Concerns:**
   - Parts 1-3: EXECUTION (what you did)
   - Part 4: JUSTIFICATION (why you did it)
   - Part 5: REFLECTION (what it means)

2. **Progressive Detail:**
   - Execution: Brief (1-2 lines)
   - Questions: Detailed (5-step)
   - No redundancy

3. **Clear Hierarchy:**
   - ## PARTS (major sections)
   - ### Subsections (logical groups)
   - #### Topics (specific operations)
   - Bold for emphasis

4. **Reader-Friendly:**
   - Executive summary at each level
   - Can skip to answers if in hurry
   - Can read execution to understand flow
   - Visual separators (---)

5. **Professional Touches:**
   - Checkpoints (save files)
   - Verification steps
   - Summary tables
   - Next steps section
   - Emoji for visual scanning (optional)

