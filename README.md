# Fashion Dataset Cleaning & Product Classification

This project involves cleaning and processing a fashion dataset, engineering useful features, and applying machine learning models to classify luxury vs non-luxury products based on attributes like brand, type, and color.

## Dataset

The dataset used is a CSV file: `fashion_dataset.csv`, which contains fields such as:
- `brand`
- `type`
- `description`
- `price_usd`



###  1. Data Cleaning
- Standardized text columns (`brand`, `type`)
- Removed duplicate rows
- Handled and flagged outliers (e.g., products priced above $5000)

### 2. Feature Engineering
- Extracted color and product type from `description` using keyword matching
- Added a `colors_combined` column based on recognized CSS and fashion color names
- Labeled products as **luxury** if the brand's average price was above $1000

### 3. Data Transformation
- One-hot encoded `type`, `colors`, and `product_type` columns for ML modeling
- Renamed and dropped intermediate columns to tidy the dataset

### 4. Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Each model was evaluated on classification accuracy and a classification report was generated to assess precision, recall, and F1-score.

---

## 📈 Results

- Model accuracies were compared, with **XGBoost** typically showing the highest accuracy.
- Top features influencing the luxury classification were extracted and visualized using a horizontal bar plot.

---
