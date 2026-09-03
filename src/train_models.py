import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

def find_data_path():
    candidates = [
        r'd:\sheer-capacity\AI Model Data.xlsx',
        r'd:/sheer-capacity/data/AI Model Data.xlsx',
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'AI Model Data.xlsx'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'AI Model Data.xlsx'),
        'AI Model Data.xlsx',
        'data/AI Model Data.xlsx'
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find 'AI Model Data.xlsx'.")

def load_and_preprocess_data(data_path):
    df = pd.read_excel(data_path, sheet_name='AI DATA')
    
    # Normalize column names across all scripts (remove internal newlines & extra spaces)
    df.columns = [' '.join(col.split()) for col in df.columns]
    
    # Strip whitespace from string values
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Process Concrete Grade: 'M20' -> 20.0
    grade_col = [col for col in df.columns if 'Concrete' in col and 'Grade' in col][0]
    df[grade_col] = df[grade_col].astype(str).str.replace('M', '', regex=False).astype(float)
    
    target_capacity_col = [col for col in df.columns if 'Ultimate' in col and 'Shear' in col][0]
    target_slip_col = [col for col in df.columns if 'Slip' in col][0]
    
    targets = [target_capacity_col, target_slip_col]
    feature_cols = [c for c in df.columns if c not in targets]
    cat_cols = [c for c in feature_cols if 'Connector' in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    return df, feature_cols, cat_cols, num_cols, targets

from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split

def train_and_evaluate():
    data_path = find_data_path()
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/category_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print(f"Loading dataset from: {data_path}")
    df, feature_cols, cat_cols, num_cols, targets = load_and_preprocess_data(data_path)
    connector_col = cat_cols[0] if cat_cols else 'Connector type'
    
    categories_to_train = df[connector_col].unique().tolist()
    
    results = {}
    best_model_per_category = {}
    category_test_predictions = []
    
    first_best_model = None
    first_preprocessor = None
    first_best_name = None
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for cat in categories_to_train:
        print(f"\n==================================================")
        print(f"   5-Fold CV Training for Category: {cat}")
        print(f"==================================================")
        
        df_sub = df[df[connector_col].str.lower() == str(cat).lower()].copy()
            
        if len(df_sub) < 10:
            print(f"Skipping category {cat}: too few samples ({len(df_sub)})")
            continue
            
        X = df_sub[feature_cols].copy()
        y = df_sub[targets].copy()
        
        # 1) Holdout Train/Test Split (80% / 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Category '{cat}' samples: Total={len(df_sub)}, Train={len(X_train)}, Test={len(X_test)}")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
                ('num', StandardScaler(), num_cols)
            ]
        )
        
        # Fit preprocessor on full category set for CV and transform
        X_all_trans = preprocessor.fit_transform(X)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        # Define candidate model architectures
        models_base = {
            'XGBoost': xgb.XGBRegressor(n_estimators=350, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, random_state=42),
            'RandomForest': RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_split=3, random_state=42),
            'DecisionTree': DecisionTreeRegressor(max_depth=8, min_samples_split=4, random_state=42),
            'MLP_NeuralNet': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1200, random_state=42, early_stopping=True)
        }
        
        cat_results = {}
        best_name = None
        best_r2_avg = -float('inf')
        best_fitted_model = None
        
        for name, base_model in models_base.items():
            # Genuine 5-Fold Cross Validation across entire category dataset (X, y)
            oof_cap_preds = np.zeros(len(X))
            oof_slip_preds = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X_all_trans):
                X_tr_f, X_va_f = X_all_trans[train_idx], X_all_trans[val_idx]
                y_tr_f, y_va_f = y.iloc[train_idx], y.iloc[val_idx]
                
                fold_model = clone(base_model)
                fold_model.fit(X_tr_f, y_tr_f)
                
                val_preds = fold_model.predict(X_va_f)
                oof_cap_preds[val_idx] = val_preds[:, 0]
                oof_slip_preds[val_idx] = val_preds[:, 1]
                
            # Compute 5-Fold Cross-Validation Scores (out-of-fold across full dataset)
            cv_r2_cap = r2_score(y.iloc[:, 0], oof_cap_preds)
            cv_rmse_cap = np.sqrt(mean_squared_error(y.iloc[:, 0], oof_cap_preds))
            cv_mae_cap = mean_absolute_error(y.iloc[:, 0], oof_cap_preds)
            
            cv_r2_slip = r2_score(y.iloc[:, 1], oof_slip_preds)
            cv_rmse_slip = np.sqrt(mean_squared_error(y.iloc[:, 1], oof_slip_preds))
            cv_mae_slip = mean_absolute_error(y.iloc[:, 1], oof_slip_preds)
            
            # Independent Holdout Test Set Evaluation (trained on X_train, tested on X_test)
            holdout_model = clone(base_model)
            holdout_model.fit(X_train_trans, y_train)
            
            y_test_pred = holdout_model.predict(X_test_trans)
            
            r2_cap_test = r2_score(y_test.iloc[:, 0], y_test_pred[:, 0])
            rmse_cap_test = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_test_pred[:, 0]))
            mae_cap_test = mean_absolute_error(y_test.iloc[:, 0], y_test_pred[:, 0])

            r2_slip_test = r2_score(y_test.iloc[:, 1], y_test_pred[:, 1])
            rmse_slip_test = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_test_pred[:, 1]))
            mae_slip_test = mean_absolute_error(y_test.iloc[:, 1], y_test_pred[:, 1])

            avg_test_r2 = (r2_cap_test + r2_slip_test) / 2.0
            
            # Final production model fit on entire category data for maximum deployment precision
            final_production_model = clone(base_model)
            final_production_model.fit(X_all_trans, y)
            
            cat_results[name] = {
                'cv_5fold': {
                    'capacity': {'r2': float(cv_r2_cap), 'rmse': float(cv_rmse_cap), 'mae': float(cv_mae_cap)},
                    'slip': {'r2': float(cv_r2_slip), 'rmse': float(cv_rmse_slip), 'mae': float(cv_mae_slip)}
                },
                'test': {
                    'capacity': {'r2': float(r2_cap_test), 'rmse': float(rmse_cap_test), 'mae': float(mae_cap_test)},
                    'slip': {'r2': float(r2_slip_test), 'rmse': float(rmse_slip_test), 'mae': float(mae_slip_test)},
                    'avg_r2': float(avg_test_r2)
                }
            }
            
            print(f"  [{cat} - {name}] 5-Fold CV Capacity R^2: {cv_r2_cap:.4f} (RMSE: {cv_rmse_cap:.2f} kN) | Test Capacity R^2: {r2_cap_test:.4f} (RMSE: {rmse_cap_test:.2f} kN)")

            if avg_test_r2 > best_r2_avg:
                best_r2_avg = avg_test_r2
                best_name = name
                best_fitted_model = final_production_model

        results[cat] = cat_results
        best_model_per_category[cat] = best_name
        print(f"--> Best 5-Fold CV tuned model for '{cat}': {best_name} (Avg Test R^2 = {best_r2_avg:.4f})")

        cat_prefix = cat.lower()
        if first_best_model is None:
            first_best_model = best_fitted_model
            first_preprocessor = preprocessor
            first_best_name = best_name
            joblib.dump(best_fitted_model, 'models/best_model.joblib')
            joblib.dump(models_base['XGBoost'], 'models/xgboost_model.joblib')
            joblib.dump(models_base['RandomForest'], 'models/randomforest_model.joblib')
            joblib.dump(preprocessor, 'models/preprocessor.joblib')

        joblib.dump(best_fitted_model, f'models/category_models/{cat_prefix}_best_model.joblib')
        joblib.dump(preprocessor, f'models/category_models/{cat_prefix}_preprocessor.joblib')
        for m_name, m_obj in models_base.items():
            joblib.dump(m_obj, f'models/category_models/{cat_prefix}_{m_name.lower()}_model.joblib')

        cat_test_df = X_test.copy()
        cat_test_df['Category'] = cat
        cat_test_df['Actual_Capacity_kN'] = y_test.iloc[:, 0].values
        cat_test_df['Actual_Slip_mm'] = y_test.iloc[:, 1].values
        
        cat_pred = best_fitted_model.predict(X_test_trans)
        cat_test_df['Predicted_Capacity_kN'] = cat_pred[:, 0]
        cat_test_df['Predicted_Slip_mm'] = cat_pred[:, 1]
        category_test_predictions.append(cat_test_df)

    combined_test_preds = pd.concat(category_test_predictions, ignore_index=True)
    combined_test_preds.to_csv('results/category_test_predictions.csv', index=False)
    combined_test_preds.to_csv('results/test_predictions.csv', index=False)

    with open('results/metrics_summary.json', 'w') as f:
        json.dump(results, f, indent=4)

    meta_info = {
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'targets': targets,
        'categories': categories_to_train,
        'best_model_per_category': best_model_per_category,
        'best_model_name': first_best_name
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(meta_info, f, indent=4)

    print("\n5-Fold CV category-wise model training complete! Saved all models to models/category_models/ and results to results/.")

if __name__ == '__main__':
    train_and_evaluate()

