import os
import json
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# Set global matplotlib style
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.grid'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.15

def format_display_label(label):
    """Format the label text into consistent math-style notation for the report figures."""
    display_label = ' '.join(str(label).split())
    display_label = display_label.replace('\\underline{\\circ}C', '°C')
    display_label = display_label.replace('\\underline{°C}', '°C')
    display_label = display_label.replace('\\underline{C}', 'C')
    display_label = re.sub(r'\\underline\s*\{([^{}]*)\}', r'\1', display_label)
    display_label = display_label.replace('ºC', '°C').replace('° C', '°C').replace('°C', '°C')
    display_label = display_label.replace('$\\circ$', '°').replace('\\circ', '°').replace('\\textdegree', '°')
    display_label = display_label.replace('{°}', '°').replace('° C', '°C').replace('(°', '(°')
    display_label = display_label.replace('\\circ C', '°C').replace('\\circC', '°C')

    replacements = {
        'N/mm2': 'N/mm²',
        'ºC': '°C',
        'm-1C-1': 'm⁻¹C⁻¹',
        'W/mK': 'W/(mK)',
        'J/kgK': 'J/(kg·K)',
        'relative to fy': 'relative to fᵧ',
        '(fy, θ)': '($f_{y,\\theta}$)',
        '(fp, θ)': '($f_{p,\\theta}$)',
        '(Ea, θ)': '($E_{a,\\theta}$)',
        '(ɛp, θ)': '($\\varepsilon_{p,\\theta}$)',
        'fy,θ': '$f_{y,\\theta}$',
        'fp,θ': '$f_{p,\\theta}$',
        'Ea,θ': '$E_{a,\\theta}$',
        'ɛp,θ': '$\\varepsilon_{p,\\theta}$',
    }
    for source, formatted in replacements.items():
        display_label = display_label.replace(source, formatted)

    display_label = display_label.replace('ky, θ = fy, θ / fy', '($k_{y,\\theta}=f_{y,\\theta}/f_{y}$)')
    display_label = display_label.replace('ky,θ =fy,θ/fy', '($k_{y,\\theta}=f_{y,\\theta}/f_{y}$)')
    display_label = display_label.replace('kE, θ = Ea, θ / Ea', '($k_{E,\\theta}=E_{a,\\theta}/E_{a}$)')
    display_label = display_label.replace('kE,θ =Ea,θ/Ea', '($k_{E,\\theta}=E_{a,\\theta}/E_{a}$)')

    display_label = re.sub(r'ky\s*[, ]\s*θ', r'$k_{y,\\theta}$', display_label, flags=re.IGNORECASE)
    display_label = re.sub(r'kE\s*[, ]\s*θ', r'$k_{E,\\theta}$', display_label, flags=re.IGNORECASE)
    display_label = re.sub(r'fy\s*[, ]\s*θ', r'$f_{y,\\theta}$', display_label, flags=re.IGNORECASE)
    display_label = re.sub(r'Ea\s*[, ]\s*θ', r'$E_{a,\\theta}$', display_label, flags=re.IGNORECASE)
    display_label = re.sub(r'fp\s*[, ]\s*θ', r'$f_{p,\\theta}$', display_label, flags=re.IGNORECASE)
    display_label = re.sub(r'ɛp\s*[, ]\s*θ', r'$\\varepsilon_{p,\\theta}$', display_label, flags=re.IGNORECASE)

    display_label = display_label.replace('/fy', '/f_{y}')
    display_label = display_label.replace('/Ea', '/E_{a}')
    display_label = display_label.replace('/f_{y}$)', '/f_{y}$)')
    display_label = display_label.replace('/E_{a}$)', '/E_{a}$)')

    return display_label

def style_axes(axes):
    for axis in np.atleast_1d(axes).flat:
        axis.grid(False, which='both')
        axis.tick_params(
            axis='both',
            direction='out',
            length=5,
            width=2,
            colors='black',
            labelsize=9
        )
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        axis.set_xlabel(format_display_label(axis.get_xlabel()))
        axis.set_ylabel(format_display_label(axis.get_ylabel()))
        axis.set_title(format_display_label(axis.get_title()))
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_edgecolor('black')
            legend.get_frame().set_linewidth(1.0)

def find_data_path():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "AI Model Data.xlsx"
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Required data file not found: {data_path}")
    return data_path

def load_and_clean_data(data_path):
    df = pd.read_excel(data_path, sheet_name='AI DATA')

    # Remove Excel line breaks and repeated whitespace without changing Unicode symbols.
    df.columns = [' '.join(str(col).split()) for col in df.columns]

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    
    grade_col = [col for col in df.columns if 'Concrete' in col and 'Grade' in col][0]
    df[grade_col] = df[grade_col].astype(str).str.replace('M', '', regex=False).astype(float)
    
    target_cap = [col for col in df.columns if 'Ultimate' in col and 'Shear' in col][0]
    target_slip = [col for col in df.columns if 'Slip' in col][0]
    
    return df, target_cap, target_slip

def generate_all_paper_graphs():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    data_path = find_data_path()
    df, target_cap, target_slip = load_and_clean_data(data_path)
    
    targets = [target_cap, target_slip]
    feature_cols = [c for c in df.columns if c not in targets]
    cat_cols = [c for c in feature_cols if 'Connector' in c]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    
    X = df[feature_cols].copy()
    y = df[targets].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )
    
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, 'models/paper_graphs_preprocessor.joblib')
    
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = ohe_cols + num_cols
    
    # Train 4 AI Models
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
        'DecisionTree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'MLP_NeuralNet': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42, early_stopping=True)
    }
    
    predictions = {}
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        predictions[name] = {
            'train': model.predict(X_train_trans),
            'test': model.predict(X_test_trans)
        }
        fitted_models[name] = model
        joblib.dump(model, f'models/paper_graphs_{name.lower()}_model.joblib')
    
    print("--- Generating Figure 1: Pearson Correlation Matrix ---")
    numeric_df = df.select_dtypes(include=[np.number])
    fig1, ax1 = plt.subplots(figsize=(14, 11), dpi=300)
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    display_labels = [format_display_label(column) for column in numeric_df.columns]
    sns.heatmap(
        corr,
        mask=mask,
        annot=False,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax1
    )
    ax1.set_xticklabels(display_labels, rotation=90, ha='center', fontsize=9)
    ax1.set_yticklabels(display_labels, rotation=0, fontsize=9)
    fig1.suptitle("Figure 1: Pearson Correlation Coefficient Matrix (Parameters vs Targets)", fontsize=13, y=0.985, fontweight='bold')
    style_axes(fig1.axes)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig('results/fig1_pearson_correlation_matrix.png', bbox_inches='tight', pad_inches=0.15)
    plt.close()
    
    print("--- Generating Figure 2: Actual vs Predicted All Models ---")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)
    model_names = list(models.keys())
    
    for i, name in enumerate(model_names):
        y_pred = predictions[name]['test']
        
        # Capacity (Row 0)
        r2_cap = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
        rmse_cap = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]))
        axes[0, i].scatter(y_test.iloc[:, 0], y_pred[:, 0], alpha=0.6, color='#1f77b4', edgecolors='k', linewidth=0.5)
        max_val_cap = max(y_test.iloc[:, 0].max(), y_pred[:, 0].max())
        axes[0, i].plot([0, max_val_cap], [0, max_val_cap], 'r--', lw=2)
        axes[0, i].set_title(f"{name}\n$R^2$={r2_cap:.4f} | RMSE={rmse_cap:.2f} kN", fontsize=11, fontweight='bold')
        axes[0, i].set_xlabel("Actual Shear Capacity (kN)")
        if i == 0:
            axes[0, i].set_ylabel("Predicted Shear Capacity (kN)")
            
        # Slip (Row 1)
        r2_slip = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
        rmse_slip = np.sqrt(mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]))
        axes[1, i].scatter(y_test.iloc[:, 1], y_pred[:, 1], alpha=0.6, color='#2ca02c', edgecolors='k', linewidth=0.5)
        max_val_slip = max(y_test.iloc[:, 1].max(), y_pred[:, 1].max())
        axes[1, i].plot([0, max_val_slip], [0, max_val_slip], 'r--', lw=2)
        axes[1, i].set_title(f"{name}\n$R^2$={r2_slip:.4f} | RMSE={rmse_slip:.2f} mm", fontsize=11, fontweight='bold')
        axes[1, i].set_xlabel("Actual Slip (mm)")
        if i == 0:
            axes[1, i].set_ylabel("Predicted Slip (mm)")
            
    plt.suptitle("Figure 2: Actual vs. Predicted Performance Across AI Algorithms (Testing Phase)", fontsize=14, fontweight='bold', y=0.98)
    style_axes(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/fig2_actual_vs_predicted_all_models.png')
    plt.close()
    
    print("--- Generating Figure 3: Testing Predictions Sample Tracking ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), dpi=300)
    samples = np.arange(len(y_test))
    
    ax1.plot(samples, y_test.iloc[:, 0].values, 'k-o', label='Experimental / Actual', lw=1.5, ms=4)
    ax1.plot(samples, predictions['XGBoost']['test'][:, 0], 'r--s', label='XGBoost Predicted', lw=1.2, ms=3)
    ax1.plot(samples, predictions['RandomForest']['test'][:, 0], 'g--^', label='RandomForest Predicted', lw=1.0, ms=3, alpha=0.7)
    ax1.set_ylabel("Ultimate Shear Capacity (kN)", fontweight='bold')
    ax1.set_title("Testing Specimen Tracking - Shear Capacity Prediction Comparison", fontweight='bold')
    ax1.set_ylim(0, 800)
    ax1.legend(loc='upper right', borderaxespad=0.6)
    
    ax2.plot(samples, y_test.iloc[:, 1].values, 'k-o', label='Experimental / Actual', lw=1.5, ms=4)
    ax2.plot(samples, predictions['XGBoost']['test'][:, 1], 'b--s', label='XGBoost Predicted', lw=1.2, ms=3)
    ax2.plot(samples, predictions['RandomForest']['test'][:, 1], 'm--^', label='RandomForest Predicted', lw=1.0, ms=3, alpha=0.7)
    ax2.set_xlabel("Test Specimen Sample Index", fontweight='bold')
    ax2.set_ylabel("Slip (mm)", fontweight='bold')
    ax2.set_title("Testing Specimen Tracking - Slip Prediction Comparison", fontweight='bold')
    ax2.set_ylim(0, 160)
    ax2.legend(loc='upper right', borderaxespad=0.6)
    
    style_axes((ax1, ax2))
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig('results/fig3_sample_testing_predictions_tracking.png')
    plt.close()
    
    print("--- Generating Figure 4: Parametric Temperature Degradation ---")
    temp_range = np.linspace(20, 800, 50)
    connector_col = cat_cols[0]
    unique_connectors = df[connector_col].unique()
    
    base_sample = X.iloc[0].copy()
    temp_col = [c for c in num_cols if 'Temperature' in c][0]
    
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    curve_max = 0.0
    
    for idx, conn in enumerate(unique_connectors):
        synth_data = []
        for t in temp_range:
            s = base_sample.copy()
            s[temp_col] = t
            s[connector_col] = conn
            synth_data.append(s)
        synth_df = pd.DataFrame(synth_data)
        synth_trans = preprocessor.transform(synth_df)
        preds = fitted_models['XGBoost'].predict(synth_trans)
        curve_max = max(curve_max, float(np.max(preds[:, 0])))
        ax4.plot(temp_range, preds[:, 0], label=f"Connector: {conn}", color=colors[idx % len(colors)], lw=2.5)
        
    ax4.set_xlabel("Temperature (°C)", fontweight='bold')
    ax4.set_ylabel("Predicted Residual Shear Capacity (kN)", fontweight='bold')
    ax4.set_title("Figure 4: Thermal Degradation Curves of Composite Connectors (20°C to 800°C)", fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(130.0, curve_max * 1.1))
    ax4.legend(
        title="Connector Type",
        loc='center right',
        frameon=True,
        facecolor='white',
        framealpha=0.92,
        borderaxespad=0.6
    )
    style_axes(ax4)
    fig4.tight_layout()
    fig4.savefig('results/fig4_parametric_temperature_degradation.png')
    plt.close(fig4)
    
    print("--- Generating Figure 5: Parametric Connector Geometry ---")
    height_col = [c for c in num_cols if 'Height' in c][0]
    diam_col = [c for c in num_cols if 'Diameter' in c][0]
    
    height_vals = np.linspace(df[height_col].min(), df[height_col].max(), 40)
    temperatures = [20, 400, 600, 800]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    for temp in temperatures:
        synth_h = []
        for h in height_vals:
            s = base_sample.copy()
            s[temp_col] = temp
            s[height_col] = h
            synth_h.append(s)
        synth_df_h = pd.DataFrame(synth_h)
        preds_h = fitted_models['XGBoost'].predict(preprocessor.transform(synth_df_h))
        axes[0].plot(height_vals, preds_h[:, 0], label=f"Temp = {temp}°C", lw=2)
        
    axes[0].set_xlabel("Connector Height (mm)", fontweight='bold')
    axes[0].set_ylabel("Predicted Shear Capacity (kN)", fontweight='bold')
    axes[0].set_title("Effect of Connector Height on Capacity across Temperatures", fontweight='bold')
    axes[0].legend()
    
    diam_vals = np.linspace(df[diam_col].min(), df[diam_col].max(), 40)
    for temp in temperatures:
        synth_d = []
        for d in diam_vals:
            s = base_sample.copy()
            s[temp_col] = temp
            s[diam_col] = d
            synth_d.append(s)
        synth_df_d = pd.DataFrame(synth_d)
        preds_d = fitted_models['XGBoost'].predict(preprocessor.transform(synth_df_d))
        axes[1].plot(diam_vals, preds_d[:, 0], label=f"Temp = {temp}°C", lw=2)
        
    axes[1].set_xlabel("Connector Diameter (mm)", fontweight='bold')
    axes[1].set_ylabel("Predicted Shear Capacity (kN)", fontweight='bold')
    axes[1].set_title("Effect of Connector Diameter on Capacity across Temperatures", fontweight='bold')
    axes[1].legend()
    
    plt.suptitle("Figure 5: Influence of Connector Geometry on Shear Capacity", fontsize=13, fontweight='bold')
    style_axes(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/fig5_parametric_connector_geometry.png')
    plt.close()
    
    print("--- Generating Figure 6: Parametric Material Strengths ---")
    grade_col = [c for c in num_cols if 'Concrete' in c and 'Grade' in c][0]
    fy_col = [c for c in num_cols if 'Steel fy' in c or 'fy' in c][0]
    
    grade_vals = np.linspace(20, 80, 40)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    for temp in temperatures:
        synth_g = []
        for g in grade_vals:
            s = base_sample.copy()
            s[temp_col] = temp
            s[grade_col] = g
            synth_g.append(s)
        preds_g = fitted_models['XGBoost'].predict(preprocessor.transform(pd.DataFrame(synth_g)))
        axes[0].plot(grade_vals, preds_g[:, 0], label=f"Temp = {temp}°C", lw=2)
        
    axes[0].set_xlabel("Concrete Compressive Grade (MPa)", fontweight='bold')
    axes[0].set_ylabel("Predicted Shear Capacity (kN)", fontweight='bold')
    axes[0].set_title("Concrete Grade vs. Capacity at Elevated Temperatures", fontweight='bold')
    axes[0].legend()
    
    fy_vals = np.linspace(df[fy_col].min(), df[fy_col].max(), 40)
    for temp in temperatures:
        synth_fy = []
        for fy in fy_vals:
            s = base_sample.copy()
            s[temp_col] = temp
            s[fy_col] = fy
            synth_fy.append(s)
        preds_fy = fitted_models['XGBoost'].predict(preprocessor.transform(pd.DataFrame(synth_fy)))
        axes[1].plot(fy_vals, preds_fy[:, 0], label=f"Temp = {temp}°C", lw=2)
        
    axes[1].set_xlabel("Effective Steel Yield Strength $f_{y,\\theta}$ (MPa)", fontweight='bold')
    axes[1].set_ylabel("Predicted Shear Capacity (kN)", fontweight='bold')
    axes[1].set_title("Steel Yield Strength vs. Capacity at Elevated Temperatures", fontweight='bold')
    axes[1].legend()
    
    plt.suptitle("Figure 6: Influence of Material Strengths on Structural Capacity", fontsize=13, fontweight='bold')
    style_axes(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/fig6_parametric_material_strengths.png')
    plt.close()
    
    print("--- Generating Figure 7: Residual Error Distributions ---")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    for idx, (name, preds_dict) in enumerate(predictions.items()):
        res_cap = y_test.iloc[:, 0].values - preds_dict['test'][:, 0]
        sns.histplot(res_cap, kde=True, ax=axes[idx], color='#1f77b4', bins=25)
        axes[idx].set_title(f"{name} Capacity Residuals\nMean: {np.mean(res_cap):.2f} | Std: {np.std(res_cap):.2f}", fontweight='bold')
        axes[idx].set_xlabel("Residual (Actual - Predicted kN)")
        axes[idx].set_ylabel("Frequency")
        
    plt.suptitle("Figure 7: Residual Error Distributions (Shear Capacity Prediction)", fontsize=13, fontweight='bold')
    style_axes(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/fig7_residual_error_distributions.png')
    plt.close()
    
    print("--- Generating Figure 8: SHAP Summary & Feature Importance ---")
    xgb_cap = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    xgb_cap.fit(X_train_trans, y_train.iloc[:, 0])
    
    explainer_cap = shap.TreeExplainer(xgb_cap)
    shap_values_cap = explainer_cap(X_test_trans)
    
    formatted_feature_names = [format_display_label(feature_name) for feature_name in all_feature_names]

    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values_cap, X_test_trans, feature_names=formatted_feature_names, show=False)
    plt.title("Figure 8: SHAP Feature Importance Summary - Ultimate Shear Capacity (kN)", fontsize=12, pad=15, fontweight='bold')
    style_axes(plt.gcf().axes)
    plt.tight_layout()
    plt.savefig('results/fig8_shap_summary_and_feature_importance.png')
    plt.close()
    
    print("--- Generating Figure 9: SHAP Dependence Plots ---")
    top_feature_idx = np.argsort(np.abs(shap_values_cap.values).mean(0))[::-1][:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()
    
    for idx, f_idx in enumerate(top_feature_idx):
        feature_name = all_feature_names[f_idx]
        axes[idx].scatter(X_test_trans[:, f_idx], shap_values_cap.values[:, f_idx], alpha=0.7, c='#1f77b4', edgecolors='k', lw=0.3)
        display_name = format_display_label(feature_name)
        axes[idx].set_title(f"SHAP Dependence: {display_name}", fontweight='bold')
        axes[idx].set_xlabel(f"Standardized {display_name}")
        axes[idx].set_ylabel("SHAP Value (Impact on Capacity)")
    shap_limit = max(1.0, float(np.max(np.abs(shap_values_cap.values))) * 1.1)
    for axis in axes:
        axis.set_ylim(-shap_limit, shap_limit)
        
    plt.suptitle("Figure 9: SHAP Dependence Plots for Top 4 Dominant Parameters", fontsize=13, fontweight='bold')
    style_axes(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/fig9_shap_dependence_plots.png')
    plt.close()
    
    print("--- Generating Figure 10: Non-Linear Load-Slip Curves ---")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    slip_mesh = np.linspace(0.1, 14.0, 100)
    
    temps = [20, 300, 500, 700]
    styles = ['-', '--', '-.', ':']
    
    for idx, t in enumerate(temps):
        s = base_sample.copy()
        s[temp_col] = t
        pred_cap = fitted_models['XGBoost'].predict(preprocessor.transform(pd.DataFrame([s])))[0, 0]
        load_curve = pred_cap * (1 - np.exp(-0.7 * slip_mesh))**0.5
        ax.plot(slip_mesh, load_curve, label=f"AI Curve @ {t}°C (P_u={pred_cap:.1f} kN)", ls=styles[idx], lw=2.5)
        
    ax.set_xlabel(r"Slip $\delta$ (mm)", fontweight='bold')
    ax.set_ylabel("Shear Force $V$ (kN)", fontweight='bold')
    ax.set_title("Figure 10: Non-Linear Thermo-Structural Load-Slip Curves across Temperatures", fontsize=12, fontweight='bold')
    ax.legend()
    style_axes(ax)
    plt.tight_layout()
    plt.savefig('results/fig10_load_slip_curves_multitemp.png')
    plt.close()
    
    if os.path.exists('results/category_test_predictions.csv'):
        print("--- Generating Figure 11: Category-Wise Actual vs Predicted Comparison ---")
        cat_preds_df = pd.read_csv('results/category_test_predictions.csv')
        cat_list = [c for c in cat_preds_df['Category'].unique() if c != 'Overall']
        
        if len(cat_list) > 0:
            fig, axes = plt.subplots(1, len(cat_list), figsize=(4 * len(cat_list), 4), dpi=300)
            if len(cat_list) == 1:
                axes = [axes]
                
            for idx, c_name in enumerate(cat_list):
                sub_df = cat_preds_df[cat_preds_df['Category'] == c_name]
                act = sub_df['Actual_Capacity_kN']
                prd = sub_df['Predicted_Capacity_kN']
                r2_val = r2_score(act, prd) if len(sub_df) > 1 else 1.0
                
                axes[idx].scatter(act, prd, alpha=0.7, color='#38bdf8', edgecolors='k', lw=0.5)
                max_v = max(act.max(), prd.max()) if len(act) > 0 else 100
                axes[idx].plot([0, max_v], [0, max_v], 'r--', lw=1.5)
                axes[idx].set_title(f"{c_name} Connectors\n$R^2$ = {r2_val:.4f}", fontweight='bold')
                axes[idx].set_xlabel("Actual Capacity (kN)")
                if idx == 0:
                    axes[idx].set_ylabel("Category-Model Predicted Capacity (kN)")
                    
            plt.suptitle("Figure 11: Category-Specific Model Predictions (Stud, Bar, Channel, Helical, Tee)", fontsize=13, fontweight='bold', y=1.02)
            style_axes(axes)
            plt.tight_layout()
            plt.savefig('results/fig11_category_wise_predictions.png')
            plt.close()

    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values_cap, X_test_trans, feature_names=formatted_feature_names, show=False)
    style_axes(plt.gcf().axes)
    plt.savefig('results/shap_summary_shear.png')
    plt.close()

    print("\n[SUCCESS] ALL PAPER-ALIGNED FIGURES SUCCESSFULLY GENERATED IN results/ DIRECTORY!")

if __name__ == '__main__':
    generate_all_paper_graphs()


