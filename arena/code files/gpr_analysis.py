import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# === Load and preprocess data ===
df = pd.read_csv('/home/ridhima/workspaces/surge/src/arena/code files/arena1.csv')
df = df.rename(columns={'mf.Bx (T)': 'Bx', 'mf.By (T)': 'By', 'mf.Bz (T)': 'Bz'})
df_xy = df[np.abs(df['z']) < 7 + 1e-3]  # Flat z-slice
X_known = df_xy[['x', 'y']].values

# === Load trained models ===
components = ['Bx', 'By', 'Bz']
models = {}

for comp in components:
    filename = f'sgpr_model_{comp}.pkl'
    with open(filename, 'rb') as f:
        model, scaler = pickle.load(f)
    models[comp] = (model, scaler)
    print(f"✅ Loaded model for {comp} from {filename}")

# === Rescale known input points for prediction ===
scaler = models['Bx'][1]  # Same scaler used for all
X_known_scaled = scaler.transform(X_known)

# === Generate prediction grid ===
x_lin = np.linspace(df_xy['x'].min(), df_xy['x'].max(), 200)
y_lin = np.linspace(df_xy['y'].min(), df_xy['y'].max(), 200)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
X_pred_scaled = scaler.transform(X_pred)

# === Evaluation function (normal space) ===
def evaluate_sgpr_component(comp, model_tuple, df_xy, X_known_scaled, X_pred_scaled, X_grid, Y_grid):
    print(f"\n📊 Evaluation for {comp}")

    m, _ = model_tuple
    y_true = df_xy[comp].values.reshape(-1, 1)
    y_pred, _ = m.predict(X_known_scaled)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    # --- True vs Predicted plot ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=5, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{comp}: True vs Predicted")
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Residual histogram ---
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=50, color='gray', edgecolor='black')
    plt.title(f"{comp} Residuals")
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Uncertainty (std dev heatmap) ---
    _, var_grid = m.predict(X_pred_scaled)
    std_grid = np.sqrt(var_grid).reshape(X_grid.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_grid, Y_grid, std_grid, levels=100, cmap='viridis')
    plt.colorbar(contour, label='Prediction Std Dev')
    plt.title(f"{comp} Prediction Uncertainty")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# === Evaluate all components ===
for comp in components:
    evaluate_sgpr_component(comp, models[comp], df_xy, X_known_scaled, X_pred_scaled, X_grid, Y_grid)
