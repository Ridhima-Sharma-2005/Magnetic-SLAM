import pandas as pd
import numpy as np
import GPy
import pickle
import matplotlib.pyplot as plt
import os

# === Configuration ===
CSV_PATH = 'code files/arena1.csv'  # <-- replace with full path
SCALE_FACTOR = 1e6       # T → µT for numerical stability
MODEL_DIR = 'models'
GRID_RES_MM = 2

# === Load and preprocess ===
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    'mf.Bx (T)': 'Bx',
    'mf.By (T)': 'By',
    'mf.Bz (T)': 'Bz'
})
df = df[np.abs(df['z']) < 7+1e-2]  # Keep z ≈ 0

# === Training inputs ===
X_train = df[['x', 'y']].values

# === Train models ===
components = ['Bx', 'By', 'Bz']
models = {}

os.makedirs(MODEL_DIR, exist_ok=True)

for comp in components:
    y_raw = df[comp].values.reshape(-1, 1)
    y_scaled = y_raw * SCALE_FACTOR

    print(f"\n🔧 Training SGPR model for {comp}...")

    kernel = GPy.kern.RBF(input_dim=2, lengthscale=50.0)
    m = GPy.models.SparseGPRegression(X_train, y_scaled, kernel=kernel, num_inducing=200)
    m.optimize(messages=True, max_iters=500)

    models[comp] = m

    model_path = os.path.join(MODEL_DIR, f'gp_sparse_model_{comp.lower()}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(m, f)

    print(f"✅ Model saved to: {model_path}")

# === Optional: Predict on a 2D grid and save result ===
x_lin = np.arange(df['x'].min(), df['x'].max(), GRID_RES_MM)
y_lin = np.arange(df['y'].min(), df['y'].max(), GRID_RES_MM)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])

output = {
    'x_grid': x_lin,
    'y_grid': y_lin
}

for comp in components:
    m = models[comp]
    pred_scaled, _ = m.predict(X_pred)
    pred = pred_scaled / SCALE_FACTOR
    pred_grid = pred.reshape(X_grid.shape)
    output[comp.lower()] = pred_grid

    # Plot each component
    plt.figure(figsize=(8, 6))
    plt.contourf(X_grid, Y_grid, pred_grid, levels=100, cmap='coolwarm')
    plt.colorbar(label=f'{comp} (T)')
    plt.title(f'SGPR Magnetic Field – {comp}')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Save full grid
np.savez_compressed(os.path.join(MODEL_DIR, 'magnetic_field_grid_2mm.npz'), **output)
print(f"\n🧾 Interpolated grid saved to: {MODEL_DIR}/magnetic_field_grid_2mm.npz")
