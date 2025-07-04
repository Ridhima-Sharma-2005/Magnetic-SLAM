import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# --- Helper for signed log1p ---
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

# --- 1. Load and preprocess data ---
df = pd.read_csv('/home/ridhima/workspaces/surge/src/arena/code files/arena1.csv')
df = df.rename(columns={'mf.Bx (T)': 'Bx', 'mf.By (T)': 'By', 'mf.Bz (T)': 'Bz'})
df_xy = df[np.abs(df['z']) < 7 + 1e-3]  # Flat z-slice

X_known = df_xy[['x', 'y']].values
scaler = StandardScaler()
X_known_scaled = scaler.fit_transform(X_known)

# --- 2. Estimate initial RBF lengthscale ---
X_sample = X_known_scaled[np.random.choice(len(X_known_scaled), 2000, replace=False)]
dists = pairwise_distances(X_sample)
dists_flat = dists[np.triu_indices_from(dists, k=1)]
initial_lengthscale = np.percentile(dists_flat, 20)
print(f"📏 Initial RBF lengthscale: {initial_lengthscale:.2f}")

# --- 3. Train SGPR model for each component ---
models = {}

for comp in ['Bx', 'By', 'Bz']:
    print(f"\n🔧 Training SGPR for {comp}...")

    y_raw = df_xy[comp].values.reshape(-1, 1)
    y_log = signed_log1p(y_raw)

    # Weighted inducing sampling to bias toward high-field areas
    B_mag = np.sqrt(df_xy['Bx']**2 + df_xy['By']**2 + df_xy['Bz']**2)
    weights = B_mag + 1e-6
    weights /= weights.sum()
    inducing_idx = np.random.choice(len(X_known_scaled), 100, replace=False, p=weights)
    Z = X_known_scaled[inducing_idx]

    # Define kernel
    kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=initial_lengthscale)

    # Create model
    m = GPy.models.SparseGPRegression(X_known_scaled, y_log, kernel=kernel, Z=Z)

    # --- Priors and constraints ---
    m.kern.variance.set_prior(GPy.priors.LogGaussian(0, 1))
    m.kern.lengthscale.set_prior(GPy.priors.LogGaussian(np.log(initial_lengthscale), 0.5))
    # m.kern.variance.constrain_bounded(1e-6, 10.0)
    # m.kern.lengthscale.constrain_bounded(1e-2, 10.0)

    m.likelihood.variance = 1e-2
    m.likelihood.variance.set_prior(GPy.priors.LogGaussian(np.log(1e-2), 1.0))
    # m.likelihood.variance.constrain_bounded(1e-6, 1.0)

    # --- Sanity checks ---
    print(f"Initial lengthscale: {m.kern.lengthscale.values}")
    print(f"Initial variance: {m.kern.variance.values}")
    print(f"Initial noise variance: {m.likelihood.variance.values}")

    # --- Two-stage optimization ---
    m.optimize(optimizer='scg', messages=True, max_iters=300)
    m.optimize(optimizer='lbfgs', messages=True, max_iters=500)

    print(m)

    # Save model and scaler
    model_path = f'gp_sparse_model_{comp.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((m, scaler), f)
    print(f"✅ Model for {comp} saved to: {model_path}")

    models[comp] = (m, scaler)

# --- 4. Prediction Grid and Visualization ---
x_lin = np.linspace(df_xy['x'].min(), df_xy['x'].max(), 200)
y_lin = np.linspace(df_xy['y'].min(), df_xy['y'].max(), 200)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
X_pred_scaled = scaler.transform(X_pred)

for comp in ['Bx', 'By', 'Bz']:
    print(f"\n📈 Predicting and plotting for {comp}")
    m, _ = models[comp]
    pred_log, _ = m.predict(X_pred_scaled)
    pred = np.sign(pred_log) * (np.expm1(np.abs(pred_log)))  # Inverse of signed_log1p
    B_pred_grid = pred.reshape(X_grid.shape)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X_grid, Y_grid, B_pred_grid, levels=100, cmap='coolwarm')
    plt.colorbar(contour, label=f'Interpolated {comp} (T)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title(f'SGPR-Interpolated Magnetic Field: {comp}')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
