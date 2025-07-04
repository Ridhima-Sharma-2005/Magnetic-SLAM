import pandas as pd
import numpy as np
import GPy
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# --- 1. Load CSV and preprocess ---
df = pd.read_csv('/home/ridhima/workspaces/surge/src/arena/code files/arena1.csv')
df = df.rename(columns={'mf.Bx (T)': 'Bx', 'mf.By (T)': 'By', 'mf.Bz (T)': 'Bz'})

# Take a slice with fixed z for simplicity
df_xy = df[np.abs(df['z']) < 7 + 1e-3]

X = df_xy[['x', 'y']].values
Y = df_xy[['Bz']].values

# Scale inputs (not target)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 2. Define kernel and inducing points ---
kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=1.0)
Z = X_scaled[np.random.choice(len(X_scaled), size=200, replace=False)]  # 200 inducing points

# --- 3. Create and train SGPR model ---
model = GPy.models.SparseGPRegression(X_scaled, Y, kernel=kernel, Z=Z)
model.optimize(messages=True)

# --- 4. Prediction on a grid ---
x_lin = np.linspace(df_xy['x'].min(), df_xy['x'].max(), 100)
y_lin = np.linspace(df_xy['y'].min(), df_xy['y'].max(), 100)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
X_pred_scaled = scaler.transform(X_pred)

Y_pred, _ = model.predict(X_pred_scaled)
Y_pred_grid = Y_pred.reshape(X_grid.shape)

# --- 5. Plot ---
plt.figure(figsize=(8, 6))
contour = plt.contourf(X_grid, Y_grid, Y_pred_grid, levels=100, cmap='coolwarm')
plt.colorbar(contour, label='Interpolated Bx (T)')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.title('SGPR Interpolation: Bx Component')
plt.axis('equal')
plt.tight_layout()
plt.show()

# --- 6. Save model and scaler ---
with open("sgpr_model_Bz2.pkl", "wb") as f:
    pickle.dump((model, scaler), f)
print("✅ SGPR model and scaler saved to sgpr_model_Bx.pkl")
