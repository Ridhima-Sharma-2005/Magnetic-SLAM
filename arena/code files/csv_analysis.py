import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# Load data
df = pd.read_csv('/home/ridhima/workspaces/surge/src/arena/code files/arena1.csv')
df = df.rename(columns={'mf.Bx (T)': 'Bx', 'mf.By (T)': 'By', 'mf.Bz (T)': 'Bz'})

# 1. Print basic stats
print("📌 Basic statistics:\n")
print(df[['x', 'y', 'z', 'Bx', 'By', 'Bz']].describe())

# 2. Filter flat z-slice
df_xy = df[np.abs(df['z']) < 7 + 1e-3]

print("\n✅ Filtered data at z ≈ 0. Points remaining:", len(df_xy))

# 3. Scatter plot of spatial distribution
plt.figure(figsize=(6, 5))
plt.scatter(df_xy['x'], df_xy['y'], s=2, alpha=0.6)
plt.title("Spatial Distribution (x vs y)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.axis('equal')
plt.tight_layout()
plt.show()

# 4. Magnetic field histogram
for comp in ['Bx', 'By', 'Bz']:
    plt.figure()
    plt.hist(df_xy[comp], bins=100, alpha=0.7, color='steelblue')
    plt.title(f"Histogram of {comp} (T)")
    plt.xlabel(f"{comp} (T)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 5. Field magnitude and log-scale distribution
B_mag = np.sqrt(df_xy['Bx']**2 + df_xy['By']**2 + df_xy['Bz']**2)

plt.figure()
plt.hist(B_mag, bins=100, alpha=0.8, color='darkorange')
plt.title("Magnetic Field Magnitude (T)")
plt.xlabel("|B| (T)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Log distribution check
B_mag_log = np.log1p(B_mag)

plt.figure()
plt.hist(B_mag_log, bins=100, alpha=0.8, color='purple')
plt.title("Log(1 + |B|) Distribution")
plt.xlabel("log(1 + |B|)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Heatmaps of each component
x = df_xy['x']
y = df_xy['y']
for comp in ['Bx', 'By', 'Bz']:
    plt.figure(figsize=(7, 5))
    plt.tricontourf(x, y, df_xy[comp], levels=100, cmap='coolwarm')
    plt.colorbar(label=f'{comp} (T)')
    plt.title(f"{comp} Field Heatmap")
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# 7. Pairwise distance histogram (use a sample for efficiency)
X = df_xy[['x', 'y']].values
if len(X) > 2000:
    X_sample = X[np.random.choice(len(X), 2000, replace=False)]
else:
    X_sample = X

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)
dists = pairwise_distances(X_scaled)
dists_flat = dists[np.triu_indices_from(dists, k=1)]

plt.figure()
plt.hist(dists_flat, bins=100, color='green', alpha=0.7)
plt.title("Pairwise Distance Distribution (standardized space)")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"\n💡 Suggested initial RBF lengthscale: ~{np.percentile(dists_flat, 20):.2f}")
