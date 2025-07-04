import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

# 1. Load full CSV
df = pd.read_csv('arena1.csv')

# 2. Rename columns for simplicity
df = df.rename(columns={
    'mf.normB (T)': 'B',
    'mf.Bx (T)': 'Bx',
    'mf.By (T)': 'By',
    'mf.Bz (T)': 'Bz'
})

# 3. Filter: only keep rows where z ≈ 0
tolerance = 7+1e-3
df_xy = df[np.abs(df['z']) < tolerance]

# 4. Extract x, y, and B
x = df_xy['x']
y = df_xy['y']
B = df_xy['B']

# 5. Set small threshold to avoid log(0)
B[B < 1e-12] = 1e-12  # Prevent log(0)

# 6. Plot using log color scale
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    x, y, c=B,
    cmap='plasma',
    norm=colors.LogNorm(vmin=B.min(), vmax=B.max()),
    s=5, marker='s'
)
plt.colorbar(sc, label='Magnetic Field Magnitude (T, log scale)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Magnetic Field Magnitude in XY Plane (Log Scale, z ≈ 0)')
plt.axis('equal')
plt.tight_layout()
plt.show()
