import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# === Load and rename CSV columns ===
df = pd.read_csv("code files/arena1.csv")  # Update path as needed

df = df.rename(columns={
    'mf.normB (T)': 'b_norm',
    'mf.Bx (T)': 'bx',
    'mf.By (T)': 'by',
    'mf.Bz (T)': 'bz'
})

# === Shift centered coordinates to positive frame ===
x = df['x'].values + 914       # [-914, +914] → [0, 1828]
y = df['y'].values + 609.5     # [-609.5, +609.5] → [0, 1219]

bx = df['bx'].values
by = df['by'].values
bz = df['bz'].values
b_norm = df['b_norm'].values

# === Create uniform 2 mm grid ===
x_grid = np.arange(0, 1828 + 2, 2)
y_grid = np.arange(0, 1219 + 2, 2)
X, Y = np.meshgrid(x_grid, y_grid)

# === Interpolate to grid ===
points = np.column_stack((x, y))

bx_grid = griddata(points, bx, (X, Y), method='linear', fill_value=np.nan)
by_grid = griddata(points, by, (X, Y), method='linear', fill_value=np.nan)
bz_grid = griddata(points, bz, (X, Y), method='linear', fill_value=np.nan)
b_norm_grid = griddata(points, b_norm, (X, Y), method='linear', fill_value=np.nan)

# === Remove negligible values (set to NaN if below threshold) ===
threshold = 1e-8  # Tesla
bx_grid[np.abs(bx_grid) < threshold] = np.nan
by_grid[np.abs(by_grid) < threshold] = np.nan
bz_grid[np.abs(bz_grid) < threshold] = np.nan
b_norm_grid[b_norm_grid < threshold] = np.nan

# === Recalculate magnitude from components ===
bmag_grid = np.sqrt(bx_grid**2 + by_grid**2 + bz_grid**2)

# === Save interpolated grids ===
np.savez("magnetic_field_grid_2mm_latest.npz",
         x_grid=x_grid,
         y_grid=y_grid,
         bx=bx_grid,
         by=by_grid,
         bz=bz_grid,
         bnorm=b_norm_grid,
         bmag=bmag_grid)

# === Visualize magnetic field magnitude ===
plt.imshow(bmag_grid, origin='lower', extent=[0, 1828, 0, 1219])
plt.title("Interpolated Magnetic Field Magnitude |B|")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar(label="|B| (T)")
plt.show()

# === Visualize missing regions ===
plt.imshow(np.isnan(bx_grid), origin='lower', extent=[0, 1828, 0, 1219])
plt.title("Missing Data Regions (Bx < threshold or no data)")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar(label="1 = Missing")
plt.show()
