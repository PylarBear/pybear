import numpy as n, pandas as p, matplotlib as m
# import matplotlib.colors as mc
# import seaborn as sb
from data_validation import validate_user_input as vui

#STUFF TO CREATE DUMMY DATA ############################################################################################
dimensions = ['x1','x2']

for dim_idx in range(len(dimensions)):
    dimensions[dim_idx] = [round(x,4) for x in
                        n.linspace(
                        vui.validate_user_float(f'Enter lower bound for {dimensions[dim_idx]} > '),
                        vui.validate_user_float(f'Enter upper bound for {dimensions[dim_idx]} > '),
                        vui.validate_user_int(f'Enter number of partitions for {dimensions[dim_idx]} > ')+1,
                        )
    ]

#END STUFF TO CREATE DUMMY DATA ########################################################################################





GRID = []
for x1 in dimensions[0]:
    GRID.append([])
    for x2 in dimensions[1]:
        GRID[-1].append([x1,x2])

print(f'COORD GRID[:5][:5]:')
[print(GRID[x][:5]) for x in range(len(GRID[:5]))]


mu = [0,0]
sigma = 1
SIGMA = [[sigma * ( _ == dim) for _ in range(len(dimensions))] for dim in range(len(dimensions))]

sign_, determ_ = n.linalg.slogdet(SIGMA)
gauss_coeff = 1 / ((2 * n.pi)**(0.5 * len(dimensions)) * n.sqrt(sign_ * n.exp(determ_)))


GAUSS_GRID = GRID.copy()
for row_idx in range(len(GAUSS_GRID)):
    for col_idx in range(len(GAUSS_GRID[row_idx])):
        coordinate = GAUSS_GRID[row_idx][col_idx]
        x_minus_mu = n.subtract(coordinate,mu)
        exp_term = -.5 * n.matmul(x_minus_mu, n.matmul(n.linalg.inv(SIGMA), n.transpose(x_minus_mu), dtype=float), dtype=float)
        GAUSS_GRID[row_idx][col_idx] = gauss_coeff * n.exp(exp_term)


print(f'GAUSS GRID:')
[print([round(x,5) for x in X]) for X in GRID]


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()







