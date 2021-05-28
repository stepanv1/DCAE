
import matplotlib
from matplotlib import cm
import numpy as np

magma_cmap = matplotlib.cm.get_cmap('magma')
viridis_cmap = matplotlib.cm.get_cmap('viridis')

n=30

viridis_rgb = []
magma_rgb = []

norm = matplotlib.colors.Normalize(vmin=0, vmax=n)

for i in range(0, n):
    k = matplotlib.colors.colorConverter.to_rgb(magma_cmap(norm(i)))
    magma_rgb.append(k)

for i in range(0, n):
    k = matplotlib.colors.colorConverter.to_rgb(viridis_cmap(norm(i)))
    viridis_rgb.append(k)



def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k * h)[:3]) * n)
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


magma = matplotlib_to_plotly(magma_cmap, n)
viridis = matplotlib_to_plotly(viridis_cmap, n)
