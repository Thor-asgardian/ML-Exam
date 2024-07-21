import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 12)

sns.heatmap(data, annot = True, cmap='viridis')
plt.title('Heat Map')
plt.show()