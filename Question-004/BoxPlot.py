
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.randn(100)

sns.boxplot(data=data)
plt.title('Box plot')
plt.show()