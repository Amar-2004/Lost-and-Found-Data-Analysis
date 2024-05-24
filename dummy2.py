import seaborn as sns
data_pivot = data.pivot(index="month", columns="year", values="passengers")
plt.figure(figsize=(10, 8))