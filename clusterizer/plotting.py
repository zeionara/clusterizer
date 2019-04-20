import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import cycle
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

def output_plot(filename, models, numberized, x_min = -0.1, x_max = 1.0, y_distance = 10, x_step = 0.2):
	if (x_min < -1) or (x_max < 1) or (x_min > 1) or (x_max > 1) or (x_min > x_max):
		raise ValueError('Incorrect bounds for plotting silhouette score')
	if (y_distance <= 0):
		raise ValueError('Incorrect y distance value')
	if (x_step < 0) or ((x_max - x_min) < x_step):
		raise ValueError('Incorrect x step value')

	fig, axs = plt.subplots(1, len(models))
	fig.set_size_inches(7 * len(models), 18)
	axs_cycle = cycle(axs)

	for model in models:
		ax1 = next(axs_cycle)
		number_of_clusters = model.number_of_clusters

		predicted_labels = model.instance.fit_predict(numberized)
		silhouette_avg = silhouette_score(numberized, predicted_labels)
		silhouette_sample_values = silhouette_samples(numberized, predicted_labels)

		ax1.set_xlim([x_min, x_max])
		ax1.set_ylim([0, numberized.shape[0] + (number_of_clusters + 1) * y_distance])

		y_lower = y_distance
		for j in range(number_of_clusters):
			ith_cluster_values = silhouette_sample_values[predicted_labels == j]
			ith_cluster_values.sort()

			ith_cluster_size = ith_cluster_values.shape[0]
			y_upper = y_lower + ith_cluster_size

			color = cm.nipy_spectral(float(j) / number_of_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values, facecolor=color, edgecolor=color, alpha=0.7)
			ax1.text(-0.05, y_lower + 0.5*ith_cluster_size, str(j))
			y_lower = y_upper + 10

		ax1.set_title(f'The silhouette plot for {model.name} clustering')
		ax1.set_xlabel('The silhouette coefficient values')
		ax1.set_ylabel('The index of cluster')

		ax1.axvline(x = silhouette_avg, color = 'red', linestyle = '--')
		ax1.set_yticks([])
		ax1.set_xticks(np.arange(x_min, x_max, x_step))

	plt.savefig(f'{filename}.jpeg', bbox_inches = 'tight')