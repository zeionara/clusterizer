from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering

def clusterize_kmeans(numberized, number_of_clusters, iterations):
	model = KMeans(n_clusters = number_of_clusters, init = 'k-means++', max_iter = iterations, n_init = 1)
	model.fit(numberized)
	return model

def clusterize_affinity_propagation(numberized, iterations):
	model = AffinityPropagation(max_iter = iterations)
	model.fit(numberized)
	return model

def clusterize_spectrally(numberized, number_of_clusters):
	model = SpectralClustering(n_clusters = number_of_clusters, assign_labels = "discretize")
	model.fit(numberized)
	return model