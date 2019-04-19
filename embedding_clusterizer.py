from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

from lemmatizer import log

from gensim.models.doc2vec import TaggedDocument, Doc2Vec

FILENAME = '/mnt/c/Users/prote/big_data/reduced_without_stop_words_1k.txt'
OUT_FOLDER = '/mnt/c/Users/prote/big_data/results'
OUT_FILE = 'cluster'
NUMBER_OF_CLUSTERS = 2
MAX_NUMBER_OF_CLUSTERS = 9
INITIAL_NUMBER_OF_CLUSTERS = NUMBER_OF_CLUSTERS
MAX_ITERATIONS = 100

VECTOR_SIZES = [100, 200, 300, 400, 500, 600]

embedding_results = {}
results = {}

# helping functions

def mkdir(dir_name):
	try:
		os.mkdir(dir_name)
	except OSError:
		print(f'Error creating directory {dir_name}')
	else:
		print(f'Directory {dir_name} successfully created')

def read_file(filename):
	result = []
	with open(filename, 'r', encoding = 'utf-8') as file:
		return file.readlines()

def write_file_with_labels(filename, documents, labels):
	print(labels)
	with open(filename, 'w', encoding = 'utf-8') as file:
		file.writelines(list(map(lambda label, document: str(label) + '==' + document, labels, documents)))

def write_files_with_labels(out_folder, out_filename, documents, labels):
	mkdir(out_folder)
	for i in range(np.amax(labels) + 1):
		documents_to_write = []
		for j in range(len(documents)):
			if labels[j] == i:
				documents_to_write.append(documents[j])
		with open(f'{out_folder}/{out_filename}.{i}', 'w', encoding = 'utf-8') as file:
			file.writelines(documents_to_write)

# turning texts into numbers

def numberize_tfidf(FILENAME):
	vectorizer = TfidfVectorizer()
	file_content = read_file(FILENAME)
	return vectorizer, vectorizer.fit_transform(file_content), file_content

def numberize_hash(FILENAME):
	vectorizer = HashingVectorizer(n_features=2**14)
	return vectorizer, vectorizer.fit_transform(read_file(FILENAME))

def docvecs_to_2d_array(docvecs):
	result = []
	try:
		for docvec in docvecs:
			result.append(docvec)
	except KeyError:
			pass
	return np.array(result)


def get_embeddings(filename, vector_size = 100, window = 10, min_count = 1, workers = 4):
	file_content = read_file(filename)
	documents = [TaggedDocument(document, [i]) for i, document in enumerate(file_content)]
	model = Doc2Vec(documents, vector_size = vector_size, window = window, min_count = min_count, workers = workers)
	model.delete_temporary_training_data(keep_doctags_vectors = True, keep_inference = True)
	docvecs_to_2d_array(model.docvecs)
	return model, docvecs_to_2d_array(model.docvecs), file_content

# clusterizing vectors

def clusterize_kmean(numberized):
	model = KMeans(n_clusters = NUMBER_OF_CLUSTERS, init = 'k-means++', max_iter = MAX_ITERATIONS, n_init = 1)
	model.fit(numberized)
	return model

def clusterize_affinity_propagation(numberized):
	model = AffinityPropagation(max_iter = MAX_ITERATIONS)
	model.fit(numberized)
	return model

def clusterize_spectrally(numberized):
	model = SpectralClustering(n_clusters = NUMBER_OF_CLUSTERS, assign_labels="discretize")
	model.fit(numberized)
	return model

# evaluating results

def print_keywords(model, vectorizer):
	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(NUMBER_OF_CLUSTERS):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind])

def update_results(models, numberized, file_content):
	global results

	#fig, axs = plt.subplots(1, len(models))
	#fig.set_size_inches(7 * len(models), 18)
	
	results[NUMBER_OF_CLUSTERS] = {}

	for i in range(len(models)):
		model = models[i][0]
		#ax1 = axs[i]
		number_of_clusters = models[i][2]

		predicted_labels = model.fit_predict(numberized)
		write_files_with_labels(f'{OUT_FOLDER}/{models[i][1]}_{NUMBER_OF_CLUSTERS}_clusters_{MAX_ITERATIONS}_iterations', OUT_FILE, file_content, predicted_labels)
		silhouette_avg = silhouette_score(numberized, predicted_labels)
		log(f'Avg slh for {models[i][1]} = {silhouette_avg}')

		
		results[NUMBER_OF_CLUSTERS][models[i][1]] = silhouette_avg;
		"""
		silhouette_sample_values = silhouette_samples(numberized, predicted_labels)

		ax1.set_xlim([-0.1, 1.0])
		ax1.set_ylim([0, numberized.shape[0] + (number_of_clusters + 1) * 10])

		y_lower = 10
		for j in range(number_of_clusters):
			ith_cluster_values = silhouette_sample_values[predicted_labels == j]
			ith_cluster_values.sort()

			ith_cluster_size = ith_cluster_values.shape[0]
			y_upper = y_lower + ith_cluster_size

			color = cm.nipy_spectral(float(j) / number_of_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values, facecolor=color, edgecolor=color, alpha=0.7)
			ax1.text(-0.05, y_lower + 0.5*ith_cluster_size, str(j))
			y_lower = y_upper + 10

		ax1.set_title(f'The silhouette plot for {models[i][1]} clustering')
		ax1.set_xlabel('The silhouette coefficient values')
		ax1.set_ylabel('The index of cluster')

		ax1.axvline(x=silhouette_avg, color='red', linestyle='--')

		ax1.set_yticks([])
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
		"""

	#plt.savefig(f'{NUMBER_OF_CLUSTERS}_{MAX_ITERATIONS}.jpeg')

def output_embedding_results(embedding_results, filename):
	path, extension = filename.split('.')
	for vector_size in embedding_results:
		output_results(embedding_results[vector_size], f'{path}_{vector_size}_dimension_embedding.{extension}')

def output_results(results, filename):
	with open(filename, 'w') as file:
		counter = 0
		for number_of_clusters in results:
			output_string = '\t'.join([str(results[number_of_clusters][method]) for method in results[number_of_clusters]])
			extra_output_string = '\t'.join([method for method in results[number_of_clusters]])
			"""
			for method in results[number_of_clusters]:
				extra_output_string += '\t' + str(method)
				output_string += '\t' + str(results[number_of_clusters][method])
			"""
			if counter <= 0:
				file.write(extra_output_string + '\n')
				counter += 1
			file.write(output_string.replace('.',',') + '\n')

# making everything work properly

def main():
	global NUMBER_OF_CLUSTERS
	global results

	for vector_size in VECTOR_SIZES:
		print(f'{vector_size} DIMENSION')
		results = {}
		NUMBER_OF_CLUSTERS = INITIAL_NUMBER_OF_CLUSTERS
		model, numberized, file_content = get_embeddings(FILENAME, vector_size = vector_size)
		#print(dir(model))

		#vectorizer, numberized, file_content = numberize_tfidf(FILENAME)
		model_affinity_propagation = clusterize_affinity_propagation(numberized)
		log("Finished building affinity propagation")
		while NUMBER_OF_CLUSTERS <= MAX_NUMBER_OF_CLUSTERS:
			model_kmean = clusterize_kmean(numberized)
			model_spectral = clusterize_spectrally(numberized)

			if INITIAL_NUMBER_OF_CLUSTERS == NUMBER_OF_CLUSTERS:
				update_results([
					(model_kmean, 'kmeans', NUMBER_OF_CLUSTERS), 
					(model_spectral, 'spectral', NUMBER_OF_CLUSTERS), 
					(model_affinity_propagation, 'affinity propagation', model_affinity_propagation.cluster_centers_.shape[0])], 
					numberized, file_content)
			else:
				update_results([
					(model_kmean, 'kmeans', NUMBER_OF_CLUSTERS), 
					(model_spectral, 'spectral', NUMBER_OF_CLUSTERS)], 
					numberized, file_content)
			log("Finished analyzing results")
			NUMBER_OF_CLUSTERS += 1
		embedding_results[vector_size] = results
		output_results(embedding_results[vector_size], f'results_{vector_size}_dimension_embedding.txt')
	#output_embedding_results(embedding_results, 'results.txt')

if __name__ == "__main__":
	main()