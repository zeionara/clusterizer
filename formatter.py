from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

import numpy as np
import pandas as pd
import re

import time
import os

from lemmatizer import log

RESULTS_FOLDER = '/mnt/c/Users/prote/big_data/results/test_resultsss_600'
OUTPUT_FILE = '/mnt/c/Users/prote/big_data/final.txt'
# helping functions

def read_file(filename):
	with open(filename, 'r', encoding = 'utf-8') as file:
		return file.readlines()

def write_file(filename, documents):
	with open(filename, 'w', encoding = 'utf-8') as file:
		file.writelines(documents)

def get_files(dir_name):
	return os.listdir(dir_name)

def numberize_tfidf(filename):
	vectorizer = TfidfVectorizer()
	file_content = read_file(filename)
	return vectorizer, vectorizer.fit_transform(file_content), file_content

def get_terms_with_tfidf_higher_than_mean(vectorizer, numberized):
	tf_idf = pd.DataFrame(numberized.toarray(), columns = vectorizer.get_feature_names())
	accumulated_tf_idf = tf_idf.sum().sort_values(ascending = False)
	mean_accumulated_tf_idf = accumulated_tf_idf.mean()
	filtered_accumulated_tf_idf = accumulated_tf_idf[accumulated_tf_idf.apply(lambda term: term > mean_accumulated_tf_idf)]
	return list(filtered_accumulated_tf_idf.index)

def get_terms_with_top_tfidf(vectorizer, numberized, count):
	tf_idf = pd.DataFrame(numberized.toarray(), columns = vectorizer.get_feature_names())
	accumulated_tf_idf = tf_idf.sum().sort_values(ascending = False)
	top_terms = accumulated_tf_idf.head(count)
	return list(top_terms.index), list(top_terms)

def get_cluster_description(path):
	vectorizer, numberized, file_content = numberize_tfidf(path)
	return ' '.join([f'{pair[0]}({pair[1]})' for pair in zip(*get_terms_with_top_tfidf(vectorizer, numberized, 5))]), len(file_content)

def main():
	variants = get_files(RESULTS_FOLDER)
	variants_descriptions = []
	variants_number_of_documents = []
	variants_names = ['cluster_index']
	max_len = 0
	for variant in variants:
		log(f'analyzing variant {variant}...')
		variant_clusters_description = []
		variant_number_of_documents = []
		variant_path = f'{RESULTS_FOLDER}/{variant}'
		variants_names.append(variant)
		clusters = get_files(variant_path)
		counter = 0
		for cluster in clusters:
			log(f'analyzing cluster {counter}...')
			cluster_path = f'{variant_path}/{cluster}'
			description, number_of_documents = get_cluster_description(cluster_path)
			variant_clusters_description.append(description)
			variant_number_of_documents.append(number_of_documents)
			counter += 1
		if len(variant_clusters_description) > max_len:
			max_len = len(variant_clusters_description)
		variants_descriptions.append(variant_clusters_description)
		variants_number_of_documents.append(variant_number_of_documents)

	lines_to_write = []
	lines_to_write.append('\t\t'.join(variants_names) + '\n')
	#print(variants_number_of_documents)
	for i in range(max_len):
		lines_to_write.append('\t'.join([str(i + 1), ] + [variants_descriptions[j][i] + '\t' + str(variants_number_of_documents[j][i]) if i < len(variants_descriptions[j]) else '-\t-' for j in range(len(variants_descriptions))]) + '\n')

	write_file(OUTPUT_FILE, lines_to_write)

	

if __name__ == "__main__":
	main()