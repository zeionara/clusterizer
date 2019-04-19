from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import re

import time

from lemmatizer import log

FILENAME_IN = '/mnt/c/Users/prote/big_data/without_stop_words_20k.txt'
FILENAME_OUT = '/mnt/c/Users/prote/big_data/reduced_without_stop_words_20k.txt'

# helping functions

def read_file(filename):
	with open(filename, 'r', encoding = 'utf-8') as file:
		return file.readlines()

def write_file(filename, documents):
	with open(filename, 'w', encoding = 'utf-8') as file:
		file.writelines(documents)

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

def get_terms_with_top_tfidf(vectorizer, numberized, percent):
	tf_idf = pd.DataFrame(numberized.toarray(), columns = vectorizer.get_feature_names())
	accumulated_tf_idf = tf_idf.sum().sort_values(ascending = False)
	top_terms = accumulated_tf_idf.head(int(len(accumulated_tf_idf.index) * percent))
	return list(top_terms.index)

def apply_reduced_vocabulary(reduced_vocabulary, documents):
	new_documents = []
	for document in documents:
		new_document = []
		for word in re.findall('[а-я]+', document):
			if word in reduced_vocabulary:
				new_document.append(word)
		new_documents.append(' '.join(new_document) + '\n')
	return new_documents

# clusterizing vectors

def main():
	vectorizer, numberized, file_content = numberize_tfidf(FILENAME_IN)
	reduced_vocabulary = get_terms_with_tfidf_higher_than_mean(vectorizer, numberized);
	write_file(FILENAME_OUT, apply_reduced_vocabulary(reduced_vocabulary, file_content))

if __name__ == "__main__":
	main()