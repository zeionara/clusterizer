from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from clusterizer.filesystem import read_file

def numberize_tfidf(filename, max_features = None):
	vectorizer = TfidfVectorizer(max_features = max_features)
	file_content = read_file(filename)
	return vectorizer, vectorizer.fit_transform(file_content), file_content

def numberize_hash(filename, number_of_features):
	vectorizer = HashingVectorizer(n_features = number_of_features)
	return vectorizer, vectorizer.fit_transform(read_file(filename))