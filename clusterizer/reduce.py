import argparse, re
import pandas as pd

from clusterizer.filesystem import read_file, write_file
from clusterizer.numberizing import numberize_tfidf
from clusterizer.loggers import log

# helping functions

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

def get_terms_with_head_tfidf(vectorizer, numberized, count):
	tf_idf = pd.DataFrame(numberized.toarray(), columns = vectorizer.get_feature_names())
	accumulated_tf_idf = tf_idf.sum().sort_values(ascending = False)
	top_terms = accumulated_tf_idf.head(count)
	return list(top_terms.index), list(top_terms)

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

def parse_args():
	parser = argparse.ArgumentParser(description = 'Reduce vocabulary size for the given set of documents')

	parser.add_argument('-in', '--input_filename', help = 'input file containing documents', type = str, required = True, dest = 'input_filename')
	parser.add_argument('-out', '--output_filename', help = 'output file containing documents', type = str, required = True, dest = 'output_filename')
	parser.add_argument('-n', '--head', help = """preserve only N percents of terms with the highest tf-idf score;
	 					if not set, terms with tf-idf higher than mean will be preserveed""", type = int, required = False, dest = 'head')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')

	return parser.parse_args()

def main():
	args = parse_args()

	vectorizer, numberized, file_content = numberize_tfidf(args.input_filename)
	if args.head:
		reduced_vocabulary = get_terms_with_top_tfidf(vectorizer, numberized, args.head)
	else:
		reduced_vocabulary = get_terms_with_tfidf_higher_than_mean(vectorizer, numberized)
	log(f'Vocabulary has been reduced from {numberized.shape[1]} to {len(reduced_vocabulary)}', args.verbose)
	write_file(args.output_filename, apply_reduced_vocabulary(reduced_vocabulary, file_content))
	log(f'Result of reducing has been written to file', args.verbose)

if __name__ == "__main__":
	main()