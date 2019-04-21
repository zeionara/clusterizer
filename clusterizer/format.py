import argparse

from clusterizer.filesystem import read_file, write_file, get_files
from clusterizer.numberizing import numberize_tfidf
from clusterizer.loggers import log
from clusterizer.performance import measure
from clusterizer.reduce import get_terms_with_head_tfidf

# helping functions

def get_cluster_description(path, max_features):
	try:
		vectorizer, numberized, file_content = numberize_tfidf(path, max_features)
		return ' '.join([f'{pair[0]}({pair[1]})' for pair in zip(*get_terms_with_head_tfidf(vectorizer, numberized, max_features))]), len(file_content)
	except ValueError:
		return '-', 0

def stage_clusters_descriptions(results_folder, output_file, max_features, verbose):
	variants = get_files(results_folder)
	variants_descriptions = []
	variants_number_of_documents = []
	variants_names = ['cluster_index']
	max_len = 0
	for variant in variants:
		log(f'analyzing variant {variant}...', verbose)
		variant_clusters_description = []
		variant_number_of_documents = []
		variant_path = f'{results_folder}/{variant}'
		variants_names.append(variant)
		clusters = get_files(variant_path)
		counter = 0
		for cluster in clusters:
			#log(f'analyzing cluster {counter}...', verbose)
			cluster_path = f'{variant_path}/{cluster}'
			description, number_of_documents = get_cluster_description(cluster_path, max_features)
			variant_clusters_description.append(description)
			variant_number_of_documents.append(number_of_documents)
			counter += 1
		if len(variant_clusters_description) > max_len:
			max_len = len(variant_clusters_description)
		variants_descriptions.append(variant_clusters_description)
		variants_number_of_documents.append(variant_number_of_documents)

	lines_to_write = []
	lines_to_write.append(variants_names[0] + '\t' + '\t\t'.join([variant_name for variant_name in variants_names[1:]]) + '\n')
	#print(variants_number_of_documents)
	for i in range(max_len):
		lines_to_write.append(
			'\t'.join(
				[str(i + 1), ] + 
				[variants_descriptions[j][i] + '\t' + str(variants_number_of_documents[j][i]) 
					if i < len(variants_descriptions[j]) 
					else '-\t-' 
					for j in range(len(variants_descriptions))
				]
			) + '\n'
		)

	write_file(output_file, lines_to_write)

def parse_args():
	parser = argparse.ArgumentParser(description = 'Staging cluster descriptions after clusterizing')

	parser.add_argument('-in', '--input-folder', 
						help = 'name of folder, which contains files representing clusters organized into subdirectories according to parameters',
						type = str, required = True, dest = 'input_folder')
	parser.add_argument('-out', '--output-file', help = 'output filename to save report', type = str, required = True, dest = 'output_file')
	parser.add_argument('-n', '--number-of-keywords', help = 'number of keywords to extract from cluster for describing their content',
						type = int, required = True, dest = 'number_of_keywords')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')

	return parser.parse_args()

@measure()
def main():
	args = parse_args()
	stage_clusters_descriptions(args.input_folder, args.output_file, args.number_of_keywords, args.verbose)
	log('Report has been saved', args.verbose)

if __name__ == "__main__":
	main()