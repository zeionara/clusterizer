from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples, calinski_harabaz_score, davies_bouldin_score

import numpy as np
import os, argparse
from collections import namedtuple

from clusterizer.loggers import log, log_finishing_builing_model
from clusterizer.filesystem import mkdir, read_file, write_file, write_files_with_labels, output_results, output_general_results
from clusterizer.numberizing import numberize_tfidf, numberize_hash
from clusterizer.clusterizing import clusterize_kmeans, clusterize_affinity_propagation, clusterize_spectrally
from clusterizer.embedding import get_embeddings
from clusterizer.plotting import output_plot
from clusterizer.performance import measure

import time

Model = namedtuple('Model', ['instance', 'name', 'number_of_clusters'])

def parse_args():
	parser = argparse.ArgumentParser(description = "Text clusterizer using kmeans, spectral and affinity propagation clustering techniques")
	parser.add_argument('-min', '--min-number-of-clusters', help = 'minimal number of clusters for kmeans and spectral clustering', 
						type = int, required = True, dest = 'min_number_of_clusters')
	parser.add_argument('-max', '--max-number-of-clusters', help = 'maximal number of clusters for kmeans and spectral clustering', 
						type = int, required = True, dest = 'max_number_of_clusters')
	parser.add_argument('-i', '--iterations', help = 'maximal number of iterations to perform for training models',
						type = int, required = True, dest = 'iterations')
	parser.add_argument('-in', '--input-file', help = 'input file with one document per line', type = str, required = True, dest = 'input_filename')
	parser.add_argument('-ofo', '--output-folder', help = 'output folder name template for saving clusterized documents', 
						type = str, required = False, dest = 'output_foldername_template')
	parser.add_argument('-ofi', '--output-file', help = 'output filename template (without extension) for saving clusterized documents', 
						type = str, required = False, dest = 'output_filename_template')
	parser.add_argument('-gfo', '--graphics-folder', help = 'output folder name template for saving plots', 
						type = str, required = False, dest = 'graphics_foldername_template')
	parser.add_argument('-gfi', '--graphics-file', help = 'graphics filename template (without extension) for saving plots', 
						type = str, required = False, dest = 'graphics_filename_template')
	parser.add_argument('-rfi', '--results-file', help = 'results filename template (without extension) for saving average silhouette score', 
						type = str, required = True, dest = 'results_filename_template')
	parser.add_argument('-d', '--dimensions', help = 'list of embedding representation dimensions to evaluate', nargs = '+',
						type = int, required = True, dest = 'dimensions')
	parser.add_argument('-f', '--faster', help = 'exclude affinity propagation model from every summary except first', action = 'store_const',
						const = True, default = False, dest = 'faster')
	parser.add_argument('-e', '--embeddings', help = 'use document embeddings for clusterization', action = 'store_const',
						const = True, default = False, dest = 'embeddings')
	parser.add_argument('-s', '--singular', help = 'generate just one singular file with final results', action = 'store_const',
						const = True, default = False, dest = 'singular')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')

	parser.add_argument('-slh', '--silhouette', help = 'compute silhouette score as a metric of clustering quality', action = 'store_const',
						const = True, default = False, dest = 'silhouette')
	parser.add_argument('-chi', '--calinski-harabaz-index', help = 'compute calinski harabaz index as a metric of clustering quality', action = 'store_const',
						const = True, default = False, dest = 'calinski_harabaz_index')
	parser.add_argument('-dbs', '--davies-bouldin-score', help = 'compute davies bouldin score as a metric of clustering quality', action = 'store_const',
						const = True, default = False, dest = 'davies_bouldin_score')
	parser.add_argument('-hbr', '--hybrid', help = 'compute hybrid coefficient which is equal to silhouette_score * calinski-harabaz-index / davies-bouldin-score',
						action = 'store_const', const = True, default = False, dest = 'hybrid')

	return parser.parse_args()

def stage_intermediate_results(models, numberized, file_content, results, number_of_clusters, 
	verbose, max_iterations, output_folder, output_file, dimensions, graphics_folder, graphics_file,
	write_silhouette, write_calinski_harabaz_index, write_davies_bouldin_score, write_hybrid):
	results[number_of_clusters] = {}

	for model in models:
		predicted_labels = model.instance.fit_predict(numberized)
		if output_folder and output_file:
			mkdir(f'{output_folder}_{dimensions}', verbose)
			write_files_with_labels(f'{output_folder}_{dimensions}/{model.name}_{model.number_of_clusters}_clusters_{max_iterations}_iterations', 
									output_file, file_content, predicted_labels, verbose)
		
		if not results[number_of_clusters].get(model.name):
				results[number_of_clusters][model.name] = {}
		if write_silhouette:
			silhouette_avg = silhouette_score(numberized, predicted_labels)
			log(f'Average slh for {model.name} = {silhouette_avg}', verbose)
			results[number_of_clusters][model.name]['silhouette'] = silhouette_avg
		if write_calinski_harabaz_index:
			calinski_harabaz_index = calinski_harabaz_score(numberized, predicted_labels)
			log(f'Calinski-Harabaz index for {model.name} = {calinski_harabaz_index}', verbose)
			results[number_of_clusters][model.name]['calinski_harabaz_index'] = calinski_harabaz_index
		if write_davies_bouldin_score:
			davies_bouldin_score_value = davies_bouldin_score(numberized, predicted_labels)
			log(f'Davies Bouldin score for {model.name} = {davies_bouldin_score_value}', verbose)
			results[number_of_clusters][model.name]['davies_bouldin_score'] = davies_bouldin_score_value
		if write_hybrid:
			hybrid = silhouette_score(numberized, predicted_labels) * calinski_harabaz_score(numberized, predicted_labels) / davies_bouldin_score(numberized, predicted_labels)
			log(f'Hybrid score for {model.name} = {hybrid}', verbose)
			results[number_of_clusters][model.name]['hybrid'] = hybrid

	if graphics_folder and graphics_file:
		mkdir(f'{graphics_folder}_{dimensions}', verbose)
		output_plot(f'{graphics_folder}_{dimensions}/{graphics_file}_{number_of_clusters}_clusters', models, numberized)

@measure()
def main():
	args = parse_args()
	general_results = {}

	metrics = {
		'silhouette': args.silhouette, 
		'calinski_harabaz_index': args.calinski_harabaz_index,
		'davies_bouldin_score': args.davies_bouldin_score,
		'hybrid': args.hybrid
	}

	log('Starting...', args.verbose)
	for vector_size in args.dimensions if args.embeddings else [1, ]:
		results = {}
		number_of_clusters = args.min_number_of_clusters
		
		vectorizer, numberized, file_content = get_embeddings(args.input_filename, vector_size = vector_size)\
											   if args.embeddings else \
											   numberize_tfidf(args.input_filename)

		

		affinity_propagation_model_instance = clusterize_affinity_propagation(numberized, args.iterations)
		affinity_propagation_number_of_clusters = affinity_propagation_model_instance.cluster_centers_.shape[0]
		model_affinity_propagation = Model(affinity_propagation_model_instance, 
										   f'affinity_propagation_{affinity_propagation_number_of_clusters}', 
										   affinity_propagation_number_of_clusters)
		log_finishing_builing_model(model_affinity_propagation, args.verbose)
		
		while number_of_clusters <= args.max_number_of_clusters:
			start = time.time()
			model_kmeans = Model(clusterize_kmeans(numberized, number_of_clusters, args.iterations), 'kmeans', number_of_clusters)
			log_finishing_builing_model(model_kmeans, args.verbose)
			model_spectral = Model(clusterize_spectrally(numberized, number_of_clusters), 'spectral', number_of_clusters)
			log_finishing_builing_model(model_spectral, args.verbose)

			stage_intermediate_results((model_kmeans, model_spectral, model_affinity_propagation) 
						   if (args.min_number_of_clusters == number_of_clusters) or not args.faster else 
						   (model_kmeans, model_spectral), 
						   numberized, file_content, results, number_of_clusters, args.verbose, args.iterations, 
						   args.output_foldername_template, args.output_filename_template, vector_size if args.embeddings else numberized.shape[1],
						   args.graphics_foldername_template, args.graphics_filename_template,
						   args.silhouette, args.calinski_harabaz_index, args.davies_bouldin_score, args.hybrid)

			log(f'After {time.time() - start} seconds finished analyzing results for {number_of_clusters} clusters' + f' and {vector_size} dimensions' if args.embeddings else '', args.verbose)
			number_of_clusters += 1
		
		current_index = vector_size if args.embeddings else numberized.shape[1]
		general_results[current_index] = results
		
		if not args.singular:
			for metric in metrics:
				if metrics[metric]:
					output_results(general_results[current_index], f'{args.results_filename_template}_{current_index}.{metric}', metric)

	if args.singular:
		for metric in metrics:
				if metrics[metric]:
					output_general_results(general_results, f'{args.results_filename_template}.{metric}', metric)

if __name__ == "__main__":
	main()