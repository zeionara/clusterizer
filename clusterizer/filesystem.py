import os
import numpy as np
from loggers import log

def mkdir(dir_name, verbose):
	if not os.path.isdir(dir_name):
		try:
			os.mkdir(dir_name)
		except OSError:
			log(f'Error creating directory {dir_name}', verbose)
		else:
			log(f'Directory {dir_name} successfully created', verbose)

def read_file(filename):
	with open(filename, 'r', encoding = 'utf-8') as file:
		return file.readlines()

def write_file(filename, documents):
	with open(filename, 'w', encoding = 'utf-8') as file:
		file.writelines(documents)

def write_file_with_labels(filename, documents, labels):
	with open(filename, 'w', encoding = 'utf-8') as file:
		file.writelines(list(map(lambda label, document: str(label) + '==' + document, labels, documents)))

def write_files_with_labels(out_folder, out_filename, documents, labels, verbose):
	mkdir(out_folder, verbose)
	for i in range(0, np.amax(labels) + 1):
		documents_to_write = []
		for j in range(len(documents)):
			if labels[j] == i:
				documents_to_write.append(documents[j])
		write_file(f'{out_folder}/{out_filename}_{i}.txt', documents_to_write)

def output_results(results, filename):
	lines_to_write = []
	counter = 0
	for number_of_clusters in results:
		output_string = '\t'.join([str(results[number_of_clusters][method]) for method in results[number_of_clusters]]).replace('.',',')
		if counter <= 0:
			lines_to_write.append('\t'.join([method for method in results[number_of_clusters]]))
			counter += 1
		lines_to_write.append(output_string)
	write_file(filename, [line_to_write + '\n' for line_to_write in lines_to_write])

def output_general_results(results, filename):
	lines_to_write = {}
	extra_output_strings = []
	for vector_size in results:
		counter = 0
		for number_of_clusters in results[vector_size]:
			if (counter <= 0):
				extra_output_strings.append('\t'.join([f'{method}_{vector_size}d' for method in results[vector_size][number_of_clusters]]))
				counter += 1
			
			output_string = '\t'.join([str(results[vector_size][number_of_clusters][method]) for method in results[vector_size][number_of_clusters]]).replace('.',',')

			if not lines_to_write.get(number_of_clusters):
				lines_to_write[number_of_clusters] = [str(number_of_clusters), output_string]
			else:
				lines_to_write[number_of_clusters].append(output_string)

	write_file(filename, ['\t'.join(['number_of_clusters', ] + extra_output_strings) + '\n',] + 
			   ['\t'.join(lines_to_write[number_of_clusters]) + '\n' for number_of_clusters in lines_to_write])