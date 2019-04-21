import os, random, re
import numpy as np
from clusterizer.loggers import log, log_percents

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

def append_file(filename, documents):
	with open(filename, 'a+', encoding = 'utf-8') as file:
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

def get_number_of_lines(filename):
	with open(filename, 'r') as file:
		for i, _ in enumerate(file):
			pass
	return i + 1

def get_random_lines(filename, number_of_lines):
	whole_number_of_lines = get_number_of_lines(filename)
	if number_of_lines > whole_number_of_lines:
		raise ValueError('Incorrect number of lines to retrieve')
	indexes_of_selected_lines = random.sample(range(0, whole_number_of_lines), number_of_lines)

	index_of_line = 0
	lines_to_return = []
	with open(filename, 'r') as file:
		while True:
			line = file.readline()
			if not line:
				break
			if index_of_line in indexes_of_selected_lines:
				lines_to_return.append(line)
			index_of_line += 1
	
	return lines_to_return

def get_files(dir_name):
	return os.listdir(dir_name)

def clear_file(filename):
	write_file(filename, [])

#
# filtering
#

def read_keywords(filename):
	return [row.strip() for row in read_file(filename)]

def is_there_keywords(text, keyword_matchers):
	for keywords_matcher in keyword_matchers:
		if keywords_matcher.search(text):
			return True
	return False

def start_iteration(counter, abs_step, whole_size, number_of_articles, verbose):
	counter += 1
	#print(counter)
	if (whole_size > 0) and (counter % abs_step == 0):
		log_percents(counter / whole_size * 100, verbose)
	continue_handling = True
	if (counter >= number_of_articles) and (number_of_articles > 0):
		continue_handling = False
	return counter, continue_handling

def init_handling(whole_size, log_step):
	return 0, int(whole_size * log_step / 100)

def append_if_relevant(relevant_articles, article, keywords_matchers):
	if is_there_keywords(article, keywords_matchers):
		relevant_articles.append(article.replace('\n', '') + '\n')

def make_keyword_matchers(keywords_filename):
	return [re.compile(r'\s'+keyword+r'\s') for keyword in read_keywords(keywords_filename)]

def extract(input_filename, output_filename, make_reader, get_article, whole_size, log_step, number_of_articles, verbose, keywords_filename):
	keywords_matchers = make_keyword_matchers(keywords_filename)
	relevant_articles = []
	with open(input_filename, 'r', encoding = 'utf-8', errors = 'replace') as input_file:
		counter, abs_step = init_handling(whole_size, log_step)
		for row in make_reader(input_file):
			#print('ok')
			counter, continue_handling = start_iteration(counter, abs_step, whole_size, number_of_articles, verbose)
			#print(continue_handling)
			if not continue_handling:
				break

			article = get_article(row)
			if not article:
				break

			append_if_relevant(relevant_articles, article, keywords_matchers)

	append_file(output_filename, relevant_articles)
	return counter
