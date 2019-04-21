import csv, subprocess
from pymystem3 import Mystem
import time, datetime

import argparse
from clusterizer.performance import measure
from clusterizer.loggers import log, log_percents

def insert_lemmatized_text_into_row(row, lemmatized_text, index):
	return row[:index] + [lemmatized_text.replace('\n', '')] + row[index + 1:]

def extract(input_filename, output_filename, number_of_documents, log_step, whole_size, index, verbose):
	m = Mystem()
	with open(output_filename, 'w', encoding='utf-8') as csvfile_out:
		with open(input_filename, "r", encoding="utf-8") as csvfile_in:
		    datareader = csv.reader(csvfile_in)
		    datawriter = csv.writer(csvfile_out)
		    abs_step = int(whole_size * log_step / 100)
		    count = 0
		    for row in datareader:
		        if count == 0:
		        	datawriter.writerow(insert_lemmatized_text_into_row(row, 'text_lemmas', index))
		        else:
		        	datawriter.writerow(insert_lemmatized_text_into_row(row, ''.join(m.lemmatize(row[index])), index))
		        if (number_of_documents > 0) and (count >= number_of_documents):
		            return count
		        elif (whole_size > 0) and (log_step > 0) and (abs_step > 0) and (count % abs_step == 0):
		        	log_percents(count / whole_size * 100, verbose)
		        count += 1
		    return count

def parse_args():
	parser = argparse.ArgumentParser(description = 'Lemmatize articles in csv file using mystem')

	parser.add_argument('-in', '--input', help = 'input file contating text without lemmatization', type = str, required = True, dest = 'input_filename')
	parser.add_argument('-out', '--output', help = 'output file for saving text after lemmatization', type = str, required = True, dest = 'output_filename')
	parser.add_argument('-n', '--number-of-documents', help = 'number of documents to extract from the input file', 
						type = int, required = False, default = -1, dest = 'number_of_documents')
	parser.add_argument('-s', '--step', help = """step to output logging messages about the process (percents) - 
						make sense only if whole_size parameter also has been set up""", type = float, required = False, default = -1, dest = 'log_step')
	parser.add_argument('-w', '--whole-size', help = 'whole number of articles in the document - useful for generating logging messages',
						type = int, required = False, default = -1, dest = 'whole_size')
	parser.add_argument('-i', '--index', help = 'index of the line (starting from 0) contating text for lemmatization', 
						type = int, required = True, dest = 'index')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')

	return parser.parse_args()

def main():
	args = parse_args()

	log('Starting lemmatization...', args.verbose)
	measure(args.whole_size)(extract)(args.input_filename, args.output_filename, 
									  args.number_of_documents, args.log_step, args.whole_size, args.index, args.verbose)
	log('Finished lemmatization!', args.verbose)

if __name__ == "__main__":
	main()
	
	
