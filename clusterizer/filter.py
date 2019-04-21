import re, csv, time
import datetime, argparse

from clusterizer.loggers import log, log_percents
from clusterizer.performance import measure
from clusterizer.filesystem import read_file, write_file, clear_file, extract

def parse_args():
	parser = argparse.ArgumentParser(description = 'Filter out articles which do not contain given keywords')

	parser.add_argument('-in', '--input', help = 'input file contating lemmatized text for filtering', type = str, required = True, dest = 'input_filename')
	parser.add_argument('-out', '--output', help = 'output file for saving text after filtering', type = str, required = True, dest = 'output_filename')
	parser.add_argument('-k', '--keywords', help = 'file containing keywords for filtering', type = str, required = True, dest = 'keywords_filename')
	parser.add_argument('-n', '--number-of-documents', help = 'number of documents to analyze from the input file', 
						type = int, required = False, default = -1, dest = 'number_of_documents')
	parser.add_argument('-s', '--step', help = """step to output logging messages about the process (percents) - 
						make sense only if whole-size parameter also has been set up""", type = float, required = False, default = -1, dest = 'log_step')
	parser.add_argument('-w', '--whole-size', help = 'whole number of articles in the document - useful for generating logging messages',
						type = int, required = False, default = -1, dest = 'whole_size')
	parser.add_argument('-i', '--index', help = 'index of the line (starting from 0) contating text for lemmatization if input file has .csv format', 
						type = int, required = False, dest = 'index')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')
	parser.add_argument('-c', '--clear', help = 'clear output file before start', action = 'store_const', const = True, default = False, dest = 'clear')

	return parser.parse_args()

def main():
	args = parse_args()

	log('Starting extraction of the required documents...', args.verbose)
	
	if args.clear:
		clear_file(args.output_filename)
	
	is_csv = args.input_filename.split('.')[1] == 'csv'
	measure(args.whole_size)(extract)(args.input_filename, args.output_filename, 
										(lambda file: csv.reader(file)) if is_csv else (lambda file: file), 
										(lambda row: row[args.index]) if is_csv else (lambda row: row), 
			   							args.whole_size, args.log_step, args.number_of_documents, args.verbose, args.keywords_filename)
	
	log('Finished extraction of the required documents!', args.verbose)

if __name__ == '__main__':
	main()