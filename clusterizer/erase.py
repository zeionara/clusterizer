import re, csv, time, nltk, string, argparse

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from clusterizer.loggers import log, log_percents
from clusterizer.filesystem import read_file, write_file
from clusterizer.performance import measure

def delete_stop_words(input_filename, output_filename):
	stop_words = set(stopwords.words('russian') + ['это', 'который', 'некоторый', 'поэтому'] + list(string.punctuation) + ['\'\''+'""'+'``'])
	articles = read_file(input_filename)
	write_file(output_filename, [
		' '.join([
			w for w in word_tokenize(article) if not w in stop_words
		]).replace('``','').replace("''",'').replace('...', '').replace('.', ' ') + '\n'
		for article in articles
	])

def parse_args():
	parser = argparse.ArgumentParser(description = 'Remove stop words from each document in the given set')

	parser.add_argument('-in', '--input-file', help = 'name of the file contatining documents for removing stop words', 
						type = str, required = True, dest = 'input_filename')
	parser.add_argument('-out', '--output', help = 'output file for saving text after filtering', type = str, required = True, dest = 'output_filename')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')

	return parser.parse_args()

@measure()
def main():
	args = parse_args()

	log('Starting removing stop words...', args.verbose)
	delete_stop_words(args.input_filename, args.output_filename)
	log('Finished removing stop words!', args.verbose)

if __name__ == '__main__':
	main()