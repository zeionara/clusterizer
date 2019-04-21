import argparse
from clusterizer.filesystem import get_random_lines, write_file
from clusterizer.loggers import log

def parse_args():
	parser = argparse.ArgumentParser(description = "Generate file containing random subset of lines from a larger file")

	parser.add_argument('-in', '--input', help = 'input file for loading lines', type = str, required = True, dest = 'input_filename')
	parser.add_argument('-out', '--output', help = 'output file for saving retrieved lines', type = str, required = True, dest = 'output_filename')
	parser.add_argument('-n', '--number-of-lines', help = 'number of lines to select from input file', type = int, required = True, dest = 'number_of_lines')
	parser.add_argument('-v', '--verbose', help = 'turn on logging messages', action = 'store_const', const = True, default = False, dest = 'verbose')

	return parser.parse_args()

def main():
	args = parse_args()

	write_file(args.output_filename, get_random_lines(args.input_filename, args.number_of_lines))
	log('Result of selecting random lines have been written', args.verbose)

if __name__ == "__main__":
	main()