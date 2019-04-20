import csv, subprocess
from pymystem3 import Mystem
import time, datetime

CSV_INPUT = '/mnt/c/Users/prote/big_data/news_lenta.csv' #'C:/Users/prote/big_data/news_lenta.csv'
CSV_OUTPUT = '/mnt/c/Users/prote/big_data/news_lenta_lemmatized.csv' #'C://Users/prote/big_data/news_lenta_lemmatized.csv'

NUMBER_OF_DOCUMENTS_TO_EXTRACT = 800000
STEP_TO_LOG = 0.05 # percents
WHOLE_SIZE = 700000	# articles

def log(message, verbose = True):
	if verbose:
		print(f'[{datetime.datetime.now().time()}] ' + message)

def log_percents(percents):
	log(f'{percents:.2f}% completed')

def measure(original_size):
	def measure_performance(inspected_function):
		def inspect_time(*args, **kwargs):
			start = time.time()
			count = inspected_function(*args, **kwargs)
			end = time.time()
			print(f'Execution took {end - start:.2f} seconds ({((end - start) * original_size / count) / 3600 / 24:.2f} days to handle full dataset)')
		return inspect_time
	return measure_performance

def insert_lemmatized_text_into_row(row, lemmatized_text):
	return row[:1] + [lemmatized_text.replace('\n', '')] + row[2:]

@measure(WHOLE_SIZE)
def extract(inp, outp):
	m = Mystem()
	with open(outp, 'w', encoding='utf-8') as csvfile_out:
		with open(inp, "r", encoding="utf-8") as csvfile_in:
		    datareader = csv.reader(csvfile_in)
		    datawriter = csv.writer(csvfile_out)
		    abs_step = int(WHOLE_SIZE * STEP_TO_LOG / 100)
		    count = 0
		    for row in datareader:
		        if count == 0:
		        	datawriter.writerow(insert_lemmatized_text_into_row(row, 'text_lemmas'))
		        else:
		        	datawriter.writerow(insert_lemmatized_text_into_row(row, ''.join(m.lemmatize(row[1]))))
		        if count >= NUMBER_OF_DOCUMENTS_TO_EXTRACT:
		            return count
		        elif count % abs_step == 0:
		        	log_percents(count / WHOLE_SIZE * 100)
		        count += 1
		    return count

if __name__ == "__main__":
	log('Starting lemmatization...')
	extract(CSV_INPUT, CSV_OUTPUT)
	log('Finished lemmatization!')
