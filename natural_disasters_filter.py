from lemmatizer import measure, log, log_percents
import re, csv, time
import datetime

KEYWORDS_FILE = 'natural_disasters_keywords.txt'

FILE_INPUT_RUSSIAN_NEWS = '/mnt/c/Users/prote/big_data/russian_news.txt'#'C:/Users/prote/big_data/russian_news.txt'
FILE_OUTPUT_RUSSIAN_NEWS = '/mnt/c/Users/prote/big_data/russian_news_filtered.txt'#'C:/Users/prote/big_data/russian_news_filtered.txt'

FILE_INPUT_LENTA = '/mnt/c/Users/prote/big_data/news_lenta_lemmatized.csv'
FILE_OUTPUT_LENTA = '/mnt/c/Users/prote/big_data/news_lenta_filtered.txt'

NUMBER_OF_ARTICLES_TO_HANDLE = 1600000
STEP_TO_LOG = 0.05
WHOLE_SIZE_RUSSIAN_NEWS = 1500000
WHOLE_SIZE_LENTA = 700000

def read_keywords(filename):
	with open(filename, 'r', encoding='utf-8') as file:
		return [row.strip() for row in file.readlines()]

def is_there_keywords(text, keyword_matchers):
	for keywords_matcher in keyword_matchers:
		if keywords_matcher.search(text):
			return True
	return False

@measure(WHOLE_SIZE_RUSSIAN_NEWS)
def extract_russian_news(inp, outp):
	keywords_matchers = [re.compile(r'\s'+keyword+r'\s') for keyword in read_keywords(KEYWORDS_FILE)]
	with open(outp, 'a+', encoding='utf-8') as file_out:
		with open(inp, "r", encoding="utf-8", errors='replace') as file_in:
		    counter = 0
		    abs_step = int(WHOLE_SIZE_RUSSIAN_NEWS * STEP_TO_LOG / 100)
		    while True:
		    	counter += 1
		    	if counter % abs_step == 0:
		    		log_percents(counter / WHOLE_SIZE_RUSSIAN_NEWS * 100)
		    	if counter >= NUMBER_OF_ARTICLES_TO_HANDLE:
		    		break
		    	
		    	article = file_in.readline()
		    	if not article:
		    		break
		    	
		    	is_relevant = is_there_keywords(article, keywords_matchers)
		    	if is_relevant:
		    		#print(counter)
		    		file_out.write(article)
		    return counter


@measure(WHOLE_SIZE_LENTA)
def extract_lenta(inp, outp):
	keywords_matchers = [re.compile(r'\s'+keyword+r'\s') for keyword in read_keywords(KEYWORDS_FILE)]
	with open(outp, 'a+', encoding='utf-8') as file_out:
		with open(inp, "r", encoding="utf-8") as csvfile_in:
		    datareader = csv.reader(csvfile_in)
		    counter = 0
		    abs_step = int(WHOLE_SIZE_LENTA * STEP_TO_LOG / 100)
		    for row in datareader:
		    	counter += 1
		    	if counter % abs_step == 0:
		    		log_percents(counter / WHOLE_SIZE_LENTA * 100)
		    	if counter >= NUMBER_OF_ARTICLES_TO_HANDLE:
		    		break
		    	
		    	article = row[1]
		    	if not article:
		    		break
		    	
		    	is_relevant = is_there_keywords(article, keywords_matchers)
		    	if is_relevant:
		    		#print(counter)
		    		file_out.write(article + '\n')
		    return counter

def main():
	log('Starting extraction of the required documents...')
	#extract_russian_news(FILE_INPUT_RUSSIAN_NEWS, FILE_OUTPUT_RUSSIAN_NEWS)
	#log('Switching to another file...')
	extract_lenta(FILE_INPUT_LENTA, FILE_OUTPUT_LENTA)
	log('Finished extraction of the required documents!')

if __name__ == '__main__':
	main()