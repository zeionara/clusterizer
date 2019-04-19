import re, csv, time, nltk, string
from lemmatizer import log, log_percents
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

FILES_INPUT = ['/mnt/c/Users/prote/big_data/news_lenta_filtered.txt', ]#['/mnt/c/Users/prote/big_data/russian_news_filtered.txt', ]# 
FILE_OUTPUT = '/mnt/c/Users/prote/big_data/without_stop_words.txt'

NUMBER_OF_ARTICLES_TO_HANDLE = 37070
STEP_TO_LOG = 0.05

WHOLE_SIZE = 13900

def delete_stop_words(inp, outp):
	stop_words = set(stopwords.words('russian') + list(string.punctuation) + ['\'\''+'""'+'``'])

	with open(outp, 'a+', encoding='utf-8') as file_out:
		with open(inp, "r", encoding="utf-8", errors='replace') as file_in:
		    counter = 0
		    abs_step = int(WHOLE_SIZE * STEP_TO_LOG / 100)
		    while True:
		    	counter += 1
		    	if counter % abs_step == 0:
		    		log_percents(counter / WHOLE_SIZE * 100)
		    	if counter >= NUMBER_OF_ARTICLES_TO_HANDLE:
		    		break
		    	
		    	article = file_in.readline()
		    	if not article:
		    		break

		    	file_out.write(' '.join([w for w in word_tokenize(article) if not w in stop_words]).replace('``','').replace("''",'') + '\n')
		    return counter

def main():
	log('Starting removing stop words...')
	for file in FILES_INPUT:
		log('Switching to the next file...')
		delete_stop_words(file, FILE_OUTPUT)
	log('Finished removing stop words!')

if __name__ == '__main__':
	main()