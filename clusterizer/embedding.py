from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from clusterizer.filesystem import read_file
import numpy as np


def docvecs_to_2d_array(docvecs):
	result = []
	try:
		for docvec in docvecs:
			result.append(docvec)
	except KeyError:
			pass
	return np.array(result)

def get_embeddings(filename, vector_size = 100, window = 10, min_count = 1, workers = 4):
	file_content = read_file(filename)
	documents = [TaggedDocument(document, [i]) for i, document in enumerate(file_content)]
	model = Doc2Vec(documents, vector_size = vector_size, window = window, min_count = min_count, workers = workers)
	model.delete_temporary_training_data(keep_doctags_vectors = True, keep_inference = True)
	docvecs_to_2d_array(model.docvecs)
	return model, docvecs_to_2d_array(model.docvecs), file_content