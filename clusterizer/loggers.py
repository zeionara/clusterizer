import datetime

def log(message, verbose = False):
	if verbose:
		print(f'[{datetime.datetime.now().time()}] ' + message)

def print_keywords(model, vectorizer):
	print("Top terms per cluster:")
	order_centroids = model.cluster_centers_.argsort()[:, ::-1]
	terms = vectorizer.get_feature_names()
	for i in range(NUMBER_OF_CLUSTERS):
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :10]:
			print(' %s' % terms[ind])

def log_finishing_builing_model(model, verbose):
	if verbose:
		log(f'Finished building {model.name} model', verbose)