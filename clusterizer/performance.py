import time
from clusterizer.loggers import log

def measure(original_size = -1):
	def measure_performance(inspected_function):
		def inspect_time(*args, **kwargs):
			start = time.time()
			count = inspected_function(*args, **kwargs)
			end = time.time()
			log(f'Execution took {end - start:.2f} seconds ' + 
				(f'({((end - start) * original_size / count) / 3600 / 24:.2f} days to handle full dataset)' if (original_size > 0) and (count > 0) else ''), True)
		return inspect_time
	return measure_performance