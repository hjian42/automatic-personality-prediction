# variables NOT for optimizing
FILENAME = 'data.csv'
MAX_NUM_WORDS = 10000
MAX_SEQ_LENGTH = 500
PATH_EMBEDDING = '../../fast-text/corpus.friends+nyt+wiki+amazon.fasttext.skip.d100.vec'
class Params:

	# fairly fixed variables 
	# FILENAME = 'data.csv'
	# MAX_NUM_WORDS = 10000
	# MAX_SEQ_LENGTH = 500

	# variables for optimization
	n_epoch = 2
	batch_size = 64
	embeddings_dim = 100
	dropout_rate = 0.5
	dense_layer_size = 256
	use_word_embedding = False
	train_embedding = False

	# TODO: Conv parameters
	filter_size = [3,4,5]
	num_filter = [32,32,32]
	pool_size = [2,2,2]



	def setParams(self, dct):
		if 'embeddings_dim' in dct:
			self.embeddings_dim = dct['embeddings_dim']
		if 'dense_layer_size' in dct:
			self.dense_layer_size = dct['dense_layer_size']

