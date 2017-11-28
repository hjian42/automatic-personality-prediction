# variables NOT for optimizing
FILENAME = 'data.csv'
MAX_NUM_WORDS = 10000 # vocab_size
MAX_SEQ_LENGTH = 2462
PATH_EMBEDDING = '../../fast-text/corpus.friends+nyt+wiki+amazon.fasttext.skip.d100.vec'
class Params:

	# fairly fixed variables 
	# FILENAME = 'data.csv'
	# MAX_NUM_WORDS = 10000
	# MAX_SEQ_LENGTH = 500

	# variables for optimization
	n_epoch = 30
	batch_size = 64
	embeddings_dim = 100
	dropout_rate = 0.5
	dense_layer_size = 128
	use_word_embedding = False
	train_embedding = False
	use_merge = True


	# TODO: Conv parameters
	filter_size = [2,3,4]
	num_filter = [3,3,3]
	pool_size = [3,3,3]


	def setParams(self, dct):
		if 'embeddings_dim' in dct:
			self.embeddings_dim = dct['embeddings_dim']
		if 'dense_layer_size' in dct:
			self.dense_layer_size = dct['dense_layer_size']


