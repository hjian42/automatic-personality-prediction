import fasttext as ft
from sklearn.model_selection import StratifiedKFold
import numpy
import pandas as pd
seed = 7
numpy.random.seed(seed)

# First download the dbpedia.train using https://github.com/facebookresearch/fastText/blob/master/classification-example.sh
# on test/ and move to the example directory
# current_dir = path.dirname(__file__)
input_file = 'train.ft.txt'
test_file = 'test.ft.txt'
output = '/tmp/classifier'

# set params
dim=10
lr=0.005
epoch=1
min_count=1
word_ngrams=3
bucket=2000000
thread=4
silent=1
label_prefix='__label__'
traits = ['cAGR_text','cCON_text', 'cEXT_text', 'cOPN_text', 'cNEU_text'][:1]

# load in data
df = pd.read_csv('data_fastText.csv')

# cross validation
def cross_validation(X,Y):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    kfolds = kfold.split(X, Y)
    return kfolds


# report the score 
for trait in traits:

	X = fd[trait].values.tolist()
	Y = [0 for i in range(len(X))]
	kfolds = cross_validation(X, Y)

	for train, test in kfolds:

		# form the train and test files
		with open('train.ft.txt','w') as out:
		    for line in X[train]:
		        out.write(line)
		        out.write('\n')
		with open('test.ft.txt','w') as out:
		    for line in X[test]:
		        out.write(line)
		        out.write('\n')

		# Train the classifier
		classifier = ft.supervised(input_file, output, dim=dim, lr=lr, epoch=epoch,
		    min_count=min_count, word_ngrams=word_ngrams, bucket=bucket,
		    thread=thread, silent=silent, label_prefix=label_prefix)

		# Test the classifier
		result = classifier.test(test_file)
		print 'P@1:', result.precision
		print 'R@1:', result.recall
		print 'Number of examples:', result.nexamples

# Predict some text
# (Example text is from dbpedia.train)
# texts = ['birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta \
#         high school in lakewood township new jersey . it was founded by rabbi \
#         shmuel zalmen stein in 2001 after his father rabbi chaim stein asked \
#         him to open a branch of telshe yeshiva in lakewood . as of the 2009-10 \
#         school year the school had an enrollment of 76 students and 6 . 6 \
#         classroom teachers ( on a fte basis ) for a student–teacher ratio of \
#         11 . 5 1 .']
# labels = classifier.predict(texts)
# print labels