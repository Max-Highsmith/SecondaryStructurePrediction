import numpy as np
import pdb
from Bio import SeqIO
from sklearn import preprocessing
import urllib3

http     = urllib3.PoolManager()
response = http.request('GET', "http://calla.rnet.missouri.edu/cheng_courses/mlbioinfo/ss_train.txt")
data     = response.data.decode('utf-8')
splitByLine = data.split('\n')


NUM_TRAIN_SEQ = 1180
NUM_TEST_SEQ  = 126

#preprocessing
#label Encode

outLab        = ['C','H','E']
outLabEncoder = preprocessing.LabelEncoder()
outLabEncoder.fit(outLab)

inLab        = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','X','Y','V'] 
inLabEncoder = preprocessing.LabelEncoder()
inLabEncoder.fit(inLab) 

encTrainArray = []
for i in range(0, NUM_TRAIN_SEQ):
	arrayOfCharsIn  = list(splitByLine[(i*4)+2])
	arrayOfCharsOut = list(splitByLine[(i*4)+3])
	encodedIn = inLabEncoder.transform(arrayOfCharsIn)
	encodedOut = outLabEncoder.transform(arrayOfCharsOut)
	encTrainArray.append([encodedIn, encodedOut])

#OneHot
inLab    = np.array(inLab)
outLab   = np.array(outLab)
inLab    = inLabEncoder.transform(inLab)
outLab   = outLabEncoder.transform(outLab)

inOneHot  = preprocessing.OneHotEncoder()
inOneHot.fit(inLab.reshape(-1,1))
outOneHot = preprocessing.OneHotEncoder()
outOneHot.fit(outLab.reshape(-1,1))

pdb.set_trace()
for i in range(0, NUM_TRAIN_SEQ):
	newb = inOneHot.transform(np.array(encTrainArray[i][0]).reshape(-1,1)).toarray()
	print(newb.shape)

