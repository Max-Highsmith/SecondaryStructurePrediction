import numpy as np
import pdb
#from Bio import SeqIO
from sklearn import preprocessing
import urllib3


window_size=11
win_it_size = 50
inOneHotVectSize=21 #
outOneHotVectSize=3

http        = urllib3.PoolManager()
response    = http.request('GET', "http://calla.rnet.missouri.edu/cheng_courses/mlbioinfo/ss_train.txt")
data        = response.data.decode('utf-8')
splitByLine = data.split('\n')


response2   = http.request('GET', "http://calla.rnet.missouri.edu/cheng_course/mlbioinfo/ss_test.txt")
dataTest    = response.data.decode('utf-8')
splitByLine2 = data.split('\n')

NUM_TRAIN_SEQ = 1180
NUM_TEST_SEQ  = 126

#preprocessing
#label Encode

outLab        = ['C','H','E']   #z is edge
outLabEncoder = preprocessing.LabelEncoder()
outLabEncoder.fit(outLab)

inLab        = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','X','Y','V'] #z is edge
inLabEncoder = preprocessing.LabelEncoder()
inLabEncoder.fit(inLab) 

encTrainArray = []
for i in range(0, NUM_TRAIN_SEQ):
	arrayOfCharsIn  = list(splitByLine[(i*4)+2])
	arrayOfCharsOut = list(splitByLine[(i*4)+3])
	encodedIn = inLabEncoder.transform(arrayOfCharsIn)
	encodedOut = outLabEncoder.transform(arrayOfCharsOut)
	encTrainArray.append([encodedIn, encodedOut])


#Testing
encTestArray = []
for i in range(0, NUM_TEST_SEQ):
	arrayOfCharsInTest  = list(splitByLine2[(i*4)+2])
	arrayOfCharsOutTest = list(splitByLine2((i*4)+3])
	encodedInTest = inLabEncoder.transform(arrayOfCharsInTest)
	encodedOutTest = outLabEncoder.transform(arrayOfCharsOutTest)
	encTestArray.append([encodedInTest, encodedOutTest]) 

#OneHot
#traingin
inLab    = np.array(inLab)
outLab   = np.array(outLab)
inLab    = inLabEncoder.transform(inLab)
outLab   = outLabEncoder.transform(outLab)

inOneHot  = preprocessing.OneHotEncoder()
inOneHot.fit(inLab.reshape(-1,1))
outOneHot = preprocessing.OneHotEncoder()
outOneHot.fit(outLab.reshape(-1,1))

x_train = np.ones((1, window_size, inOneHotVectSize))
y_train = np.ones((1, window_size, outOneHotVectSize))

inEnds  = np.zeros(22)
outEnds = np.zeros(4)


x_test  = np.ones((1, window_size, inOneHotVectSize))
y_test  = np.ones((1, window_size, outOneHotVectSize))


for i in range(0, NUM_TRAIN_SEQ):
	xPiece = inOneHot.transform(np.array(encTrainArray[i][0]).reshape(-1,1)).toarray()
	yPiece = outOneHot.transform(np.array(encTrainArray[i][1]).reshape(-1,1)).toarray()
	print(xPiece.shape)
	for j in range(0, xPiece.shape[0], win_it_size):
		print("i",i,"j",j)
		if j>xPiece.shape[0]-window_size:
			pass		#TODO WE need to consider the edges  little z symbol 
					#Right now training set is only values in middle
		else:
			smallXPiece = xPiece[j:j+window_size]
			smallYPiece = yPiece[j:j+window_size]
			x_train = np.vstack((x_train, np.expand_dims(smallXPiece, 0)))
			y_train = np.vstack((y_train, np.expand_dims(smallYPiece, 0)))



#to eliminate the all 1 initialization row needed for shape preservation
x_train = np.delete(x_train, 0,0)
y_train = np.delete(y_train, 0,0)
#Alignments




from keras.models import Sequential
from keras.layers import Dense, Activation
#simple Keras model

pdb.set_trace()
#inOneHotVectSize=22 #


##TODO TODO TODO
## This is where we can insert our model,  it currently takes as input a onehot encoded vector of inputs restricted to a certain window size and outputs predictions for the 

#TASKS:
## NN architecture
## HyperParameter Tuning
## Plumming for splitting inputs during evalutaion
## Dealing with edge cases
## Incorporation of Homologous Sequences 
##TODO TODO TODO
model = Sequential([Dense(outOneHotVectSize, input_shape=(window_size, inOneHotVectSize,)), Activation('relu')])



model.compile(loss='mse',
		optimizer='sgd',
		metrics=['accuracy'])
#x,y are numpy array
model.fit(x_train, y_train, epochs=500, batch_size=32)

#Evaluate model
model.predict(x_test, batch_size=32)





