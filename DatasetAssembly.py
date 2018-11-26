import numpy as np
import random
import pdb
#from Bio import SeqIO
from sklearn import preprocessing
import urllib3


window_size=11
win_it_size = 50
inOneHotVectSize=22 #
outOneHotVectSize=4

http        = urllib3.PoolManager()
response    = http.request('GET', "http://calla.rnet.missouri.edu/cheng_courses/mlbioinfo/ss_train.txt")
data        = response.data.decode('utf-8')
splitByLine = data.split('\n')


response2   = http.request('GET', "http://calla.rnet.missouri.edu/cheng_courses/mlbioinfo/ss_test.txt")
dataTest    = response.data.decode('utf-8')
splitByLine2 = dataTest.split('\n')

NUM_TRAIN_SEQ = 1180
NUM_TEST_SEQ  = 126

#preprocessing
#label Encode

outLab        = ['C','H','E','z']   #z is edge
outLabEncoder = preprocessing.LabelEncoder()
outLabEncoder.fit(outLab)

inLab        = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','X','Y','V','z'] #z is edge
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
	arrayOfCharsOutTest = list(splitByLine2[(i*4)+3])
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

encodingForInBlank  = np.zeros((1,inOneHotVectSize))
encodingForOutBlank = np.zeros((1, outOneHotVectSize))
encodingForInBlank[0][inOneHotVectSize-1]   = 1
encodingForOutBlank[0][outOneHotVectSize-1] = 1

#Assemble Training Set
for i in range(0, NUM_TRAIN_SEQ):
	xPiece = inOneHot.transform(np.array(encTrainArray[i][0]).reshape(-1,1)).toarray()
	yPiece = outOneHot.transform(np.array(encTrainArray[i][1]).reshape(-1,1)).toarray()
	print(xPiece.shape)
	start = random.randint(0, window_size-1)
	for jjj in range(-1*start, xPiece.shape[0], win_it_size):
		print("i",i,"j",jjj)
		#Left edges
		if jjj<0:
			smallXPiece = xPiece[0:window_size+jjj]
			smallYPiece = yPiece[0:window_size+jjj]
			borderNeeded = abs(jjj)
			for k in range(0, borderNeeded):
				smallXPiece = np.vstack((encodingForInBlank, smallXPiece))
				smallYPiece = np.vstack((encodingForOutBlank, smallYPiece))
			x_train = np.vstack((x_train, np.expand_dims(smallXPiece, 0)))
			y_train = np.vstack((y_train, np.expand_dims(smallYPiece, 0)))

		#Right edges
		elif jjj>xPiece.shape[0]-window_size:
			smallXPiece   = xPiece[jjj:]
			smallYPiece   = yPiece[jjj:]
			borderNeeded  = window_size - smallXPiece.shape[0]
			for k in range(0, borderNeeded):
				smallXPiece   = np.vstack((smallXPiece, encodingForInBlank))
				smallYPiece   = np.vstack((smallYPiece, encodingForOutBlank))

			x_train = np.vstack((x_train, np.expand_dims(smallXPiece, 0)))
			y_train = np.vstack((y_train, np.expand_dims(smallYPiece, 0)))
		else:
			smallXPiece = xPiece[jjj:jjj+window_size]
			smallYPiece = yPiece[jjj:jjj+window_size]
			x_train = np.vstack((x_train, np.expand_dims(smallXPiece, 0)))
			y_train = np.vstack((y_train, np.expand_dims(smallYPiece, 0)))


#Assemble Testing Set

for i in range(0, NUM_TEST_SEQ):
	xPiece = inOneHot.transform(np.array(encTestArray[i][0]).reshape(-1,1)).toarray()
	yPiece = outOneHot.transform(np.array(encTestArray[i][1]).reshape(-1,1)).toarray()
	print(xPiece.shape)
	for jjj in range(-1*int(window_size/2), xPiece.shape[0]+int(window_size/2)):
		print("i",i,"j",jjj)
		#Left edges
		if jjj<0:
			smallXPiece = xPiece[0:window_size+jjj]
			smallYPiece = yPiece[0:window_size+jjj]
			borderNeeded = abs(jjj)
			for k in range(0, borderNeeded):
				smallXPiece = np.vstack((encodingForInBlank, smallXPiece))
				smallYPiece = np.vstack((encodingForOutBlank, smallYPiece))
			x_test = np.vstack((x_test, np.expand_dims(smallXPiece, 0)))
			y_test = np.vstack((y_test, np.expand_dims(smallYPiece, 0)))

		#Right edges
		elif jjj>xPiece.shape[0]-window_size:
			smallXPiece   = xPiece[jjj:]
			smallYPiece   = yPiece[jjj:]
			borderNeeded  = window_size - smallXPiece.shape[0]
			for k in range(0, borderNeeded):
				smallXPiece   = np.vstack((smallXPiece, encodingForInBlank))
				smallYPiece   = np.vstack((smallYPiece, encodingForOutBlank))

			x_test = np.vstack((x_test, np.expand_dims(smallXPiece, 0)))
			y_test = np.vstack((y_test, np.expand_dims(smallYPiece, 0)))
		else:
			smallXPiece = xPiece[jjj:jjj+window_size]
			smallYPiece = yPiece[jjj:jjj+window_size]
			x_test = np.vstack((x_test, np.expand_dims(smallXPiece, 0)))
			y_test = np.vstack((y_test, np.expand_dims(smallYPiece, 0)))




#to eliminate the all 1 initialization row needed for shape preservation
x_train = np.delete(x_train, 0,0)
y_train = np.delete(y_train, 0,0)
x_test  = np.delete(x_test, 0,0)
y_test  = np.delete(y_test, 0,0)
#Alignments


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import *
#simple Keras model



##TODO TODO TODO
## This is where we can insert our model,  it currently takes as input a onehot encoded vector of inputs restricted to a certain window size and outputs predictions for the 

#TASKS:
## NN architecture
## HyperParameter Tuning
## Plumming for splitting inputs during evalutaion
## Dealing with edge cases
## Incorporation of Homologous Sequences 
##TODO TODO TODO

##Can run now

#model = Sequential([Dense(outOneHotVectSize, input_shape=(window_size, inOneHotVectSize,)), Activation('relu')])

model=Sequential()
#model.add(Dense(outOneHotVectSize, input_shape=(window_size, inOneHotVectSize,)), Activation('relu')])
# model.add(Conv1D(filters, kernel_size, strides=1, padding='valid', input_shape=(window_size, inOneHotVectSize), activation="relu"))
# model.add(MaxPool1D(pool_size=3, strides=1, padding="valid"))
# model.add(Flatten())
# model.add(Dense(21, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model.add(Conv1D(128, 11, strides=1, padding='same', input_shape=(window_size, inOneHotVectSize), activation="relu"))
model.add(Dropout(0.4))
model.add(Conv1D(64, 11, strides=1, padding='same', activation="relu"))
model.add(Dropout(0.4))
model.add(Conv1D(4, 11, strides=1, padding='same', activation="softmax"))

print(model.summary())

model.compile(loss='mse',
		optimizer='sgd',
		metrics=['accuracy'])
#x,y are numpy array
model.fit(x_train, y_train, epochs=10, batch_size=1)

#Evaluate model
test_prediction = model.predict(x_test, batch_size=32)
numCorrect = 0
numWrong   = 0
for ii,jj in zip(test_prediction, y_train):
	a = np.argmax(ii[int(window_size/2)])
	b = np.argmax(jj[int(window_size/2)])
	print(a,b)
	if(a==b):
		numCorrect = numCorrect +1
	else:
		numWrong  = numWrong+1
	


print(numWrong)
print(numCorrect)
pdb.set_trace()


