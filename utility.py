import os
import subprocess
import pdb
import numpy as np

def getBlastProb(sequenceFile, outputFile):
	pdb.set_trace()
	arrayOfPSSM = np.ones((1,20))
	subprocess.call("../DeepQA/tools/pspro2/bin/generate_flatblast.sh "+sequenceFile+" "+outputFile, shell=True)
	f = open(outputFile+".pssm",'r')
	lines = f.readlines()
	for lineIndex in range(3, len(lines)-7):
		arline = lines[lineIndex].split(" ")
		arline = list(filter(lambda a:a!='', arline))
		arline = list(filter(lambda a:a!='\n', arline))
		arline = arline[2:]
		arline = np.array(arline)
		arrayOfPSSM = np.vstack((arrayOfPSSM, arline.reshape(1,20)))
	pdb.set_trace()
	return arrayOfPSSM
	print("dpd")
		
