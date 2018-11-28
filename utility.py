import os
import subprocess
import pdb
import numpy as np

def sequenceArrayToFasta(lines):
	numBases = len(lines)
	x = open("dummy.fasta",'w')
	x.write('>T999 2.5F, , '+ str(numBases)+ " residues\n")
	x.write(lines)
	return("dummy.fasta")
	

def getBlastProb(sequenceFile, outputFile):
	arrayOfPSSM = np.ones((1,20))
	subprocess.call("../DeepQA/tools/pspro2/bin/generate_flatblast.sh "+sequenceFile+" "+outputFile, shell=True)
	f = open(outputFile+".pssm",'r')
	lines = f.readlines()
	for lineIndex in range(3, len(lines)-6):
		arline = lines[lineIndex].split(" ")
		arline = list(filter(lambda a:a!='', arline))
		arline = list(filter(lambda a:a!='\n', arline))
		arline = arline[2:]
		arline = np.array(arline)
		arrayOfPSSM = np.vstack((arrayOfPSSM, arline.reshape(1,20)))
	arrayOfPSSM = np.delete(arrayOfPSSM, 0,0)
	return arrayOfPSSM
	print("dpd")
		
