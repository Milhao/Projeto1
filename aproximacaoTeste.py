import sys
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import LinearLayer
import numpy
inputFile = open("teste1.txt","r")
inputs = 14
outputs =1

PorcDivTest = float(sys.argv[1])
Ciclos = int(sys.argv[2])
Learning = float(sys.argv[3])
Momentum = float(sys.argv[4])
camada1 = int(sys.argv[5])
camada2 = int(sys.argv[6])


ds = SupervisedDataSet(inputs,outputs)
trainSet = SupervisedDataSet(inputs,outputs)
testSet = SupervisedDataSet(inputs,outputs)
MaximoInput = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
MaximoTarget = 0
for line in inputFile.readlines():
	lineaux = line.strip()
	if not lineaux.startswith("%") and len(lineaux)!=0:
	    data = [float(x) for x in lineaux.split() if x != '']
	    indata = tuple(data[:inputs])
	    jj = 0
	    while(jj<14) :
	    	if(MaximoInput[jj]<indata[jj]):
	    		MaximoInput[jj] = indata[jj]
	    	jj+=1
	    outdata = tuple(data[inputs:])
	    if(MaximoTarget<outdata[0]):
	    	MaximoTarget = outdata[0]
	    ds.addSample(indata,outdata)


totalSize = len(ds)


ind=0 
x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
while(ind<int(PorcDivTest*totalSize)):
	indx = 0
	while(indx < 14):
		x[indx] =float((ds['input'][ind][indx])*(1.0/MaximoInput[indx]))
		indx+=1
	y = float((ds['target'][ind])*(1.0/MaximoTarget))
	testSet.addSample(x,y)
	ind+=1

while(ind<totalSize):
	indx = 0
	while(indx < 14):
		x[indx] =float((ds['input'][ind][indx])*(1.0/MaximoInput[indx]))
		indx+=1
	y = float((ds['target'][ind])*(1.0/MaximoTarget))
	trainSet.addSample(x,y)
	ind+=1


if(camada2==0):
	net = buildNetwork(trainSet.indim,camada1,trainSet.outdim,recurrent=True)
else:
	net = buildNetwork(trainSet.indim,camada1,camada2,trainSet.outdim,recurrent=True)
trainer = BackpropTrainer(net,dataset=trainSet,learningrate=Learning,momentum=Momentum,verbose=True)
trainer.trainOnDataset(trainSet,Ciclos)

avgErr =trainer.testOnData(trainSet,verbose=True)

#totalError =0
#n =0
#for error in errorList:
	#totalError += error
	#n+=1
#erroMedio = float(totalError/float(n))

outFile = open("outputFileTest.txt","a")
outFile.write(repr(PorcDivTest)+", "+repr(Ciclos)+", "+repr(Learning)+", "+repr(Momentum)+", "+repr(camada1)+", "+repr(camada2)+", "+repr(avgErr)+"\n")
outFile.close()
