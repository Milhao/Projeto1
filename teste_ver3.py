
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
porcDivTest = 0.25
Ciclos = 300
Momentum = 0.59  

ds = SupervisedDataSet(inputs,outputs)
trainSet = SupervisedDataSet(inputs,outputs)
testSet = SupervisedDataSet(inputs,outputs)

for line in inputFile.readlines():
	lineaux = line.strip()
	if not lineaux.startswith("%") and len(lineaux)!=0:
	    data = [float(x) for x in lineaux.split() if x != '']
	    indata = tuple(data[:inputs])
	    outdata = tuple(data[inputs:])
	    ds.addSample(indata,outdata)

#testSet, trainSet = ds.splitWithProportion( porcDivision )    	
totalSize = len(ds)
MaxTarget = 0
MaxInput = 0
ind=0
while (ind < len(ds)):
	aux = ds['target'][ind]
	aux2 = max(ds['input'][ind])
	if(MaxTarget < aux):
		MaxTarget = aux
	if(MaxInput < aux2):
		MaxInput = aux2
	ind +=1

MaxTarget = 1/MaxTarget
MaxInput = 1/MaxInput


ind=0 
while(ind<int(porcDivTest*totalSize)):
	#x = float((ds['input'][i])*MaxInput)
	x = ds['input'][ind]
	indx = 0
	while(indx < 14):
		x[indx] =float((ds['input'][ind][indx])*MaxInput)
		indx+=1
	y = float((ds['target'][ind])*MaxTarget)
	testSet.addSample(x,y)
	ind+=1

while(ind<totalSize):
	#x = float((ds['input'][i])*MaxInput)
	x = ds['input'][ind]
	indx = 0
	while(indx < 14):
		x[indx] =float((ds['input'][ind][indx])*MaxInput)
		indx+=1
	y = float((ds['target'][ind])*MaxTarget)
	trainSet.addSample(x,y)
	ind+=1


print "**************************************"
#print "Number of training patterns: ",len(testSet)
#print "Input and output dimensions: ", testSet.indim, testSet.outdim
#print testSet	
#print "Number of training patterns: ",len(trainSet)	
#print "Input and output dimensions: ", trainSet.indim, trainSet.outdim
#print trainSet
#print trainSet['input'][90], trainSet['target'][90]
#print "**************************************"



net = buildNetwork(trainSet.indim,13,trainSet.outdim,recurrent=True)
#net = buildNetwork(trainSet.indim,15,8,trainSet.outdim,outclass=LinearLayer,bias=True,recurrent=True)
trainer = BackpropTrainer(net,dataset=trainSet,learningrate=0.0001,momentum=0.69,verbose=True)
#trainer = BackpropTrainer(net,dataset=trainSet,verbose=True)
#trainer.trainUntilConvergence(dataset=trainSet,continueEpochs=10,validationProportion=0.1)
trainer.trainOnDataset(trainSet,500)
#trainer.trainEpochs(1000)


trainer.testOnData(trainSet,verbose=True)
print numpy.array([net.activate(x) for x, _ in trainSet])
print "======================================"
print numpy.array([net.activate(x)/MaxTarget for x, _ in trainSet])
#test just one
#print "======================================"
#print (testSet['input'][90])/MaxInput
#print (testSet['input'][90])
#print "first ",net.activate(testSet['input'][90]/MaxTarget)
#print "second ",net.activate(testSet['input'][90])/MaxTarget
