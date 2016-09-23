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
porcDivision = 0.25
ds = SupervisedDataSet(inputs,outputs)


for line in inputFile.readlines():
	lineaux = line.strip()
	if not lineaux.startswith("%") and len(lineaux)!=0:
	    data = [float(x) for x in lineaux.split() if x != '']
	    indata =  tuple(data[:inputs])
	    outdata = tuple(data[inputs:])
	    ds.addSample(indata,outdata)
        
testSet, trainSet = ds.splitWithProportion( porcDivision )    	

#print "**************************************"
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
trainer = BackpropTrainer(net,dataset=trainSet,learningrate=0.0001,momentum=0.59,verbose=True)
#trainer = BackpropTrainer(net,dataset=trainSet,verbose=True)
#trainer.trainUntilConvergence(dataset=trainSet,continueEpochs=10,validationProportion=0.1)
trainer.trainOnDataset(trainSet,300)
#trainer.trainEpochs(1000)
trainer.testOnData(trainSet,verbose=True)
#print numpy.array([net.activate(x) for x, _ in trainSet])
#test just one
#print net.activate(testSet['input'][90])
#print "======================================"
#print testSet['input'][90], testSet['target'][90]