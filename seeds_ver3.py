import sys
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import TanhLayer

inputFile = open("seeds.txt","r")
ds = SupervisedDataSet(7,1)
traindata = SupervisedDataSet(7,1)
testdata = SupervisedDataSet(7,1)

trainSet = ClassificationDataSet(7, nb_classes=3, class_labels = ["one", "two", "three"])
testSet = ClassificationDataSet(7, nb_classes=3)
k = 0
size = 70
porcDivTest = 0.25
category = 0;



for line in inputFile.readlines():
    data = [float(x) for x in line.strip().split() if x != '']
    indata =  tuple(data[:7])
    outdata = tuple(data[7:])
    ds.addSample(indata,outdata)
    k +=1
    if (k == size):
		testdata,aux = ds.splitWithProportion( porcDivTest )
		i=0 
		while(i<(1-porcDivTest)* size):
			x = ds['input'][i] 
			y = ds['target'][i]
			traindata.addSample(x,y)
			i+=1
		
		for inp,targ in testdata:
			testSet.appendLinked(inp,targ-1)
		for inp,targ in traindata:
			trainSet.appendLinked(inp,targ-1)
		traindata.clear() 	
		ds.clear()
		k = 0
		
#print "**************************************"
print "Number of training patterns: ",len(testSet)
print "Input and output dimensions: ", testSet.indim, testSet.outdim
#print testSet	
print "Number of training patterns: ",len(trainSet)	
print "Input and output dimensions: ", trainSet.indim, trainSet.outdim
#print trainSet
#print trainSet['input'][90], trainSet['target'][90]

#print trainSet.calculateStatistics()
#print testSet.calculateStatistics()
#print trainSet.getClass(0)
#print(len(testSet.getField('target')))




trainSet._convertToOneOfMany(bounds=[0, 1])
testSet._convertToOneOfMany(bounds=[0, 1])

#print(trainSet.getField('target'))
#print "----------------------"
#print(testSet.getField('target'))



net = buildNetwork(trainSet.indim,13,trainSet.outdim, recurrent=True)
trainer = BackpropTrainer(net,dataset = trainSet,learningrate=0.001,momentum=0.79,verbose=True)
trainer.trainOnDataset(trainSet,700)
out = net.activateOnDataset(testSet)
#print(out)
out = out.argmax(axis=1) 
print(out)





#print net.activate(trainSet['input'][90])
#trainer.testOnData(trainSet['input'][90],verbose=True)