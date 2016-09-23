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
trainSet = ClassificationDataSet(7, nb_classes=3, class_labels = ["one", "two", "three"])
testSet = ClassificationDataSet(7, nb_classes=3)
k = 0
size = 70
porcDivision = 0.25
category = 0;


for line in inputFile.readlines():
    data = [float(x) for x in line.strip().split() if x != '']
    indata =  tuple(data[:7])
    outdata = tuple(data[7:])
    ds.addSample(indata,outdata)
    k +=1
    if (k == size):
		testdata, traindata = ds.splitWithProportion( porcDivision )    	
		ds.clear() 	
		k = 0
		for inp,targ in testdata:
			testSet.appendLinked(inp,targ-1)
		for inp,targ in traindata:
			trainSet.appendLinked(inp,targ-1)

#print "**************************************"
#print "Number of training patterns: ",len(testSet)
#print "Input and output dimensions: ", testSet.indim, testSet.outdim
#print testSet	
#print "Number of training patterns: ",len(trainSet)	
#print "Input and output dimensions: ", trainSet.indim, trainSet.outdim
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



net = buildNetwork(7,12, 12,3, hiddenclass = TanhLayer, outclass=SoftmaxLayer)
trainer = BackpropTrainer(net, dataset=trainSet, verbose=True,momentum=0.5, learningrate=0.01) 
trainer.trainUntilConvergence()

out = net.activateOnDataset(testSet)
out = out.argmax(axis=1) 
print(out)

#net = buildNetwork(trainSet.indim,13,trainSet.outdim,recurrent=True)
#trainer = BackpropTrainer(net,learningrate=0.01,momentum=0.5,verbose=True)
#trainer.trainOnDataset(trainSet,100)

#print net.activate(trainSet['input'][90])

#trainer.testOnData(trainSet['input'][90],verbose=True)