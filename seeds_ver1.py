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

PorcDivTest = 0.10
Ciclos = 700
Momentum = 0.79;
camada1 = 13
camada2 = 7

k = 0
size = 70
for line in inputFile.readlines():
    data = [float(x) for x in line.strip().split() if x != '']
    indata =  tuple(data[:7])
    outdata = tuple(data[7:])
    ds.addSample(indata,outdata)
    k +=1
    if (k == size):
		testdata, traindata = ds.splitWithProportion( PorcDivTest )    	
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
#print testSet.getField('input')
#print "----------------------"
#print(testSet.getField('target'))


if(camada2==0):
	net = buildNetwork(trainSet.indim,camada1,trainSet.outdim, recurrent=True)
else :
	net = buildNetwork(trainSet.indim,camada1,camada2,trainSet.outdim, recurrent=True)
trainer = BackpropTrainer(net,dataset = trainSet,learningrate=0.001,momentum=Momentum)
trainer.trainOnDataset(trainSet,Ciclos)

out = net.activateOnDataset(testSet)
print(out)
out = out.argmax(axis=1) 
print(out)



#trainer.trainOnDataset(trainSet,100)

#print net.activate(trainSet['input'][90])

#trainer.testOnData(trainSet['input'][90],verbose=True)