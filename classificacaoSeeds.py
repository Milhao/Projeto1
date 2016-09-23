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

PorcDivTest = float(sys.argv[1])
Ciclos = int(sys.argv[2])
Learning = float(sys.argv[3])
Momentum = float(sys.argv[4])
camada1 = int(sys.argv[5])
camada2 = int(sys.argv[6])

k = 0
size = 70
for line in inputFile.readlines():
    data = [float(x) for x in line.strip().split() if x != '']
    indata = tuple(data[:7])
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
	net = buildNetwork(trainSet.indim,camada1,trainSet.outdim, recurrent = True)
else :
	net = buildNetwork(trainSet.indim,camada1,camada2,trainSet.outdim, recurrent = True)
trainer = BackpropTrainer(net,dataset = trainSet,learningrate = Learning,momentum = Momentum, verbose = True)
trainer.trainOnDataset(trainSet,Ciclos)

out = net.activateOnDataset(testSet)
#print(out)
out = out.argmax(axis=1) 
#print out

acerto = total = i = 0
for data in testSet:
	if data[1][0] == 1 and out[i] == 0:
		acerto += 1
		total += 1
	elif data[1][1] == 1 and out[i] == 1:
		acerto += 1
		total += 1
	elif data[1][2] == 1 and out[i] == 2:
		acerto += 1
		total += 1
	else:
		total += 1
	i += 1
acuracia = float(float(acerto)/float(total))

print repr(PorcDivTest)+", "+repr(Ciclos)+", "+repr(Learning)+", "+repr(Momentum)+", "+repr(camada1)+", "+repr(camada2)+", "+repr(acuracia)

outFile = open("outputFile.txt","a")
outFile.write(repr(PorcDivTest)+", "+repr(Ciclos)+", "+repr(Learning)+", "+repr(Momentum)+", "+repr(camada1)+", "+repr(camada2)+", "+repr(acuracia)+"\n")
outFile.close()

#trainer.trainOnDataset(trainSet,100)

#print net.activate(trainSet['input'][90])

#trainer.testOnData(trainSet['input'][90],verbose=True)
