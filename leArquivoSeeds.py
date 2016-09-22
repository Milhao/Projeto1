import sys
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

inputFile = open("seeds.txt","r")

dataList = []

for line in inputFile:				#le o arquivo e armazena em dataList(float)
	vector = []
	data = line.split()
	i = 0
	while(i<8):
		vector.append(float(data[i]))
		i += 1
	dataList.append(vector)

net = buildNetwork(7, 5, 1)			#cria a rede neural

dataSet = SupervisedDataSet(7, 1)		#cria o data set

for data in dataList:				#adiciona os dados no data set
	dataSet.addSample(data[0:7],data[7])

#for inpt, target in dataSet:			#printa o data set
#	print inpt, target

trainer = BackpropTrainer(net, dataSet)		#treina a rede

#print Trainer.train()

#print net.activate(dataList[15][0:7])
