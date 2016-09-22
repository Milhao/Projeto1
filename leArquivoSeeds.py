import sys
import pybrain
from pybrain.tools.shortcuts import buildNetwork

inputFile = open("seeds.txt","r")

dataList = []

for line in inputFile:
	vector = []
	data = line.split()
	i = 0
	while(i<8):
		vector.append(float(data[i]))
		i += 1
	dataList.append(vector)

#print dataList[0][0:7]

net = buildNetwork(7, 5, 1)

print net.activate(dataList[0][0:7])

