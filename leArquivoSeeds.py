import sys
import pybrain

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

print dataList[0][0:7]
