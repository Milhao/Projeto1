import sys

inputFile = open("teste1.txt","r")

inputFile.readline()
inputFile.readline()
inputFile.readline()

dataList = []
vector = []

for line in inputFile:
	data = line.split()
	i = 0
	while(i<8):
		vector.append(float(data[i]))
		i += 1
	dataList.append(vector)

#print dataList
