import sys

inputFile = open("teste1.txt","r")

inputFile.readline()
inputFile.readline()
inputFile.readline()

dataList = []

for line in inputFile:
	vector = []
	data = line.split()
	i = 0
	while(i<14):
		vector.append(float(data[i]))
		i += 1
	dataList.append(vector)

print dataList[0]
