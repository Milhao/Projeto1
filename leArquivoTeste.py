import sys

inputFile = open("teste1.txt","r")

inputFile.readline()
inputFile.readline()
inputFile.readline()

dataList=[]

for line in inputFile:
	data = line.strip()
	dataList.append(data.split(' '))

#print dataList
