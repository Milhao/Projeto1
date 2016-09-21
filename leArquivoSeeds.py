import sys
import pybrain

inputFile = open("seeds.txt","r")

dataList=[]

for line in inputFile:
	data = line.strip()
	dataList.append(data.split('\t'))

#print dataList
