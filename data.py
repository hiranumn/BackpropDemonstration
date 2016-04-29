import csv
import numpy as np


#loading mnist data.
def loadmnist(label1, label2, lim=10000):
    print
    print "Loading MNIST data"
    raw = open("mnist.csv").readlines()
    data = []
    count1 = 0
    count2 = 0
    for line in raw[1:lim]:
        temp = [int(l) for l in line.rstrip().split(",")]
        if temp[0] == label1:
            count1+=1
            data.append(dict([('label', np.matrix([True])),('features', np.matrix([i*1.0/255 for i in temp[1:]]))]))
        elif temp[0] == label2:
            count2+=1
            data.append(dict([('label', np.matrix([False])),('features', np.matrix([i*1.0/255 for i in temp[1:]]))]))
            
    print str(label1)+":", count1, "datapoints"
    print str(label2)+":", count2, "datapoints"
    print "Done.."
    print 
    return data
