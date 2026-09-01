#!/usr/bin/env python
############################################################
import os, re
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pylab
from matplotlib import rc

def getLineNumber(fileName, searchString):
    with open(fileName) as f:
        for i,line in enumerate(f, start=1):
           if searchString in line: 
               return i
    raise Exception('string not found')

file='irreps.yaml'

# Define the pattern to search for
pattern = 'ir_label:'

# Open the file in read mode
with open(file, 'r') as f:
    # Read the contents of the file
    file_contents = f.read()
    
    # Find all occurrences of the pattern
    matches = re.findall(pattern, file_contents)

    # Number of matches
    match_count = len(matches)

print(f"The pattern '{pattern}' was found {match_count} times in the file.")

n=match_count #how many mode you want to grep

q=getLineNumber(file, 'normal_modes:')
print (q,type(q))

modes=[]
freq=[]
label=[]

for i in range(n):
    reada=([x.split()[3] for x in open(file).readlines()[q+0+4*i:q+1+4*i]])
    x=reada[0]
    y=x.replace(",","")
    print (y)
    modes.append(int(y))
    #print (reada)
    readb=([x.split()[1] for x in open(file).readlines()[q+1+4*i:q+2+4*i]])
    freq.append(float(readb[0]))
    #print (readb)
    readc=([x.split()[1] for x in open(file).readlines()[q+2+4*i:q+3+4*i]])
    label.append(readc[0])
    #print (readc)

print (modes)
print (freq)
print (label)

with open('TWO_DECIMAL_TABLE_IRREPS.csv','w') as file:
    for i in range(len(modes)):
        file.write(" {0:<5}   {1:<5}   {2:<5}   {3:<5}   {4:<5}\n".format(modes[i],',',round(freq[i],2),',',label[i]))
#        file.write(" {0:<5}   {1:<5}   {2:<5}\n".format(modes[i],round(freq[i],2),label[i]))


