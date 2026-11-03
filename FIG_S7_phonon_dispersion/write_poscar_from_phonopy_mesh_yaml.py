#!/usr/bin/env python
############################################################
import os, re
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import pylab
from matplotlib import rc

from ase import *
from ase.io import *
#from ase.build import *
from ase import Atoms, Atom



def getLineNumber(fileName, searchString):
    with open(fileName) as f:
        for i,line in enumerate(f, start=1):
           if searchString in line: 
               return i
    raise Exception('string not found')

def get_value_after_match(fileName, searchString):
    select_value=[]
    with open(fileName) as f:
        for i,line in enumerate(f, start=1):
           #print (i,line)
           if searchString in line:
               #energytxt=[x.split( ) for x in line]
               list_line=line.split()
               index = list_line.index(searchString)
               valuex=list_line[index+2]
               valuey=list_line[index+3]
               valuez=list_line[index+4]
               valuex= valuex.replace(',', '')
               valuey= valuey.replace(',', '')
               valuez= valuez.replace(',', '')

               select_value.append([float(valuex),float(valuey),float(valuez)])

#return i
#   print (select_list)
    return select_value
#    raise Exception('string not found')

'''
file='POSCAR.C'

q=getLineNumber(file, 'Cartesian')
#print (q)

readf=([x.split() for x in open(file).readlines()[q-2:q-1]])
atom_list=readf[0]
atom_list=[int(i) for i in atom_list]
atom_num=sum(atom_list)
#print (atom_list,type(atom_list[0]))
print ('number of atoms is',atom_num)

pos=([x.split() for x in open(file).readlines()[q:q+12]])
#print(pos[0][0],type(pos[0][0]))
#print (pos[0][1])
'''


mode=11
file='mesh.yaml'

q=getLineNumber(file,'natom:')
readf=([x.split()[1] for x in open(file).readlines()[q-1:q]])
atom_num=int(readf[0])
print ('number of atoms is',atom_num)

'''             
q=getLineNumber(file,'- # '+ str(mode))# 12
print ('the line of mode',mode,'is',q)
vectors=[]
for i in range(atom_num):
    readf=([x.split()[2] for x in open(file).readlines()[q+3+4*i:q+3+3+4*i]])
    for j in range(len(readf)):
        readf[j] = readf[j].replace(',', '')
    vectors.append(readf)
'''
#q=getLineNumber(file,'lattice:')
lattice=[]
for i in range(2,5):
    readf=([x.split()[i] for x in open(file).readlines()[q+1:q+4]])
    for j in range(len(readf)):
        readf[j] = readf[j].replace(',', '')
        #print (readf)
    lattice.append(readf)
#print (lattice)
lattice=np.transpose(lattice)
print ('lattice is')
print (lattice)

a=get_value_after_match(file,'coordinates:')
print ('coordinates is')
print (a)

atoms= Atoms('Ta4Se16I2',scaled_positions=a,cell=lattice)
#atoms= Atoms('Co2Cl4N8C72H64',positions=pos,cell=cell_poscar)
# Make it 10 times larger along x and 5 times larger along y
write('CONTCAR.PBE.FORCE.1E-3.PC.FROM.PHONOPY.Ta4Se16I2', atoms, format ='vasp', direct =True, vasp5 = True, sort = False)

#print (vectors,vectors[2][0])

#atom_list=readf[0]

#with open('mode.dat','w') as file:
#    for i in range(atom_num):
#        file.write(" {0:>2}  {1:>15}  {2:>15}  {3:>15}  {4:>15}  {5:>15}  {6:>15}\n".format(i+1,pos[i][0],pos[i][1],pos[i][2],vectors[i][0],vectors[i][1],vectors[i][2]))

