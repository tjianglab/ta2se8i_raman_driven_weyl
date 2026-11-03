#from ase.io import read
import os

#os.chdir ( 'VASP_TEMPLATE' )
poscars = [ x.replace( 'POSCAR-', '' ) for x in os.listdir( '.' ) if 'POSCAR-' in x ]
num_disp=len(poscars)
#end=["{0:03}".format(i) for i in range(121)]
end_disp="{0:03}".format(num_disp)
print ('how many disp is',end_disp,type(end_disp))

#for item in sorted( poscars ):
#system = read( 'POSCAR' )
#num_atoms=system.get_number_of_atoms()
#print 'number of atom is ',num_atoms
#with open ('nat.txt','w') as f:
#	f.write(str(num_atoms) + '\n')
#os.system( 'cp ../cutforce.sh .')
#os.system ( './cutforce.sh')
#os.chdir ('../')
#os.system ( 'mkdir data')
#os.chdir ('data')
#os.system ('python ../collect.py')
#os.chdir ( '../' )
#num_disp=6*num_atoms
#print 'number of displacement is ',num_disp
ex1='phonopy -f disp-{001..'
ex2='}/vasprun.xml'
#ex2='}/vasprun.xml --fc_symmetry'
#os.system(ex1+end_disp+ex2)
#os.system( 'phonopy -f data/vasprun_{001..012}.xml' )
#os.system( 'phonopy -p mesh.conf --fc_symmetry' )
#os.system( 'phonopy --dim="1 1 1" -c POSCAR mesh.conf -p -s --fc-symmetry')
os.system( 'phonopy --dim="2 2 2" -c POSCAR mesh.conf -p -s --fc-symmetry')
#os.system( 'phonopy --dim="2 2 2" -c POSCAR band.conf -p -s --fc-symmetry')
#os.system( 'phonopy --dim="4 4 4" -c POSCAR mesh.conf -p -s --fc_symmetry')
#os.system( 'phonopy --dim="4 4 4" -c POSCAR band.conf -p -s --fc_symmetry')
#os.system ('> FORCE_COLLECT_FINISHED')


