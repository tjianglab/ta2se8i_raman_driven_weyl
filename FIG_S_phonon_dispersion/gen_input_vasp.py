import os

poscars = [ x.replace( 'POSCAR-', '' ) for x in os.listdir( '.' ) if 'POSCAR-' in x ]
for item in sorted( poscars ):
    folder = 'disp-' + item 
    os.system( 'mkdir -p ' + folder )
    os.chdir( folder )
    os.system( 'ln -fs ../POSCAR-' + item + ' POSCAR' )
    os.system( 'ln -fs ../KPOINTS')
    os.system( 'ln -fs ../POTCAR')
    os.system( 'ln -fs ../INCAR')
#    os.system( 'ln -fs ../modpw')
#    os.system( 'ln -fs ../label')
#    os.system( './../writeinput.sh ' )

    ex1="cat ../jobcorinclscf | sed 's/XX1/'"
#    ex1="cat ../jobcorincl | sed 's/XX1/'"
#    ex1="cat ../jobvaspgam | sed 's/XX1/'" 
    ex2="'/' > job"
#    os.system ( ex1+item+ex2 )
#    os.system ('sbatch job')

    os.chdir( '../' )

