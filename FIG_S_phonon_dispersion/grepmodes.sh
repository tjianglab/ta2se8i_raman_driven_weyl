#!/bin/bash

#n=66 #number of modes
n=`grep "frequency:" mesh.yaml | wc -l`

function grepenergy
{
 word=$(echo | grep "frequency:" mesh.yaml | head -$i | tail -1) #the last line of first n lines equal to nth

 a=$(echo $word | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
 echo "$a"
}

rm modes #remake new modes
 
for ((i=1;i<=n;i++))
#for mode in {1..225}
do 
	v=$(grepenergy)
        echo "$i $v"
	paste <(printf %s "$i") <(printf %s "$v") >> modes
done

