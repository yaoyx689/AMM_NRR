#!/bin/bash  



exef='./build/AMM_NRR'

srcf=./data/d1/source.obj
tarf=./data/d1/target.obj
outf=./data/d1/s2t
${exef} ${srcf} ${tarf} ${outf} 8 100 10

srcf=./data/d2/source.obj
tarf=./data/d2/target.obj
outf=./data/d2/s2t 
${exef} ${srcf} ${tarf} ${outf} 5 100 10