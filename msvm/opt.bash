#!/bin/bash

# RBF
for i in $(seq 1 10); do
    data_name="./data/100_data.txt"
    cls_name=./opt_cls/${i}_rbf.cls
    ./MCSVM/mcsvm-train -m `wc -l $data_name|awk '{print $1}'` -l 100 -k 5 -b 0.1 -t 0 -s $i $data_name $cls_name
done

# POLY
sed -i -e 's/define HACK ./define HACK 0/' MCSVM/kernel.c
cd MCSVM;make clean;make;cd ..
for i in $(seq 2 10); do
    data_name="./data/100_data.txt"
    cls_name=./opt_cls/${i}_poly.cls
    ./MCSVM/mcsvm-train -m `wc -l $data_name|awk '{print $1}'` -l 100 -k 5 -b 0.1 -t 2 -d $i $data_name $cls_name
done

# SIGMOID
for i in $(seq 0.01 0.05 1.1); do
    data_name="./data/100_data.txt"
    cls_name=./opt_cls/${i}_sig.cls
    sed -i -e 's/define HACK ./define HACK 1/' MCSVM/kernel.c
    sed -i -e "s/define HACK_VALUE .*$/define HACK_VALUE $i/" MCSVM/kernel.c
    cd MCSVM;make clean;make;cd ..
    ./MCSVM/mcsvm-train -m `wc -l $data_name|awk '{print $1}'` -l 100 -k 5 -b 0.1 -t 2 -d 1 $data_name $cls_name
done

for c in ./opt_cls/*.cls; do
    rpt_name=$(echo $c|sed -e 's/opt_cls\(.*\).cls/opt_rpt\1.rpt/')
    if [ -n "$(echo $rpt_name|grep _sig)" ]; then
        value=$(echo $rpt_name|sed 's/.*\/\(.*\)_sig.rpt/\1/')
        sed -i -e 's/define HACK ./define HACK 1/' MCSVM/kernel.c
        sed -i -e "s/define HACK_VALUE .*$/define HACK_VALUE $value/" MCSVM/kernel.c
         cd MCSVM;make clean;make;cd ..
    else
        sed -i -e 's/define HACK ./define HACK 0/' MCSVM/kernel.c
        cd MCSVM;make clean;make;cd ..
    fi
    ./MCSVM/mcol-test ./data/100_data.txt $c ./100_test.txt $rpt_name 3000
done

