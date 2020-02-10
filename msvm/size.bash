#!/bin/bash

# rbf kernel
#for i in ./data/*_data.txt; do
     #cls_name=$(echo $i|sed -e 's/\.\/data\(.*\)\.txt/\.\/size_cls\/\1_rbf.cls/')
     #./MCSVM/mcsvm-train -m `wc -l $i|awk '{print $1}'` -l 100 -k 5 -b 0.1 -t 0 -s 3 $i $cls_name
#done

# poly kernel
#sed -i -e 's/define HACK ./define HACK 0/' MCSVM/kernel.c
#cd MCSVM;make clean;make;cd ..
#for i in ./data/*_data.txt; do
     #cls_name=$(echo $i|sed -e 's/\.\/data\(.*\)\.txt/\.\/size_cls\/\1_poly.cls/')
     #./MCSVM/mcsvm-train -m `wc -l $i|awk '{print $1}'` -l 100 -k 5 -b 0.1 -t 2 -d 2 $i $cls_name
#done

# sigmoid kernel
#sed -i -e 's/define HACK ./define HACK 1/' MCSVM/kernel.c
#sed -i -e "s/define HACK_VALUE .*$/define HACK_VALUE 1.06/" MCSVM/kernel.c
#cd MCSVM;make clean;make;cd ..
#for i in ./data/*_data.txt; do
     #cls_name=$(echo $i|sed -e 's/\.\/data\(.*\)\.txt/\.\/size_cls\/\1_sig.cls/')
     #./MCSVM/mcsvm-train -m `wc -l $i|awk '{print $1}'` -l 100 -k 5 -b 0.1 -t 2 -d 1 $i $cls_name
#done

for c in ./size_cls/*.cls; do
    rpt_name=$(echo $c|sed -e 's/size_cls\(.*\).cls/size_rpt\1.rpt/')
    if [ -n "$(echo $rpt_name|grep _sig)" ]; then
        value=1.06
        sed -i -e 's/define HACK ./define HACK 1/' MCSVM/kernel.c
        sed -i -e "s/define HACK_VALUE .*$/define HACK_VALUE $value/" MCSVM/kernel.c
        cd MCSVM;make clean;make;cd ..
    else
        sed -i -e 's/define HACK ./define HACK 0/' MCSVM/kernel.c
        cd MCSVM;make clean;make;cd ..
    fi
    ./MCSVM/mcol-test ./data/100_data.txt $c ./100_test.txt $rpt_name 3000
done

