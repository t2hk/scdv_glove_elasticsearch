#!/bin/bash

for ((i=3 ; i<30 ; i++))
do
	num=${i}
        if [ ${i} -lt 10 ]; then
		num=0${i}
	fi

	file=sibou_db_h${num}.xlsx

	if [ ${i} -lt 27 ]; then
	  file=sibou_db_h${num}.xls
	fi

	echo ${file}
        wget https://anzeninfo.mhlw.go.jp/anzen/sib_xls/${file}
done

wget https://anzeninfo.mhlw.go.jp/anzen/sai/kikaisaigai_db28.xlsx
