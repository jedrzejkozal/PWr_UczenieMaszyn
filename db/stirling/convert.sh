#!/bin/bash
for i in $(seq 35)
do
	cd $i
	#echo "$i"
	#pwd
	#for f in *.gif; do  echo "Converting $f"; convert "$f"  "$(basename "$f" .gif).jpg"; done
	rm *.gif
	cd ..
done
