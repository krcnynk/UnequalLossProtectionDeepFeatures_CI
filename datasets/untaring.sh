#!/bin/bash
fname=files
echo "${fname}.tar"
mkdir $fname
tar xfv "${fname}.tar" --directory "$fname"
cd $fname
for t in * ; do
	echo "${t%.*}"
	mkdir "${t%.*}"
	tar xfv $t --directory "${t%.*}"
	#mv "$t" "$bn"
done
