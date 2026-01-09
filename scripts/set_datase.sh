#!/bin/bash

set -ex

for a in $(ls |grep .zip); do
		dirName="${a%.zip}"
		echo "zip: => $a extracted => $dirName"
		mkdir $dirName
		mv $a $dirName/
		cd $dirName
		unzip $a
		rm -rf $a
		cd -
	done
