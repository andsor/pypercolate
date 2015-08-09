#!/bin/sh

py.test
while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' percolate
do
	clear
	py.test
	sleep 1
done 
