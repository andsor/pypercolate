#!/bin/sh

while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' percolate
do
	clear
	py.test --cov=percolate percolate
	sleep 1
done 
