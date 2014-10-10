#!/bin/sh

while inotifywait -qq -r -e modify -e create -e move -e delete \
       --exclude '\.sw.?$' tests percolate
do
	clear
	py.test --cov=percolate tests
	sleep 1
done 
