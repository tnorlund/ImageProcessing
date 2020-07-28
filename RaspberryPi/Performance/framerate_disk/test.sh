#!/bin/bash
for thread in 1 2 3
do
    for number_frames in 200 300 400 500 600 700 800 900 1000
    do
        for batch in 1 2 3 4 5
        do 
            ./simpletest_raspicam_cv -t $thread -f $number_frames -b $batch -d new_results.csv -v
        done
    done
done