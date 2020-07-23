#!/bin/bash
for thread in 1 2 3 4 
do
    for number_frames in 50 75 100 125 150 175 200 225 250 275 300
    do
        for batch in 1 2 3 4 5 10 15 20 25 30 35 40
        do 
            ./simpletest_raspicam_cv -t $thread -f $number_frames -b $batch -d results.csv -v
            ./simpletest_raspicam_cv -t $thread -f $number_frames -b $batch -d results.csv -v
            ./simpletest_raspicam_cv -t $thread -f $number_frames -b $batch -d results.csv -v
            ./simpletest_raspicam_cv -t $thread -f $number_frames -b $batch -d results.csv -v
            ./simpletest_raspicam_cv -t $thread -f $number_frames -b $batch -d results.csv -v
        done
    done
done