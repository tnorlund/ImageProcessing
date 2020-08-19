#!/bin/bash
for time in 1 5 10 15 20 25 30 35 40 45 50 55 60 
do
    for fps in 2 5 10 15 20 25 30
    do
        for buffer in 100 200 300 400 500 600 700 800
        do 
            ./frameratetest -v --time $time --fps $fps --buffer $buffer --filename results.csv
        done
    done
done