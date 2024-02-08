#!/bin/bash
# this file can be used to call "do_summarize.py" to generate counter files in batches
# file template: packets/summary_{date}_{counter_grain}.txt

#generate us counters
for ((date=20220101; date<=20220130; date++)); do    
    input_file="processed_data/packets_${date}.txt"
    output_file="processed_data/summary_${date}_1ms.txt"
    
    # call do_summarize.py
    python do_summarize.py "$input_file" "$output_file" 0.001
    echo "$input_file" + 1ms
done

#generate us counters
for ((date=20220101; date<=20220130; date++)); do
    input_file="processed_data/packets_${date}.txt"
    output_file="processed_data/summary_${date}_us.txt"

    python do_summarize.py "$input_file" "$output_file" 0.000001
    echo "$input_file" + 1us
done

