#!/bin/bash

# for ((date=20220101; date<=20220115; date++)); do
#    input_file="${date}1400.pcap"
#    output_file="packets_${date}.txt"

#
#    python do_collection_v1.py "$input_file" "$output_file"
#    echo "$input_file" "pcap to packets"
# done

# 
for ((date=01; date<=15; date++)); do    

    input_file="summary_${date}_1s.txt"
    output_file="summary_${date}_1s_aligned.txt"
    
    python align_file.py "$input_file" "$output_file" 900
    echo "$input_file" + 1s
done

for ((date=01; date<=15; date++)); do    
    input_file="summary_${date}_1ms.txt"
    output_file="summary_${date}_1ms_aligned.txt"
    

    python align_file.py "$input_file" "$output_file" 900000
    echo "$input_file" + 1ms
done

for ((date=01; date<=15; date++)); do
    input_file="summary_${date}_1us.txt"
    output_file="summary_${date}_1us_aligned.txt"


    python align_file.py "$input_file" "$output_file" 900000000
    echo "$input_file" + 1us
done


