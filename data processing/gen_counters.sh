#!/bin/bash

#for ((date=20220126; date<=20220130; date++)); do
#    input_file="pcap/${date}1400.pcap"
#    output_file="processed_data/packets_${date}.txt"
#
#    # 执行Python命令
#    python do_collection_v1.py "$input_file" "$output_file"
#    echo "$input_file" "pcap to packets"
#done

# 循环从20220101到20220120
for ((date=20220101; date<=20220130; date++)); do    
# 构建输入文件名和输出文件名
    input_file="processed_data/packets_${date}.txt"
    output_file="processed_data/summary_${date}_1us.txt"
    
    # 执行Python命令
    python do_summary.py "$input_file" "$output_file" 0.000001
    echo "$input_file" + 0.1ms
done

for ((date=20220101; date<=20220130; date++)); do
    # 构建输入文件名和输出文件名
    input_file="processed_data/packets_${date}.txt"
    output_file="processed_data/summary_${date}_0.1us.txt"

    # 执行Python命令
    python do_summary.py "$input_file" "$output_file" 0.0000001
    echo "$input_file" + 0.01ms
done

