# This file is used to summarize packet files and then generate counters
#
import argparse

parser = argparse.ArgumentParser(description="Extract packet information from pcap file.")
parser.add_argument("input_file", help="Input collection txt file path")
parser.add_argument("output_file", help="Output text file path")
parser.add_argument("period", help="Summary period")
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
period = float(args.period)
empty_period = 0

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    lines = f_in.readlines()

    bytes_in_period = 0
    packets_in_period = 0
    start_time = None


    for line in lines:
        fields = line.split(',')
        ts = float(fields[0])
        packet_len = int(fields[1])


        if start_time is None:
            start_time = ts

        if ts <= start_time + period:
            bytes_in_period += packet_len
            packets_in_period += 1
        else:
            f_out.write(f'{start_time+period:.6f}, '
                        f'{bytes_in_period}, {packets_in_period}\n')

            start_time += period
            bytes_in_period = packet_len
            packets_in_period = 1

            while ts > start_time + period:
                f_out.write(f'{start_time+period:.6f}, 0, 0\n')
                start_time += period

    f_out.write(f'{start_time+period:.6f}, '
                f'{bytes_in_period}, {packets_in_period}\n')
