import argparse

# align processed pcap files and use '0' to fill the blank when the original file is not long enough
parser = argparse.ArgumentParser(description='Copy first N lines of a file and fill the rest with a specific pattern.')
parser.add_argument('input_file', help='Path to the input file.')
parser.add_argument('output_file', help='Path to the output file.')
parser.add_argument('max_lines', type=int, help='The maximum number of lines to be written to the output file.')
args = parser.parse_args()

input_file_path = args.input_file
output_file_path = args.output_file
max_lines = args.max_lines

# 执行文件处理
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line_number, line in enumerate(input_file, start=1):
        if line_number <= max_lines:
            output_file.write(line)
        else:
            break
    
    if line_number < max_lines:
        for _ in range(max_lines - line_number):
            output_file.write('0, 0, 0\n')

print(f"Output file has been filled to {max_lines} lines.")
