import dpkt
import socket
import argparse


parser = argparse.ArgumentParser(description="Extract packet information from pcap file.")
parser.add_argument("input_pcap", help="Input pcap file path")
parser.add_argument("output_file", help="Output text file path")
args = parser.parse_args()


pcap_file = args.input_pcap

output_file = args.output_file


with open(pcap_file, 'rb') as f, open(output_file, 'w') as f_out:
    pcap = dpkt.pcap.Reader(f)
    for ts, buf in pcap:
        eth = dpkt.ethernet.Ethernet(buf)
        if isinstance(eth.data, dpkt.ip.IP):
            ip = eth.data
            src_ip = socket.inet_ntoa(ip.src)
            dst_ip = socket.inet_ntoa(ip.dst)
            protocol = ip.p
            if isinstance(ip.data, dpkt.tcp.TCP):
                src_port = ip.data.sport
                dst_port = ip.data.dport
            elif isinstance(ip.data, dpkt.udp.UDP):
                src_port = ip.data.sport
                dst_port = ip.data.dport
            else:
                src_port = 0
                dst_port = 0

            f_out.write(f'{ts},{len(buf)},{src_ip},{dst_ip},{src_port},{dst_port},{protocol}\n')

