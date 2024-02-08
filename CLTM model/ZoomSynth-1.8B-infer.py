##################################################
# As described in our paper, we use a 1.8B CLTM model to run our Counter-to-Packets task
# Here is the CLTM model including several GTTs which is a tree-based model.
# The factor k of each layer of GTT is 10 and the total number of GTTs is determined by specific task and our resoures
# Please modify the hyperparameters below to make it runnable on your testbed and fit to your task
##################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of avaliable GPUs: {num_gpus}")
else:
    raise SystemError("No CUDA or GPUs")

devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", 
           "cuda:4", "cuda:5", "cuda:6", "cuda:7"]

#Transformer Encoder in GTT
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(20000, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        pos_enc = self.pos_embedding(pos)
        x = x + pos_enc
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x

#Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super().__init__()
        self.embedding = nn.Linear(output_size, hidden_size)
        self.pos_embedding = nn.Embedding(20000, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, num_heads, hidden_size * 4, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, memory):
        x = self.embedding(x)
        seq_len = x.size(1)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        pos_enc = self.pos_embedding(pos)
        x = x + pos_enc
        x = x.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2) 
        x = self.transformer_decoder(x, memory)
        x = x.permute(1, 0, 2)
        x = self.output(x)
        return x

#Transformer
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout, output_size):
        super().__init__()
        self.encoder = TransformerEncoder(input_size, hidden_size, num_layers, num_heads, dropout)
        self.decoder = TransformerDecoder(input_size, hidden_size, num_layers, num_heads, dropout, output_size)
        
    def forward(self, x, y):
        memory = self.encoder(x)
        out = self.decoder(y, memory)
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.bilstm(x)
        out = self.fc(out)
        return out

class JointModel(nn.Module):
    def __init__(self, transformer, bilstm):
        super(JointModel, self).__init__()
        self.transformer = transformer
        self.bilstm = bilstm

    def forward(self, x, y):

        y_hat = self.transformer(x, y)

        y_plus = self.bilstm(y)


        return y_hat + y_plus

# 
#10x
input_summary_file_1 = './ton_test/grain_1.txt'
#100x
input_summary_file_2 = './ton_test/grain_2.txt'
#1000x
input_summary_file_3 = './ton_test/grain_3.txt'
#10000x
input_summary_file_4 = './ton_test/grain_4.txt'
#100000x
input_summary_file_5 = './ton_test/grain_5.txt'
#1000000x
input_summary_file_6 = './ton_test/grain_6.txt'
#pkt
output_packet_file = './ton_test/raw.csv'

# upscaling_times = 10
input_size = 3
hidden_size = 256
output_size = 2
num_layers = 10
num_heads = 4
flatten_len = 5 
S_processed = []
P_processed = []

total_lines = 0
num_chunks = 1
S_tensor = None
P_tensor = None

with open(input_summary_file_1,'r') as file:
    lines = file.readlines()
S_processed = []
total_lines = 0
if lines:
    line_count = len(lines)
    print("line_count: ", line_count)
for i in range(0, line_count):
    ts, byte_count, packet_count= lines[i].strip().split(',')
    ts, byte_count, packet_count= float(ts), float(byte_count), float(packet_count)
    total_lines += 1
    S_processed.append([ts, byte_count, packet_count])
S_tensor_1 = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
print("total_lines:", total_lines)
S_tensor_1 = S_tensor_1[:, :23000, :]

with open(input_summary_file_2,'r') as file:
    lines = file.readlines()
S_processed = []
total_lines = 0
if lines:
    line_count = len(lines)
    print("line_count: ", line_count)
for i in range(0, line_count):
    ts, byte_count, packet_count= lines[i].strip().split(',')
    ts, byte_count, packet_count= float(ts), float(byte_count), float(packet_count)
    total_lines += 1
    S_processed.append([ts, byte_count, packet_count])
S_tensor_2 = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
print("total_lines:", total_lines)
S_tensor_2 = S_tensor_2[:, :2300, :]

#1000
with open(input_summary_file_3,'r') as file:
    lines = file.readlines()
S_processed = []
total_lines = 0
if lines:
    line_count = len(lines)
    print("line_count: ", line_count)
for i in range(0, line_count):
    ts, byte_count, packet_count= lines[i].strip().split(',')
    ts, byte_count, packet_count= float(ts), float(byte_count), float(packet_count)
    total_lines += 1
    S_processed.append([ts, byte_count, packet_count])
S_tensor_3 = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
print("total_lines:", total_lines)
S_tensor_3 = S_tensor_3[:, :230, :]

#10000
with open(input_summary_file_4,'r') as file:
    lines = file.readlines()
S_processed = []
total_lines = 0
if lines:
    line_count = len(lines)
    print("line_count: ", line_count)
for i in range(0, line_count):
    ts, byte_count, packet_count= lines[i].strip().split(',')
    ts, byte_count, packet_count= float(ts), float(byte_count), float(packet_count)
    total_lines += 1
    S_processed.append([ts, byte_count, packet_count])
S_tensor_4 = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
print("total_lines:", total_lines)
S_tensor_4 = S_tensor_4[:, :20, :]

#100000
with open(input_summary_file_5,'r') as file:
    lines = file.readlines()
S_processed = []
total_lines = 0
if lines:
    line_count = len(lines)
    print("line_count: ", line_count)
for i in range(0, line_count):
    ts, byte_count, packet_count= lines[i].strip().split(',')
    ts, byte_count, packet_count= float(ts), float(byte_count), float(packet_count)
    total_lines += 1
    S_processed.append([ts, byte_count, packet_count])
S_tensor_5 = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
print("total_lines:", total_lines)
S_tensor_5 = S_tensor_5[:, :20, :]

#1000000
with open(input_summary_file_6,'r') as file:
    lines = file.readlines()
S_processed = []
total_lines = 0
if lines:
    line_count = len(lines)
    print("line_count: ", line_count)
for i in range(0, line_count):
    ts, byte_count, packet_count= lines[i].strip().split(',')
    ts, byte_count, packet_count= float(ts), float(byte_count), float(packet_count)
    total_lines += 1
    S_processed.append([ts, byte_count, packet_count])
S_tensor_6 = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
print("total_lines:", total_lines)


with open(output_packet_file, 'r') as f:
    lines = f.readlines()
    lines = lines[1:]
    line_count = len(lines)

P_processed = []
for i in range(0, line_count):
    srcip,dstip,srcport,dstport,proto,ts,td,pkt,byt,label,type,ts_interval,per_byt,my_label,protocol = lines[i].strip().split(',')
    srcip,dstip,srcport,dstport,protocol,ts,byte_count,my_label = float(srcip),float(dstip),float(srcport),float(dstport),float(protocol),float(ts_interval),float(per_byt),float(my_label)  
    if(byte_count < 1500):
        P_processed.append([srcip,dstip,srcport,dstport,protocol,byte_count,ts])
P_tensor = torch.tensor(P_processed, dtype=torch.float32).unsqueeze(0)
P_tensor = P_tensor[:, :70000, :]
print(S_tensor_1.shape, S_tensor_2.shape, S_tensor_3.shape, S_tensor_4.shape, S_tensor_5.shape, S_tensor_6.shape, P_tensor.shape) 
# print("S_tensor:", S_tensor)
# print("P_tensor:", P_tensor)
print("total_lines:", total_lines)
print(f"finished loading a new file")

print("data loaded, processing...")

# input_mean = S_continuous_features.mean(dim=1, keepdim = True)
# input_std =  S_continuous_features.std(dim=1, keepdim = True) + 1e-6
# S_continuous_features = (S_continuous_features - input_mean) / (input_std)
# output_mean = P_continuous_features.mean(dim=1, keepdim = True)
# output_std =  P_continuous_features.std(dim=1, keepdim = True) + 1e-6
# P_continuous_features = (P_continuous_features - output_mean) / (output_std)
# S_tensor = torch.cat((S_continuous_features, S_binary_features), dim=2)
# P_tensor = torch.cat((P_continuous_features, P_binary_features), dim=2)

S_tensor_1_mean = S_tensor_1.mean(dim=1, keepdim = True)
S_tensor_1_std =  S_tensor_1.std(dim=1, keepdim = True) + 1e-6
S_tensor_1 = (S_tensor_1 - S_tensor_1_mean) / (S_tensor_1_std)

S_tensor_2_mean = S_tensor_2.mean(dim=1, keepdim = True)
S_tensor_2_std =  S_tensor_2.std(dim=1, keepdim = True) + 1e-6
S_tensor_2 = (S_tensor_2 - S_tensor_2_mean) / (S_tensor_2_std)

S_tensor_3_mean = S_tensor_3.mean(dim=1, keepdim = True)
S_tensor_3_std =  S_tensor_3.std(dim=1, keepdim = True) + 1e-6
S_tensor_3 = (S_tensor_3 - S_tensor_3_mean) / (S_tensor_3_std)

S_tensor_4_mean = S_tensor_4.mean(dim=1, keepdim = True)
S_tensor_4_std =  S_tensor_4.std(dim=1, keepdim = True) + 1e-6
S_tensor_4 = (S_tensor_4 - S_tensor_4_mean) / (S_tensor_4_std)

S_tensor_5_mean = S_tensor_5.mean(dim=1, keepdim = True)
S_tensor_5_std =  S_tensor_5.std(dim=1, keepdim = True) + 1e-6
S_tensor_5 = (S_tensor_5 - S_tensor_5_mean) / (S_tensor_5_std)

S_tensor_6_mean = S_tensor_6.mean(dim=1, keepdim = True)
S_tensor_6_std =  S_tensor_6.std(dim=1, keepdim = True) + 1e-6
S_tensor_6 = (S_tensor_6 - S_tensor_5_mean) / (S_tensor_5_std)#use _5's mean and std

P_tensor_mean = P_tensor.mean(dim=1, keepdim = True)
P_tensor_std =  P_tensor.std(dim=1, keepdim = True) + 1e-6
P_tensor = (P_tensor - P_tensor_mean) / (P_tensor_std)

print("output_mean_shape: ", P_tensor_mean.shape, "output_std_shape: ",P_tensor_mean.shape)
# print(input_mean, input_std, output_mean, output_std)
# print(torch.isnan(input_mean).any(), torch.isinf(input_mean).any())
# print(torch.isnan(input_std).any(), torch.isinf(input_std).any())
print(torch.isnan(P_tensor_mean).any(), torch.isinf(P_tensor_mean).any())
print(torch.isnan(P_tensor_std).any(), torch.isinf(P_tensor_std).any())


# slice into batches
S_tensor_1 = S_tensor_1.view(10, -1, 3).to(device)
S_tensor_2 = S_tensor_2.to(device)
S_tensor_3 = S_tensor_3.to(device)
S_tensor_4 = S_tensor_4.view(10, -1, 3).to(device)
S_tensor_5 = S_tensor_5.to(device)
S_tensor_6 = S_tensor_6.to(device)
P_tensor = P_tensor.view(10, -1, 7).to(device)


input_test_chunk = None
output_test_chunk= None
output = None
#-------------------------10 * 10 * 10 * 10 * 10 * 10--------------------------------------------------#
transformer_model = Transformer(3, hidden_size, num_layers, num_heads, dropout, 3)
bilstm_model = BiLSTMModel(3, hidden_size, 3)
joint_model = JointModel(transformer_model, bilstm_model)
joint_model.to(devices[0])
#1' 10x
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/cidds_last6_10x.pth'))
batch_size, sequence_length, feature_length = S_tensor_5.shape
y_tensor = torch.zeros(batch_size, sequence_length, feature_length).to(devices[0])
output1 = joint_model(S_tensor_6, y_tensor)
print("S_tensor_6:", S_tensor_6)
print("output1.shape:", output1.shape)
output = output1.cpu()
output = output * S_tensor_5_std + S_tensor_5_mean
print("output1:", output)
with open('/root/traffic_recovery/code/result/cidds_1times_10x.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')
#2' 10x
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_last5_10x.pth'))
batch_size, sequence_length, feature_length = S_tensor_4.shape
y_tensor = torch.zeros(batch_size, sequence_length, feature_length).to(devices[0])
output2 = joint_model(S_tensor_5, y_tensor)
print("output2.shape:", output2.shape)
output = output2.cpu()
output = output * S_tensor_4_std + S_tensor_4_mean
with open('/root/traffic_recovery/code/result/ton_2times_10x.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')
#3' 10x
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_last4_10x.pth'))
batch_size, sequence_length, feature_length = S_tensor_3.shape
y_tensor = torch.zeros(batch_size, sequence_length, feature_length).to(devices[2])
output3 = joint_model(output2, y_tensor)
print("output3.shape:", output3.shape)
output = output3.cpu()
output = output * S_tensor_3_std + S_tensor_3_mean
with open('/root/traffic_recovery/code/result/ton_3times_10x.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')
#4' 10x
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_last3_10x.pth'))
batch_size, sequence_length, feature_length = S_tensor_2.shape
y_tensor = torch.zeros(batch_size, sequence_length, feature_length).to(devices[3])
output4 = joint_model(output3, y_tensor)
print("output4.shape:", output4.shape)
output = output4.cpu()
output = output * S_tensor_2_std + S_tensor_2_mean
with open('/root/traffic_recovery/code/result/ton_4times_10x.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')
#5' 10x
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_last2_10x.pth'))
output4 = output4.view(10, -1, 3)
batch_size, sequence_length, feature_length = S_tensor_1.shape
y_tensor = torch.zeros(batch_size, sequence_length, feature_length).to(devices[4])
output5 = joint_model(output4, y_tensor)
print("output5.shape:", output5.shape)
output = output5.cpu()
output = output * S_tensor_1_std + S_tensor_1_mean
with open('/root/traffic_recovery/code/result/ton_5times_10x.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')
#6' 10x
transformer_model = Transformer(3, hidden_size, num_layers, num_heads, dropout, 2)
bilstm_model = BiLSTMModel(2, hidden_size, 2)
joint_model = JointModel(transformer_model, bilstm_model)
joint_model.to(devices[5])
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_last_10x.pth'))
output5 = output5.view(10, -1, 3)
batch_size, sequence_length, feature_length = P_tensor.shape
y_tensor = torch.zeros(batch_size, sequence_length, 2).to(devices[5])#not feature_length but 2(ts, byte_count)
output6 = joint_model(output5, y_tensor)
print("output6.shape:", output6.shape)

# transformer_model = Transformer(3, hidden_size, num_layers, num_heads, dropout, 2)
# bilstm_model = BiLSTMModel(2, hidden_size, 2)
# joint_model = JointModel(transformer_model, bilstm_model)
# joint_model.to(device)
# joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_last_10x.pth'))
# batch_size, sequence_length, feature_length = P_tensor.shape
# y_tensor = torch.zeros(batch_size, sequence_length, 2).to(device)#not feature_length but 2(ts, byte_count)
# output3 = joint_model(output2, y_tensor)
# print("output3.shape:", output3.shape)

transformer_model = Transformer(2, hidden_size, num_layers, num_heads, dropout, 7 )
bilstm_model = BiLSTMModel(7, hidden_size, 7)
joint_model = JointModel(transformer_model, bilstm_model)
joint_model.to(devices[6])
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/ton_header_gen.pth'))
batch_size, sequence_length, feature_length = P_tensor.shape
y_tensor = torch.zeros(batch_size, sequence_length, feature_length).to(devices[6])
output7 = joint_model(output6, y_tensor)
print("output7.shape:", output7.shape)
output7 = output7.cpu()
output7 = output7 * P_tensor_std + P_tensor_mean

with open('/root/traffic_recovery/code/result/ton_infer_pkt_after_7*10x.txt', 'w') as file:
    for batch_idx in range(output7.shape[0]):
        for row_idx in range(output7.shape[1]):
            row_data = np.array(output7[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')