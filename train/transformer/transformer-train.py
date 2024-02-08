##################################################
# This file is used to build transformer of GTT and train it using coarse-grain and fine-grain data
#
##################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_summary_files = [
                       './processed_data/summary_20220101_1s.txt',
                       # ...
                       ]
output_packet_files =  [
                       './processed_data/summary_20220101_0.1s.txt', 
                       # ...
                       ]
upscaling_times = 10
input_size = 2
hidden_size = 256
output_size = 8
num_layers = 20
num_heads = 4
dropout = 0.1
lr = 0.01
flatten_len = 10
S_processed = []
P_processed = []

S_lines_in_one_batch = 100
total_lines = 0
num_chunks = 1
no = 0
S_tensor = None
P_tensor = None

#Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_embedding = nn.Embedding(5000, hidden_size)
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
        self.pos_embedding = nn.Embedding(5000, hidden_size)
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
        # x[:,:, 0] = F.relu(x[:,:, 0])
        # x[:, :, 0] = torch.sigmoid(x[:, :, 0])
        # condition = x[:, :, 0] < 0  
        # x[:, :, 0] = torch.where(condition, torch.zeros_like(x[:, :, 0]), x[:, :, 0])  

        # x[:, :, 0] = summary_timeslot * num_timeslots * x[:, :, 0] / torch.sum(x[:, :, 0], dim=1, keepdim=True)

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


for input_file, output_file in zip(input_summary_files, output_packet_files):
    print("processing no.", no)
    no += 1
    with open(input_file,'r') as file:
        lines = file.readlines()
    if lines:
        line_count = len(lines)
        S_processed = []
        total_lines = 0
        print("line_count: ", line_count)
    for i in range(0, int(line_count/flatten_len)):
        inner_list = []
        for j in range(0, flatten_len):
            # print("S:",i*flatten_len+j)
            ts,byte_count,packet_count = lines[i*flatten_len+j].strip().split(',')
            ts, byte_count, packet_count = float(ts), float(byte_count), float(packet_count)
            # print(f"byte_count: {byte_count:.2f}, packet_count: {packet_count:.2f}")
            # S_processed.append([byte_count,packet_count])
            for k in range(0, upscaling_times):
                inner_list.append(byte_count/upscaling_times)        
            total_lines += 1
        S_processed.append(inner_list)
    S = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
    if S_tensor is None:
        S_tensor = S
    else:
        S_tensor = torch.cat((S_tensor, S), dim =0)
    print("total_lines:", total_lines)

    with open(output_file, 'r') as f:
        lines = f.readlines()
        line_count = len(lines)
    P_processed = []
    for i in range(0, int(total_lines/flatten_len)):
        inner_list = []
        for j in range(0, upscaling_times*flatten_len):
            # print("P:",i*upscaling_times*flatten_len+j)
            if(i*upscaling_times*flatten_len+j >= line_count):
                inner_list.append(0)
            else:
                ts,byte_count,packet_count = lines[i*upscaling_times*flatten_len+j].strip().split(',')
                ts, byte_count, packet_count = float(ts), float(byte_count), float(packet_count)
                # P_processed.append([byte_count, packet_count])
                inner_list.append(byte_count)
        P_processed.append(inner_list)
    P = torch.tensor(P_processed, dtype=torch.float32).unsqueeze(0)
    if P_tensor is None:
        P_tensor = P
    else:
        P_tensor = torch.cat((P_tensor, P), dim=0)
print(S_tensor.shape, P_tensor.shape) 
print("S_tensor:", S_tensor)
print("P_tensor:", P_tensor)
print(f"finished loading a new file")

print("data loaded, processing...")
# S_tensor = S_tensor.view(-1,S_lines_in_one_batch, input_size)
# P_tensor = P_tensor.view(-1,S_lines_in_one_batch, output_size)
print(S_tensor.shape, P_tensor.shape) 
input_mean = S_tensor.mean(dim=2, keepdim = True)
input_std =  S_tensor.std(dim=2, keepdim = True)
S_tensor = (S_tensor - input_mean) / (input_std/10)
output_mean = P_tensor.mean(dim=2, keepdim = True)
output_std =  P_tensor.std(dim=2, keepdim = True)
P_tensor = (P_tensor - output_mean) / (output_std/10)

input_chunks = torch.chunk(S_tensor, num_chunks, dim=0)
output_chunks = torch.chunk(P_tensor, num_chunks, dim=0)

model = Transformer(input_size, hidden_size, num_layers, num_heads, dropout, output_size)
model.to(device)

#define wasserstein distance loss function(use pairwise_distance)
def wasserstein_distance(y_true, y_pred):
    batch_size, sample_num, sequence_len = y_true.size()
    y_true = y_true.reshape(batch_size * sample_num, sequence_len).to(device)
    y_pred = y_pred.reshape(batch_size * sample_num, sequence_len).to(device)
    dist = F.pairwise_distance(y_true, y_pred, p=1)
    return torch.mean(dist)

#define my MSE-plus loss function
def my_mse_loss(y_pred, y_true):
    mse = (y_pred - y_true) ** 2
    weight = torch.abs(y_pred[..., 0] - y_true[...,0])**2
    weight = weight.unsqueeze(-1)
    weighted_mse = torch.mean(mse * weight)
    return weighted_mse

# define loss function
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

input_test_chunk = None
output_test_chunk= None

num_epochs = 20000
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (input_chunk, output_chunk) in enumerate(zip(input_chunks, output_chunks)):
        optimizer.zero_grad()
        input_chunk = input_chunk.to(device)
        output_chunk = output_chunk.to(device)
        output = model(input_chunk, output_chunk) 
        loss = criterion(output, output_chunk).to(device)
        # loss = wasserstein_distance(output_chunk, output).to(device)
        # loss = my_mse_loss(output, output_chunk).to(device)
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    total_loss = total_loss / num_chunks
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))

S_test = []
total_lines = 0
with open(val_input_file,'r') as file:
    lines = file.readlines()
    line_count = len(lines)
for i in range(0, int(line_count/flatten_len)):
    S_test = []
    for j in range(0, flatten_len):
        ts,byte_count,packet_count = lines[i*flatten_len+j].strip().split(',')
        ts, byte_count, packet_count = float(ts), float(byte_count), float(packet_count)
        for k in range(0, upscaling_times):
            S_test.append(byte_count/upscaling_times)  
        total_lines += 1
S_tensor_test = torch.tensor(S_test, dtype=torch.float32).unsqueeze(0).to(device)
S_tensor_test = S_tensor_test.unsqueeze(0)

P_test =[]
inner_list = []
with open(val_output_file,'r') as file:
    lines = file.readlines()
for i in range(0, int(total_lines/flatten_len)):
    inner_list = []
    for j in range(0, upscaling_times*flatten_len):
        ts,byte_count,packet_count = lines[i*upscaling_times*flatten_len+j].strip().split(',')
        ts, byte_count, packet_count = float(ts), float(byte_count), float(packet_count)
        # P_processed.append([byte_count, packet_count])
        inner_list.append(byte_count)
    P_test.append(inner_list)
P_tensor_test = torch.tensor(P_test, dtype=torch.float32).unsqueeze(0).to(device)
print(S_tensor_test.shape, P_tensor_test.shape)
print("input:", np.array(S_tensor_test.detach().cpu()).round(2))
print("y_true:", np.array(P_tensor_test.detach().cpu()).round(2))
input_val_mean = S_tensor_test.mean(dim=2, keepdim = True)
input_val_std =  S_tensor_test.std(dim=2, keepdim = True)
S_tensor_test = (S_tensor_test - input_val_mean) / (input_val_std/ 10.0)

output_val_mean = P_tensor_test.mean(dim=2, keepdim = True)
output_val_std =  P_tensor_test.std(dim=2, keepdim = True)
P_tensor_test = (P_tensor_test - output_val_mean) / (output_val_std/10.0)

output = model(S_tensor_test,P_tensor_test) 
output = output * output_val_std + output_val_mean
output = output.cpu()

P_tensor_test = P_tensor_test * (output_val_std/10.0) + output_val_mean

print("output:", np.array(output.detach().cpu()).round(2))
with open('/root/traffic_recovery/code/result/y_true_0.1s.txt', 'w') as file:
    for batch_idx in range(P_tensor_test.shape[0]):
        for row_idx in range(P_tensor_test.shape[1]):
            row_data = np.array(P_tensor_test[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')

with open('/root/traffic_recovery/code/result/y_hat_0.1s.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')

with open('/root/traffic_recovery/code/result/y_0.1s.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')