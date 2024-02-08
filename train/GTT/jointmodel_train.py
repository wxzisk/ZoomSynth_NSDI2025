##################################################
# This file is used to train a joint model including a Transformer and a BiLSTM
#
##################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# input/output path and template name of files 
base_path = '/root/traffic_recovery/data/mawi/'
input_template = 'summary_{}_1s_aligned.txt'
output_template = 'summary_{}_1ms_aligned.txt'


dates = range(1,2)

input_size = 3
hidden_size = 256
output_size = 15
num_layers = 10
num_heads = 4
dropout = 0.1
lr = 0.01
flatten_len = 5
num_epochs = 3000
S_processed = []
P_processed = []

total_lines = 0
num_chunks = 9
S_tensor = None
P_tensor = None

# file array
input_summary_files = [base_path + input_template.format(date) for date in dates]
output_summary_files = [base_path + output_template.format(date) for date in dates]

#Transformer Encoder
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

# BiLSTM
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

# upscaling_times = 10


for input_file, output_file in zip(input_summary_files, output_summary_files):
    with open(input_file,'r') as file:
        lines = file.readlines()
    S_processed = []
    temp_list = []
    total_lines = 0
    if lines:
        line_count = len(lines)
        print("line_count: ", line_count)
    for i in range(0, line_count):
        ts, byte_count, packet_count= lines[i].strip().split(',')
        iat, byte_count, packet_count= float(1), float(byte_count), float(packet_count)
        total_lines += 1
        S_processed.append([iat, byte_count, packet_count])
    S = torch.tensor(S_processed, dtype=torch.float32).unsqueeze(0)
    if S_tensor is None:
        S_tensor = S
    else:
        S_tensor =  torch.cat((S_tensor, S), dim=0)
    

    with open(output_file, 'r') as f:
        lines = f.readlines()
        line_count = len(lines)
        print("line_count: ", line_count)
    P_processed = []
    temp_list = []
    for i in range(0, line_count):
        ts, byte_count, packet_count= lines[i].strip().split(',')
        iat, byte_count, packet_count= float(0.001), float(byte_count), float(packet_count)
        P_processed.append([iat, byte_count, packet_count])
    P = torch.tensor(P_processed, dtype=torch.float32).unsqueeze(0)
    if P_tensor is None:
        P_tensor = P
    else:
        P_tensor = torch.cat((P_tensor, P), dim=0)

batch_size, sequence_length, feature_length = P_tensor.shape
P_tensor = P_tensor.view(batch_size, sequence_length // flatten_len, feature_length * flatten_len)

print(S_tensor.shape, P_tensor.shape) 
print("S_tensor:", S_tensor)
print("P_tensor:", P_tensor)
print(f"finished loading a new file")

print("data loaded, processing...")


input_mean = S_continuous_features.mean(dim=1, keepdim = True)
input_std =  S_continuous_features.std(dim=1, keepdim = True) + 1e-6
S_continuous_features = (S_continuous_features - input_mean) / (input_std)
output_mean = P_continuous_features.mean(dim=1, keepdim = True)
output_std =  P_continuous_features.std(dim=1, keepdim = True) + 1e-6
P_continuous_features = (P_continuous_features - output_mean) / (output_std)
S_tensor = torch.cat((S_continuous_features, S_binary_features), dim=2)
P_tensor = torch.cat((P_continuous_features, P_binary_features), dim=2)


print("output_mean_shape: ", output_mean.shape, "output_std_shape: ",output_std.shape)
print(input_mean, input_std, output_mean, output_std)
print(torch.isnan(input_mean).any(), torch.isinf(input_mean).any())
print(torch.isnan(input_std).any(), torch.isinf(input_std).any())
print(torch.isnan(output_mean).any(), torch.isinf(output_mean).any())
print(torch.isnan(output_std).any(), torch.isinf(output_std).any())

# slice into batches
S_tensor = S_tensor.view(9,-1, feature_length * flatten_len)
P_tensor = P_tensor.view(9,-1, feature_length * flatten_len)
print(S_tensor.shape, P_tensor.shape) 
input_chunks = torch.chunk(S_tensor, num_chunks, dim=1)
output_chunks = torch.chunk(P_tensor, num_chunks, dim=1)
# print(input_chunks.shape, output_chunks.shape) 

transformer_model = Transformer(input_size, hidden_size, num_layers, num_heads, dropout, output_size)
bilstm_model = BiLSTMModel(output_size, hidden_size, output_size)
model = JointModel(transformer_model, bilstm_model)
# model = Transformer(input_size, hidden_size, num_layers, num_heads, dropout, output_size)
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
# criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

input_test_chunk = None
output_test_chunk= None
output = None

input_chunks = torch.chunk(S_tensor, num_chunks, dim=0)
output_chunks = torch.chunk(P_tensor, num_chunks, dim=0)

for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (input_chunk, output_chunk) in enumerate(zip(input_chunks, output_chunks)):
        optimizer.zero_grad()
        input_chunk = input_chunk.to(device)
        output_chunk = output_chunk.to(device)
        output = model(input_chunk, output_chunk) 
        # loss = criterion(output, output_chunk).to(device)
        loss = wasserstein_distance(output_chunk, output).to(device)
        # loss = my_mse_loss(output, output_chunk).to(device)
        total_loss += loss
        loss.backward()
        optimizer.step()
        
    total_loss = total_loss / num_chunks
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))
    if total_loss.item() <= 0.0001:
        print(f"Stopping training: Loss reached {total_loss.item()} at epoch {epoch+1}")
        break  

# save model param
torch.save(model.state_dict(), '/root/traffic_recovery/pretrained_models/jointmodel_s2ms_mawi.pth')


# model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/jointmodel_header_gen.pth'))

transformer_model = Transformer(input_size, hidden_size, num_layers, num_heads, dropout, output_size)
bilstm_model = BiLSTMModel(output_size, hidden_size, output_size)
joint_model = JointModel(transformer_model, bilstm_model)

# try to load model param
joint_model.load_state_dict(torch.load('/root/traffic_recovery/pretrained_models/jointmodel_s2ms_mawi.pth'))
# joint_model.load_state_dict(torch.load('joint_model_state_dict.pth', map_location=torch.device('cpu')))

output = output.cpu()
output = output * output_std + output_mean
batch_size, sequence_length, feature_length = output.shape
output = output.view(batch_size, sequence_length * flatten_len, feature_length // flatten_len)
# P_tensor_test = P_tensor_test * (output_val_std) + output_val_mean

# print("output:", np.array(output.detach().cpu()).round(2))
# with open('result/jointmodel_y_true_0.001s.txt', 'w') as file:
#     for batch_idx in range(P_tensor_test.shape[0]):
#         for row_idx in range(P_tensor_test.shape[1]):
#             row_data = np.array(P_tensor_test[batch_idx, row_idx].detach().cpu()).round(2)
#             row_str = ' '.join(map(str, row_data))
#             file.write(row_str + '\n')

with open('/root/traffic_recovery/code/result/jointmodel_mawi_ms_wdist.txt', 'w') as file:
    for batch_idx in range(output.shape[0]):
        for row_idx in range(output.shape[1]):
            row_data = np.array(output[batch_idx, row_idx].detach().cpu()).round(2)
            row_str = ' '.join(map(str, row_data))
            file.write(row_str + '\n')