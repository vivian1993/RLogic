import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


class LSTM(nn.Module):
    def __init__(self, relation_num, head_rel_num, emb_size, hidden_size, device, num_layers=1):
        super(LSTM, self).__init__()
        self.emb = nn.Embedding(relation_num, emb_size)
        self.lstm = nn.LSTM(input_size  = emb_size, 
                            hidden_size = hidden_size, 
                            num_layers  = num_layers,
                            batch_first = True)
                
        self.hid2rel = nn.Linear(hidden_size, relation_num)
        
        self.relation_num = relation_num
        self.head_rel_num = head_rel_num
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # input body embedding predict head
        self.body2head = nn.Linear(hidden_size, head_rel_num)
        # MLP
        self.fc_1 = nn.Linear(emb_size * 2, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, head_rel_num)

        
    def forward(self, inputs, hidden):
        # inputs are one-hot tensor of relation
        inputs = self.emb(inputs)    
        out, (h,c) = self.lstm(inputs, hidden)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        # P(B)
        pred = self.hid2rel(out)
        return pred, out, (h,c) 
    
    
    def MLP(self, emb_1, emb_2):
        emb_concat = torch.cat((emb_1, emb_2), dim=1)
        hid = F.relu(self.fc_1(emb_concat))
        out = self.fc_2(hid)
        return out
  

    def attention(self, prob, body_emb):
        """
        Take the prob as attention weights
        Compute the weighted sum of all relation vectors
        """
        batch_size = prob.size(0)
        idx_ = torch.LongTensor(range(self.relation_num)).repeat(batch_size, 1).to(self.device)
        # get all relation embedding
        relation_emb = self.emb(idx_)
        
        relation_emb = torch.cat((relation_emb, body_emb), dim=1)
        prob_ = prob.unsqueeze(1)
        out = prob_ @ relation_emb

        return out
    

    def get_init_hidden(self, batch_size):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                     torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        return hidden


    def predict_head_recursive(self, inputs):
        relation_emb = self.emb(inputs)
        batch_size = inputs.size(0)
        hidden = self.get_init_hidden(batch_size)
        body_hid, (h,c) = self.lstm(relation_emb, hidden)
        length = body_hid.size(1)
        # P(H, B)
        for i_ in range(length-1):
            j_ = i_+1
            # MLP
            if i_ == 0:
                emb_1 = relation_emb[:, i_, :]
                emb_2 = relation_emb[:, j_, :]
                prob = self.MLP(emb_1, emb_2)
                prob_sf = prob.softmax(dim=1)
            else:
                body_emb = body_hid[:, i_, :].unsqueeze(1)
                out = self.attention(prob_sf, body_emb)
                emb_1 = out.squeeze(1)
                emb_2 = relation_emb[:, j_, :]
                prob = self.MLP(emb_1, emb_2)
                prob_sf = prob.softmax(dim=1)
        return prob
    
    
    def predict_head(self, inputs):
        inputs = self.emb(inputs)
        batch_size = inputs.size(0)
        hidden = self.get_init_hidden(batch_size)
        out, (h,c) = self.lstm(inputs, hidden)
        body_emb = out[:, -1, :]
        # P(H, B)
        return self.body2head(body_emb)
        
    
    def get_relation_emb(self, rel):
        return self.emb(rel)


class RNN(nn.Module):
    def __init__(self, relation_num, head_rel_num, emb_size, hidden_size, device, num_layers=1):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(relation_num, emb_size)
        self.rnn = nn.RNN(input_size  = emb_size, 
                            hidden_size = hidden_size, 
                            num_layers  = num_layers,
                            batch_first = True)
                
        self.hid2rel = nn.Linear(hidden_size, relation_num)
        
        self.relation_num = relation_num
        self.head_rel_num = head_rel_num
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # input body embedding predict head
        self.body2head = nn.Linear(hidden_size, head_rel_num)
        # MLP
        self.fc_1 = nn.Linear(emb_size * 2, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, head_rel_num)

        
    def forward(self, inputs, hidden):
        # inputs are one-hot tensor of relation
        inputs = self.emb(inputs)    
        out, hid = self.rnn(inputs, hidden)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        # P(B)
        pred = self.hid2rel(out)
        return pred, out, hid
    
    
    def MLP(self, emb_1, emb_2):
        emb_concat = torch.cat((emb_1, emb_2), dim=1)
        hid = F.relu(self.fc_1(emb_concat))
        out = self.fc_2(hid)
        return out
  

    def attention(self, prob, body_emb):
        """
        Take the prob as attention weights
        Compute the weighted sum of all relation vectors
        """
        batch_size = prob.size(0)
        idx_ = torch.LongTensor(range(self.relation_num)).repeat(batch_size, 1)
        # get all relation embedding
        relation_emb = self.emb(idx_)
        
        
        relation_emb = torch.cat((relation_emb, body_emb), dim=1)
        prob_ = prob.unsqueeze(1)
        out = prob_ @ relation_emb

        return out
    

    def get_init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden


    def predict_head_recursive(self, inputs):
        relation_emb = self.emb(inputs)
        batch_size = inputs.size(0)
        hidden = self.get_init_hidden(batch_size)
        body_hid, (h,c) = self.lstm(relation_emb, hidden)
        length = body_hid.size(1)
        # P(H, B)
        for i_ in range(length-1):
            j_ = i_+1
            # MLP
            if i_ == 0:
                emb_1 = relation_emb[:, i_, :]
                emb_2 = relation_emb[:, j_, :]
                prob = self.MLP(emb_1, emb_2)
                prob_sf = prob.softmax(dim=1)
            else:
                body_emb = body_hid[:, i_, :].unsqueeze(1)
                out = self.attention(prob_sf, body_emb)
                emb_1 = out.squeeze(1)
                emb_2 = relation_emb[:, j_, :]
                prob = self.MLP(emb_1, emb_2)
                prob_sf = prob.softmax(dim=1)
        return prob
    
    
    def predict_head(self, inputs):
        inputs = self.emb(inputs)
        batch_size = inputs.size(0)
        hidden = self.get_init_hidden(batch_size)
        out, _ = self.rnn(inputs, hidden)
        body_emb = out[:, -1, :]
        # P(H, B)
        return self.body2head(body_emb)
        
    
    def get_relation_emb(self, rel):
        return self.emb(rel)



