import sys
import os
from tqdm import tqdm
from time import time

import torch
from torch import nn
import numpy as np

from dataset import SocialIQ
import model
import folds
from torch.utils.data import TensorDataset
import csv
#from sentence_transformers import SentenceTransformer






def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


class LSTM_QAV_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork(kb_dim=2208, dim=512, max_step=max_step, self_attention=self_attention,
                                    memory_gate=memory_gate)
        self.mac_running = model.MACNetwork(kb_dim=2208, dim=512, max_step=max_step, self_attention=self_attention,
                                            memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, vis):
        if MODE == 'train':
            logits, intermediate1, intermediate2, actual = self.mac(vis, q, answers)
            accumulate(self.mac_running, self.mac)
        else:
            logits, intermediate1,intermediate2, actual = self.mac_running(vis, q, answers)

        return [logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)], intermediate1, intermediate2, actual


class LSTM_QAT_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork(kb_dim=768, dim=512, max_step=max_step, self_attention=self_attention,
                                    memory_gate=memory_gate)
        self.mac_running = model.MACNetwork(kb_dim=768, dim=512, max_step=max_step, self_attention=self_attention,
                                            memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, subs, corr_ans_txt):
        if MODE == 'train':
            logits, intermediate1, intermediate2, actual = self.mac(subs, q, answers, corr_ans_txt)
            accumulate(self.mac_running, self.mac)
        else:
            logits, intermediate1,intermediate2, actual = self.mac_running(subs, q, answers, corr_ans_txt)

        return [logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)], intermediate1, intermediate2, actual


class LSTM_QAAc_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork(kb_dim=74, dim=100, max_step=max_step, self_attention=self_attention,
                                    memory_gate=memory_gate)
        self.mac_running = model.MACNetwork(kb_dim=74, dim=100, max_step=max_step, self_attention=self_attention,
                                            memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, ac):
        if MODE == 'train':
            logits = self.mac(ac, q, answers)
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(ac, q, answers)

        return logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)


class LSTM_QAVT_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork_2RUs(visual_dim=2208, dim=512, max_step=max_step, self_attention=self_attention,
                                         memory_gate=memory_gate)
        self.mac_running = model.MACNetwork_2RUs(visual_dim=2208, dim=512, max_step=max_step,
                                                 self_attention=self_attention, memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, subs, vis, corr_ans_txt):
        if MODE == 'train':
            logits, intermediate1,intermediate2, actual  = self.mac(subs, vis, q, answers, corr_ans_txt)
            accumulate(self.mac_running, self.mac)
        else:
            logits, intermediate1,intermediate2, actual  = self.mac_running(subs, vis, q, answers, corr_ans_txt)

        return [logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)], intermediate1, intermediate2, actual


class LSTM_QAVTAc_MAC(nn.Module):
    def __init__(self, max_step, classes=2, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork_3RUs(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                         self_attention=self_attention, memory_gate=memory_gate, classes=classes)
        self.mac_running = model.MACNetwork_3RUs(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                                 self_attention=self_attention, memory_gate=memory_gate,
                                                 classes=classes)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, subs, vis, ac, corr_ans_txt):
        if MODE == 'train':
            logits, intermediate1,intermediate2, actual = self.mac(subs, vis, ac, q, answers, corr_ans_txt)
            accumulate(self.mac_running, self.mac)
        else:
            logits, intermediate1,intermediate2, actual = self.mac_running(subs, vis, ac, q, answers, corr_ans_txt)

        return [logits[:, i].unsqueeze(-1) for i in range(logits.shape[1])], intermediate1, intermediate2, actual


class LSTM_QAVTAc_MAC_LATE(nn.Module):
    def __init__(self, max_step, classes=2, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork_LateFuse(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                             self_attention=self_attention, memory_gate=memory_gate, classes=classes)
        self.mac_running = model.MACNetwork_LateFuse(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                                     self_attention=self_attention, memory_gate=memory_gate,
                                                     classes=classes)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, subs, vis, ac, corr_ans_txt):
        if MODE == 'train':
            logits, intermediate1,intermediate2, actual = self.mac(subs, vis, ac, q, answers, corr_ans_txt)
            accumulate(self.mac_running, self.mac)
        else:
            logits, intermediate1,intermediate2, actual = self.mac_running(subs, vis, ac, q, answers, corr_ans_txt)

        return [logits[:, i].unsqueeze(-1) for i in range(logits.shape[1])], intermediate1, intermediate2, actual


# ##########################################################################################
# #
# ##########################################################################################


def com_train(corrs, acc_loss, preds, l, loss, gen_text1, gen_text2, actual1, actual2, a4=False):
    if not a4:
        correct = torch.cat(preds, -1).gather(1, l.cuda().unsqueeze(-1))
        incorrect = torch.cat(preds, -1).gather(1, (1 - l).cuda().unsqueeze(-1))

        if loss == 'ce':
            logits = torch.cat(preds, -1)
            loss = nn.CrossEntropyLoss()(logits, l.cuda())
        else:
            loss = nn.MSELoss()(correct.mean(), torch.tensor(1.0).cuda()) + nn.MSELoss()(incorrect.mean(),
                                                                                         torch.tensor(0.0).cuda())
            +nn.MSELoss()(gen_text1,actual1) + nn.MSELoss()(gen_text2,actual2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        corrs += (correct.detach().cpu() > incorrect.detach().cpu()).sum()
        acc_loss += loss

    else:
        correct = torch.cat(preds, -1).gather(1, l.cuda().unsqueeze(-1))
        inc_ls = []
        for _l in l:
            l_all = {0, 1, 2, 3}
            l_all.remove(_l.item())
            inc_ls.append(list(l_all))
        inc_ls = torch.tensor(inc_ls).transpose(0, 1)
        incorrects = []
        for label in inc_ls:
            incorrects.append(torch.cat(preds, -1).gather(1, label.cuda().unsqueeze(-1)))
        if loss == 'ce':
            logits = torch.cat(preds, -1)
            loss = nn.CrossEntropyLoss()(logits, l.cuda())
        else:
            loss = nn.MSELoss()(correct.mean(), torch.tensor(1.0).cuda()) + nn.MSELoss()(
                torch.cat(incorrects, dim=0).mean(), torch.tensor(0.0).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        corrs += torch.logical_and(torch.logical_and(correct.detach().cpu() > incorrects[0].detach().cpu(),
                                                     correct.detach().cpu() > incorrects[1].detach().cpu()),
                                   correct.detach().cpu() > incorrects[2].detach().cpu()
                                   ).sum()
        acc_loss += loss

    return corrs, acc_loss


    


def train(num_mods, loss='mse', a4=False):
    corrs = 0
    acc_loss = 0
    net.train()

    if not a4:
        if num_mods == 0:
            for q, a1, a2, l  in tqdm(train_loader):
                preds = net(q.cuda(), [a1.cuda(), a2.cuda()])
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss)
                
        elif num_mods == 1:
            for q, a1, a2, l, m1 in tqdm(train_loader):
                
                #correct_ans_txt =torch.as_tensor('Yes, I am stupid')
                #correct_ans_txt = torch.LongTensor([[1,2,4,5],[4,3,2,9]]).cuda()
                #the correct answer text has to be minimum 32 length
                correct_ans_txt = torch.LongTensor([[0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0],[1,2,1,5,7,1,1,0,0,2,0,5,6,1,1,0,0,2,0,6,6,1,2,0,0,2,0,5,6,1,1,0]]).cuda()
                
                # Convert caption to tensor of word ids.
    
                
                
                preds, gen_text1, gen_text2, actual = net(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(),correct_ans_txt)
                
                #print(a1.shape, a2.shape)
                
                #print(gen_text)
                
                    
                
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, gen_text1, gen_text2, actual[0], actual[1])
        elif num_mods == 2:
            for q, a1, a2, l, m1, m2 in tqdm(train_loader):
                correct_ans_txt = torch.LongTensor([[0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0],[1,2,1,5,7,1,1,0,0,2,0,5,6,1,1,0,0,2,0,6,6,1,2,0,0,2,0,5,6,1,1,0]]).cuda()

                preds, gen_text1, gen_text2, actual = net(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(), m2.cuda(),correct_ans_txt)
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, gen_text1, gen_text2, actual[0], actual[1])
        elif num_mods == 3:
            for q, a1, a2, l, m1, m2, m3 in tqdm(train_loader):
                correct_ans_txt = torch.LongTensor([[0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0],[1,2,1,5,7,1,1,0,0,2,0,5,6,1,1,0,0,2,0,6,6,1,2,0,0,2,0,5,6,1,1,0]]).cuda()

                preds, gen_text1, gen_text2, actual = net(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(), m2.cuda(), m3.cuda(),correct_ans_txt)
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, gen_text1, gen_text2, actual[0], actual[1])
    else:
        if num_mods == 0:
            for q, a1, a2, a3, a4, l in tqdm(train_loader):
                preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()])
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, True)
        elif num_mods == 1:
            for q, a1, a2, a3, a4, l, m1 in tqdm(train_loader):
                preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda())
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, True)
        elif num_mods == 2:
            for q, a1, a2, a3, a4, l, m1, m2 in tqdm(train_loader):
                preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda(), m2.cuda())
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, True)
        elif num_mods == 3:
            for q, a1, a2, a3, a4, l, m1, m2, m3 in tqdm(train_loader):
                preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda(), m2.cuda(), m3.cuda())
                corrs, acc_loss = com_train(corrs, acc_loss, preds, l, loss, True)

    return corrs / len(train_dataset), acc_loss / len(train_loader)


def com_val(corrs, acc_loss, preds, l, loss, a4=False):
    if not a4:
        correct = torch.cat(preds, -1).gather(1, l.cuda().unsqueeze(-1))
        incorrect = torch.cat(preds, -1).gather(1, (1 - l).cuda().unsqueeze(-1))

        if loss == 'ce':
            logits = torch.cat(preds, -1)
            loss = nn.CrossEntropyLoss()(logits, l.cuda())
        else:
            loss = nn.MSELoss()(correct.mean(), torch.tensor(1.0).cuda()) + nn.MSELoss()(incorrect.mean(),
                                                                                         torch.tensor(0.0).cuda())

        corrs += (correct.detach().cpu() > incorrect.detach().cpu()).sum()
        acc_loss += loss

    else:
        correct = torch.cat(preds, -1).gather(1, l.cuda().unsqueeze(-1))
        inc_ls = []
        for _l in l:
            l_all = {0, 1, 2, 3}
            l_all.remove(_l.item())
            inc_ls.append(list(l_all))
        inc_ls = torch.tensor(inc_ls).transpose(0, 1)
        incorrects = []
        for label in inc_ls:
            incorrects.append(torch.cat(preds, -1).gather(1, label.cuda().unsqueeze(-1)))
        if loss == 'ce':
            logits = torch.cat(preds, -1)
            loss = nn.CrossEntropyLoss()(logits, l.cuda())
        else:
            loss = nn.MSELoss()(correct.mean(), torch.tensor(1.0).cuda()) + nn.MSELoss()(
                torch.cat(incorrects, dim=0).mean(), torch.tensor(0.0).cuda())

        corrs += torch.logical_and(torch.logical_and(correct.detach().cpu() > incorrects[0].detach().cpu(),
                                                     correct.detach().cpu() > incorrects[1].detach().cpu()),
                                   correct.detach().cpu() > incorrects[2].detach().cpu()
                                   ).sum()
        acc_loss += loss

    return corrs, acc_loss


def validate(num_mods, loss='mse', a4=False):
    corrs = 0
    acc_loss = 0
    net.eval()

    with torch.no_grad():
        if not a4:
            if num_mods == 0:
                for q, a1, a2, l in tqdm(val_loader):
                    preds = net(q.cuda(), [a1.cuda(), a2.cuda()])
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
            elif num_mods == 1:
                for q, a1, a2, l, m1 in tqdm(val_loader):
                    correct_ans_txt = torch.LongTensor([[0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0],[1,2,1,5,7,1,1,0,0,2,0,5,6,1,1,0,0,2,0,6,6,1,2,0,0,2,0,5,6,1,1,0]]).cuda()

                    preds,_, _, _ = net(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(),correct_ans_txt)
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
            elif num_mods == 2:
                for q, a1, a2, l, m1, m2 in tqdm(val_loader):
                    correct_ans_txt = torch.LongTensor([[0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0],[1,2,1,5,7,1,1,0,0,2,0,5,6,1,1,0,0,2,0,6,6,1,2,0,0,2,0,5,6,1,1,0]]).cuda()
                    preds,_, _, _ = net(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(), m2.cuda(),correct_ans_txt)
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
            elif num_mods == 3:
                for q, a1, a2, l, m1, m2, m3 in tqdm(val_loader):
                    correct_ans_txt = torch.LongTensor([[0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0,0,2,0,5,6,1,1,0],[1,2,1,5,7,1,1,0,0,2,0,5,6,1,1,0,0,2,0,6,6,1,2,0,0,2,0,5,6,1,1,0]]).cuda()

                    preds,_, _, _  = net(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(), m2.cuda(), m3.cuda(),correct_ans_txt)
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
        else:
            if num_mods == 0:
                for q, a1, a2, a3, a4, l in tqdm(val_loader):
                    preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()])
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)
            elif num_mods == 1:
                for q, a1, a2, a3, a4, l, m1 in tqdm(val_loader):
                    preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)
            elif num_mods == 2:
                for q, a1, a2, a3, a4, l, m1, m2 in tqdm(val_loader):
                    preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda(), m2.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)
            elif num_mods == 3:
                for q, a1, a2, a3, a4, l, m1, m2, m3 in tqdm(val_loader):
                    preds = net(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda(), m2.cuda(), m3.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)

    return corrs / len(val_dataset), acc_loss / len(val_loader)


# ##########################################################################################
# #
# ##########################################################################################


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '/data/scratch/sargechris'

    train_dataset = SocialIQ(data_path, 'train', mods={ 'v', 't'})
    val_dataset = SocialIQ(data_path, 'val', mods={ 'v', 't'})
    
    
   
    #embedding_torch = torch.load('/ocean/projects/dmr120014p/sulagna/macx-socialiq-1modes-lstm/embedding_tensor.pt')

    
    
    
    batch_size = 32

    #train_loader = torch.utils.data.DataLoader(TensorDataset(train_dataset,embedding_torch),batch_size=batch_size, shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ma = []
    times = []
    for i in range(3):
        print('###### LSTM_QAVT_MAC - 2 class')

        net = LSTM_QAVT_MAC(max_step=12).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        num_epochs = 10
        t_start = time()
        max_acc = 0
        for epoch in range(num_epochs):
            
            print(f'\nEpoch {epoch + 1} ............')
            print('Training ............')
            MODE = 'train'
            train_acc, train_loss = train(2)
            print(f'\nAcc: {train_acc}, Loss: {train_loss}\n')
            print('Evaluating ............')
            MODE = 'eval'
            val_acc, val_loss = validate(2)
            print(f'\nAcc: {val_acc}, Loss: {val_loss}\n')
            if (val_acc > max_acc):
                max_acc = val_acc
            print(f'MaxValAcc: {max_acc}\n')
        ma.append(max_acc)
        torch.save(net, '/ocean/projects/dmr120014p/sulagna/binaryqa_VT_lstm.pth')
        t_end = time()
        print(f'time: {t_end - t_start}')
        times.append(t_end - t_start)
    print(ma)
    print(times)

    # # ##########################################################################################
    # # #
    # # ##########################################################################################
    """
    train_dataset = SocialIQ(data_path, 'train', mods={'v'}, a4=True)
    val_dataset = SocialIQ(data_path, 'val', mods={'v'}, a4=True)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False)
    

    ma = []
    times = []
    for i in range(3):
        print('###### LSTM_QAV_MAC - 4 class')

        net = LSTM_QAV_MAC(max_step=12, classes=4).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        num_epochs = 10
        t_start = time()
        max_acc = 0
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1} ............')
            print('Training ............')
            MODE = 'train'
            train_acc, train_loss = train(1, a4=True)
            print(f'\nAcc: {train_acc}, Loss: {train_loss}\n')
            print('Evaluating ............')
            MODE = 'eval'
            val_acc, val_loss = validate(1, a4=True)
            print(f'\nAcc: {val_acc}, Loss: {val_loss}\n')
            if (val_acc > max_acc):
                max_acc = val_acc
            print(f'MaxValAcc: {max_acc}\n')
        ma.append(max_acc)
        torch.save(net, '/ocean/projects/dmr120014p/sulagna/mcqa_V.pth')
        t_end = time()
        print(f'time: {t_end - t_start}')
        times.append(t_end - t_start)
    print(ma)
    print(times)
    """
