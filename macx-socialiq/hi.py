

import sys
import os
from tqdm import tqdm
from time import time

import torch
from torch import nn

from dataset import SocialIQ
import model



val_dataset = SocialIQ('/ocean/projects/dmr120014p/sulagna/Social-IQ/code/socialiq/', 'val', mods={'ac', 'v', 't'})


val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=32, shuffle=False)




MODE = 'eval'

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

    def forward(self, q, a1, a2, vis):
        if MODE == 'train':
            logits = self.mac(vis, q, [a1, a2])
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(vis, q, [a1, a2])

        return logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)


class LSTM_QAT_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork(kb_dim=768, dim=512, max_step=max_step, self_attention=self_attention,
                                    memory_gate=memory_gate)
        self.mac_running = model.MACNetwork(kb_dim=768, dim=512, max_step=max_step, self_attention=self_attention,
                                            memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, a1, a2, subs):
        if MODE == 'train':
            logits = self.mac(subs, q, [a1, a2])
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(subs, q, [a1, a2])

        return logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)


class LSTM_QAAc_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork(kb_dim=74, dim=100, max_step=max_step, self_attention=self_attention,
                                    memory_gate=memory_gate)
        self.mac_running = model.MACNetwork(kb_dim=74, dim=100, max_step=max_step, self_attention=self_attention,
                                            memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, a1, a2, ac):
        if MODE == 'train':
            logits = self.mac(ac, q, [a1, a2])
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(ac, q, [a1, a2])

        return logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)


class LSTM_QAVT_MAC(nn.Module):
    def __init__(self, max_step, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork_2RUs(visual_dim=2208, dim=512, max_step=max_step, self_attention=self_attention,
                                         memory_gate=memory_gate)
        self.mac_running = model.MACNetwork_2RUs(visual_dim=2208, dim=512, max_step=max_step,
                                                 self_attention=self_attention, memory_gate=memory_gate)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, a1, a2, subs, vis):
        if MODE == 'train':
            logits = self.mac(subs, vis, q, [a1, a2])
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(subs, vis, q, [a1, a2])

        return logits[:, 0].unsqueeze(-1), logits[:, 1].unsqueeze(-1)


class LSTM_QAVTAc_MAC(nn.Module):
    def __init__(self, max_step, classes=2, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork_3RUs(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                         self_attention=self_attention, memory_gate=memory_gate, classes=classes)
        self.mac_running = model.MACNetwork_3RUs(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                                 self_attention=self_attention, memory_gate=memory_gate,
                                                 classes=classes)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, subs, vis, ac):
        if MODE == 'train':
            logits = self.mac(subs, vis, ac, q, answers)
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(subs, vis, ac, q, answers)

        return [logits[:, i].unsqueeze(-1) for i in range(logits.shape[1])]


class LSTM_QAVTAc_MAC_LATE(nn.Module):
    def __init__(self, max_step, classes=2, self_attention=False, memory_gate=False):
        super().__init__()

        self.mac = model.MACNetwork_LateFuse(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                             self_attention=self_attention, memory_gate=memory_gate, classes=classes)
        self.mac_running = model.MACNetwork_LateFuse(visual_dim=2208, ac_dim=74, dim=512, max_step=max_step,
                                                     self_attention=self_attention, memory_gate=memory_gate,
                                                     classes=classes)
        accumulate(self.mac_running, self.mac, 0)

    def forward(self, q, answers, subs, vis, ac):
        if MODE == 'train':
            logits = self.mac(subs, vis, ac, q, answers)
            accumulate(self.mac_running, self.mac)
        else:
            logits = self.mac_running(subs, vis, ac, q, answers)

        return [logits[:, i].unsqueeze(-1) for i in range(logits.shape[1])]


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

binary_qa = torch.load('/ocean/projects/dmr120014p/sulagna/binaryqa.pth')



def validate(num_mods, loss='mse', a4=False):
    corrs = 0
    acc_loss = 0
    binary_qa.eval()

    with torch.no_grad():
        if not a4:
            if num_mods == 0:
                for q, a1, a2, l in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda()])
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
            elif num_mods == 1:
                for q, a1, a2, l, m1 in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
            elif num_mods == 2:
                for q, a1, a2, l, m1, m2 in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(), m2.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
            elif num_mods == 3:
                for q, a1, a2, l, m1, m2, m3 in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda()], m1.cuda(), m2.cuda(), m3.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss)
        else:
            if num_mods == 0:
                for q, a1, a2, a3, a4, l in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()])
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)
            elif num_mods == 1:
                for q, a1, a2, a3, a4, l, m1 in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)
            elif num_mods == 2:
                for q, a1, a2, a3, a4, l, m1, m2 in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda(), m2.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)
            elif num_mods == 3:
                for q, a1, a2, a3, a4, l, m1, m2, m3 in tqdm(val_loader):
                    preds = binary_qa(q.cuda(), [a1.cuda(), a2.cuda(), a3.cuda(), a4.cuda()], m1.cuda(), m2.cuda(), m3.cuda())
                    corrs, acc_loss = com_val(corrs, acc_loss, preds, l, loss, True)

    return corrs / len(val_dataset), acc_loss / len(val_loader)


val_acc, val_loss = validate(3)



print(f'\nAcc: {val_acc}, Loss: {val_loss}\n')




