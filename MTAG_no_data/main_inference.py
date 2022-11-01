from torch.autograd import Variable

import traceback
import torch.utils.data as Data
from datetime import datetime
import json
import os
import sys
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import gc as g
from graph_model.iemocap_inverse_sample_count_ce_loss import IEMOCAPInverseSampleCountCELoss
from model import NetMTGATAverageUnalignedConcatMHA
import logging
import util
import pathlib
import random
from arg_defaults import defaults
from consts import GlobalConsts as gc

from models.mylstm import MyLSTM
from models.social_iq import *
from models.alex_utils import *
from models.common import *
from models.graph_builder import construct_time_aware_dynamic_graph, build_time_aware_dynamic_graph_uni_modal, build_time_aware_dynamic_graph_cross_modal
from models.global_const import gc
from models.mosi import*
import sys; sys.path.append('/home/shounak_rtml/11777/Standard-Grid'); import standard_grid


import gc as g
from sklearn.metrics import accuracy_score

SEEDS = list(range(100))

ie_emos = ["Neutral", "Happy", "Sad", "Angry"]

def count_params(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def debug_mem():
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    r = torch.cuda.memory_reserved(0) / 1e9
    a = torch.cuda.memory_allocated(0) / 1e9
    f = r-a  # free inside reserved
    print('-- Debug Mem --')
    print(f'Allocated:{a:.4f}')
    print(f'Reserved:{r:.4f}')
    print(f'Free:{f:.4f}')
    print('-- --')

train_loader, dev_loader = None, None
def train_model_social():
    global train_loader, dev_loader, gc

    if gc['net'] == 'graphqa':
        from models.graphqa import get_loader_solograph, Solograph
    
    elif gc['net'] == 'factorized':
        if gc['gran'] == 'chunk':
            print("AAA")
            from models.factorized import get_loader_solograph_chunk as get_loader_solograph
        else:
            print("BBB")
            from models.factorized import get_loader_solograph_word as get_loader_solograph
            from models.factorized import get_loader_solograph_word_val as get_loader_solograph_val
            
        from models.factorized import Solograph
    else:
        assert gc['net'] == 'factorized', f'gc[net]needs to be factorized but is {gc["net"]}'

    model = torch.load("f2fcl.pth")#Solograph()
    
    # if gc["pretrain_finetune"]: # Self-supervised uses file is for fine-tune
    #     print("Running finetuning from pretrained model")
    #     assert (gc['net'], gc['dataset']) == ('factorized', 'social') # only implemented on factorized social iq
    #     # Load pretrained contrastive learning weights,
    #     # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    #     print("Loading pretrained model from " + gc['model_path'])
    #     pretrained_dict = torch.load(gc['model_path'])
    #     model_dict = model.state_dict()
    #     # Load everything except the projection head
    #     # 1. filter out unnecessary keys
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     # 2. overwrite entries in the existing state dict
    #     model_dict.update(pretrained_dict) 
    #     # 3. load the new state dict
    #     model.load_state_dict(model_dict)

    # else:
    #     print("Running supervised")

    if train_loader is None: # cache train and dev loader so skip data loading in multiple iterations
        print('Building loaders for social')
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        # These videos have some issues in training set

        bads=['f5NJQiY9AuY','aHBLOkfJSYI']
        folds=[trk,dek]
        for bad in bads:
            for fold in folds:
                try:
                    fold.remove(bad)
                except:
                    pass
        print(len(trk))
        print(len(dek))
        print(dek)
        dek=[dek[0]]
        print(dek)
        #preloaded_train=process_data(trk, 'train', gc)
        preloaded_dev=process_data(dek, 'dev', gc)
        
        #print(preloaded_dev)
        # preloaded_dev2=np.asarray(list(preloaded_dev))
        # preloaded_dev3=np.asarray(list(preloaded_dev2[1]))
        # print("OP",preloaded_dev3[1].shape)
        intervals2= [len(a) for a in preloaded_dev]
        print("OP",intervals2)

        #for i in range(10):
        #    print(np.ndim(np.asarray(preloaded_dev[i][0])))
        #replace_inf(preloaded_train[3])
        replace_inf(preloaded_dev[3])

        # TODO: clean up global variables here
        # intervals_subpath = 'vad_intervals_squashed.pk' if gc['gran'] == 'chunk' else 'bert_features.pk'
        intervals = load_pk(join(gc['raw_data'], 'text', 'vad', 'full_bert_feats.pk'))
        intervals2= [len(a) for a in intervals]
        #print(len(intervals2))

        if gc['net'] == 'graphqa':
            #train_loader = get_loader_solograph(preloaded_train, 'social_train')
            #print("THIS")
            dev_loader = get_loader_solograph(preloaded_dev, 'social_dev')
        else:
            print("THAT")
            #rain_loader, gc = get_loader_solograph(preloaded_train, intervals, 'social_train', gc)
            dev_loader, gc = get_loader_solograph_val(preloaded_dev, intervals, 'social_dev', gc)
            #print(dev_loader)
        #del preloaded_train
        del preloaded_dev

    #Initializing parameter optimizer
    model = model.to(gc['device'])
    #print(gc)
    #params= list(model.q_lstm.parameters())+list(model.a_lstm.parameters())+list(model.judge.parameters())
    #optimizer=optim.Adam(params,lr=gc['global_lr'])

    # graph optimizer (only for supervised)
    # if not gc["pretrain_finetune"]: # supervised, not ssl
    #     graph_optimizer = torch.optim.AdamW(
    #         model.hetero_gnn.parameters(),
    #         lr=gc['global_lr'],
    #         weight_decay=gc['weight_decay']
    #     ) 

    # print('Training...')
    metrics = {
        'train_acc_best':  0,
        'train_accs': [],
        'train_losses': [],

        'val_acc_best':  0,
        'val_accs':  [],
        'val_losses': [],
    }

    # epochs_since_new_max = 0 # early stopping
    for i in range(gc['epochs']):
        # if epochs_since_new_max > gc['early_stopping_patience'] and i > 15: # often has a slow start
        #     break
        print ("Epoch %d"%i)
        #model.train()
        # train_losses, train_accs = [],[]

        # train_block = gc['train_block']
        # for batch_i, batch in enumerate(tqdm(train_loader)):
        #     batch = batch.to(gc['device'])

        #     if batch_i == 0:
        #         with torch.no_grad():  # Initialize lazy modules.
        #             correct, incorrect = model(batch)
        #             del correct
        #             del incorrect
        #             torch.cuda.empty_cache()
            
        #     cont = False
        #     for mod in mods:
        #         if not np.any([mod in elt for elt in batch.edge_index_dict.keys()]):
        #             print(mod, 'dropped from train loader!')
        #             cont = True
        #     if cont:
        #         continue
            
        #     for mod in mods:
        #         assert not batch[mod]['x'].isnan().any()
        #         assert not batch[mod]['batch'].isnan().any()
        #         assert not batch[mod]['ptr'].isnan().any()

        #     edges = lkeys(batch.__dict__['_edge_store_dict'])

        #     for (l,name,r) in edges: # check for out of bounds accesses
        #         assert batch[l]['x'].shape[0] >= batch[l,name,r]['edge_index'][0,:].max()
        #         assert batch[r]['x'].shape[0] >= batch[l,name,r]['edge_index'][1,:].max()
            
        #     correct, incorrect = model(batch)

        #     assert not (correct.isnan().any() or incorrect.isnan().any())

        #     correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
        #     incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()

        #     optimizer.zero_grad()
        #     if not gc["pretrain_finetune"]: # supervised, not ssl
        #         graph_optimizer.zero_grad()

        #     loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))
        #     loss.backward()
            
        #     optimizer.step()
        #     if not gc["pretrain_finetune"]: # supervised, not ssl
        #         graph_optimizer.step()

        #     train_losses.append(loss.cpu().detach().numpy())
        #     train_accs.append(calc_accuracy(correct,incorrect))
        #     torch.cuda.empty_cache()

        # train_loss = np.mean(train_losses)
        # train_acc = np.mean(train_accs)
        # metrics['train_accs'].append(train_acc)
        # metrics['train_losses'].append(train_loss)
        # print (f"Train Acc: {train_acc:.4f}")
        # print(f'Train loss:{train_loss:.4f}')

        # metrics['train_acc_best'] = max(train_acc, metrics['train_acc_best'])

        val_losses, val_accs = [], []
        model.eval()
        with torch.no_grad():
            for batch in dev_loader:
                batch = batch.to(gc['device'])
                # print(batch.metadata)
                # print(batch.num_nodes)
                # a=batch.to_dict()
                # print(a)
                correct, incorrect = model(batch)
                #print(correct)
                #print("IN")
                #print(incorrect)
                correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
                incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
                #print(correct_mean.data.cpu().numpy(),"HAA",incorrect_mean.data.cpu().numpy())
                acc = calc_accuracy(correct,incorrect)
                #print(acc,"OO")
                print("Cases",(incorrect_cases(correct,incorrect)))
                val_accs.append(acc)

                loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean)).cpu().detach().numpy()
                val_losses.append(loss)
                
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            metrics['val_accs'].append(val_acc)
            metrics['val_losses'].append(val_loss)
            print (f"Dev Acc: {val_acc:.4f}")
            print(f'Dev loss: {val_loss:.4f}')

            # epochs_since_new_max += 1
            # if val_acc > metrics['val_acc_best']:
            #     epochs_since_new_max = 0
            #     metrics['val_acc_best'] = val_acc
    #torch.save(model,"f2fcl.pth")
    print('Model parameters:', count_params(model))
    metrics['model_params'] = count_params(model)
    return metrics

trk,dek,preloaded_train,preloaded_dev = None, None, None, None
def train_social_baseline():
    global trk,dek,preloaded_train,preloaded_dev
    bs=32
    if trk is None:
        print('Loading data...')
        trk,dek=mmdatasdk.socialiq.standard_folds.standard_train_fold,mmdatasdk.socialiq.standard_folds.standard_valid_fold
        #This video has some issues in training set
        bads=['f5NJQiY9AuY','aHBLOkfJSYI']
        folds=[trk,dek]
        for bad in bads:
            for fold in folds:
                try:
                    fold.remove(bad)
                except:
                    pass
        
        if gc['factorized_key_subset']:
            factorized_keys = load_pk(join(gc['proc_data'], 'social_train_keys_word.pk')) + load_pk(join(gc['proc_data'], 'social_dev_keys_word.pk'))
            trk = lfilter(lambda elt: elt in factorized_keys, trk)
            dek = lfilter(lambda elt: elt in factorized_keys, dek)

        preloaded_train=process_data(trk, 'train', gc)
        preloaded_dev=process_data(dek, 'dev', gc)
        replace_inf(preloaded_train[3])
        replace_inf(preloaded_dev[3])

        trk = preloaded_train[-2]
        dek = preloaded_dev[-2]
    
    q_lstm,a_lstm,t_lstm,v_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn=init_tensor_mfn_modules()
    judge=get_judge().cuda()

    #Initializing parameter optimizer
    params=	list(q_lstm.parameters())+list(a_lstm.parameters())+list(judge.parameters())+\
        list(t_lstm.parameters())+list(v_lstm.parameters())+list(ac_lstm.parameters())+\
        list(mfn_mem.parameters())+list(mfn_delta1.parameters())+list(mfn_delta2.parameters())+list(mfn_tfn.linear_layer.parameters())

    optimizer=optim.Adam(params,lr=gc['global_lr'])

    metrics = {
        'train_acc_best':  0,
        'train_accs': [],
        'train_losses': [],

        'val_acc_best':  0,
        'val_accs':  [],
        'val_losses': [],
    }


    for i in range(gc['epochs']):
        print ("Epoch %d"%i)
        losses=[]
        accs=[]
        ds_size=len(trk)
        for j in range(int(ds_size/bs)+1):

            this_trk=[j*bs,(j+1)*bs]

            q_rep,a_rep,i_rep,v_rep,t_rep,ac_rep,mfn_rep=feed_forward(this_trk,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_train)

            real_bs=float(q_rep.shape[0])

            correct=judge(torch.cat((q_rep,a_rep,i_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
            incorrect=judge(torch.cat((q_rep,i_rep,a_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
    
            correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
            incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
            
            optimizer.zero_grad()
            loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            accs.append(calc_accuracy(correct,incorrect))
            
        print ("Loss %f",numpy.array(losses,dtype="float32").mean())
        print ("Accs %f",numpy.array(accs,dtype="float32").mean())
        metrics['train_accs'].append(numpy.array(accs,dtype="float32").mean())
        metrics['train_losses'].append(numpy.array(losses,dtype="float32").mean())

        with torch.no_grad():
            _losses,_accs=[],[]
            ds_size=len(dek)
            for j in range(int(ds_size/bs)):

                this_dek=[j*bs,(j+1)*bs]

                q_rep,a_rep,i_rep,v_rep,t_rep,ac_rep,mfn_rep=feed_forward(this_dek,q_lstm,a_lstm,v_lstm,t_lstm,ac_lstm,mfn_mem,mfn_delta1,mfn_delta2,mfn_tfn,preloaded_dev)
                real_bs=float(q_rep.shape[0])

                correct=judge(torch.cat((q_rep,a_rep,i_rep,t_rep,v_rep,ac_rep,mfn_rep),1))
                incorrect=judge(torch.cat((q_rep,i_rep,a_rep,t_rep,v_rep,ac_rep,mfn_rep),1))

                correct_mean=Variable(torch.Tensor(numpy.array([1.0])),requires_grad=False).cuda()
                incorrect_mean=Variable(torch.Tensor(numpy.array([0.])),requires_grad=False).cuda()
                loss=(nn.MSELoss()(correct.mean(),correct_mean)+nn.MSELoss()(incorrect.mean(),incorrect_mean))

                _accs.append(calc_accuracy(correct,incorrect))
                _losses.append(loss.cpu().detach().numpy())
            
        print ("Dev Accs %f",numpy.array(_accs,dtype="float32").mean())
        print ("Dev Losses %f",numpy.array(_losses,dtype="float32").mean())
        print ("-----------")
        metrics['val_accs'].append(numpy.array(_accs,dtype="float32").mean())
        metrics['val_losses'].append(numpy.array(_losses,dtype="float32").mean())
    
    metrics['model_params'] = sum(p.numel() for p in params if p.requires_grad)
    metrics['val_acc_best'] = max(metrics['val_accs'])
    metrics['train_acc_best'] = max(metrics['train_accs'])



    return metrics

def get_arguments():
    parser = standard_grid.ArgParser()
    for arg in defaults:
        parser.register_parameter(*arg)

    args = parser.compile_argparse()

    global gc
    for arg, val in args.__dict__.items():
        gc[arg] = val
    
    gc['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    get_arguments() # updates gc

    assert gc['data_path'] is not None, "You havn't provided the dataset path! Use the default one."
    gc['data_path'] = join(gc['data_path'], gc['dataset'])
    gc['proc_data'] = join(gc['data_path'], 'processed')
    gc['csd_data'] = join(gc['data_path'], 'csd')
    gc['raw_data'] = join(gc['data_path'], 'raw')

    start_time = time.time()
    
    if 'mosi' in gc['dataset']:
        train_fn = train_model_mosi

    elif gc['social_baseline']:
        train_fn = train_social_baseline
    else:
        train_fn = train_model_social
    
    all_results = []
    for trial in range(gc['trials']):
        print(f'\nTrial {trial}')
        #util.set_seed(SEEDS[trial])
        best_results = train_fn()
        all_results.append(best_results)

    all_results = {k: [dic[k] for dic in all_results] for k in all_results[0].keys()}
    all_results['model_params'] = all_results['model_params'][0]

    all_results['val_acc_best'] = ar(all_results['val_accs']).max(axis=-1)
    all_results['val_acc_best_mean'] = ar(all_results['val_accs']).max(axis=-1).mean()

    print('Best val accs:', all_results['val_acc_best'])
    print('Best mean val accs:', all_results['val_acc_best_mean'])
    
    elapsed_time = time.time() - start_time
    out_dir = join(gc['out_dir'])
    mkdirp(out_dir)

    save_json(join(out_dir, 'results.json'), all_results)

    with open(join(gc['out_dir'], 'success.txt'), 'w'):
        pass
    
