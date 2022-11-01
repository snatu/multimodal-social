import copy
from .common import *
from .augmentation import drop_nodes, permute_edges, subgraph, mask_nodes
import torch.nn.functional as F

z_conns = [
    ('text', 'text_z', 'z'),
    ('z', 'z_text', 'text'),
    ('audio', 'audio_z', 'z'),
    ('z', 'z_audio', 'audio'),
    ('video', 'video_z', 'z'),
    ('z', 'z_video', 'video'),
    ('z', 'z_z', 'z'),
]
non_mod_nodes = ['z']
def get_loader_solograph_chunk(ds, vad_intervals, dsname, gc):
    print('Chunk graphs')
    print(f'Regenerating graphs for {dsname}')
    keys, intervals = ds[-2:]

    facet=torch.from_numpy(ds[1][:,:,:].transpose(1,0,2))
    words=torch.from_numpy(ds[2][:,:,:].transpose(1,0,2))
    covarep=torch.from_numpy(ds[3][:,:,:].transpose(1,0,2))

    # trim to just word keys
    word_keys = load_pk(join(gc['proc_data'], f'{dsname}_keys_word.pk'))
    idxs = ar([keys.index(elt) for elt in word_keys])
    assert (np.sort(idxs)==idxs).all(), 'word_keys are sorted differently from chunk keys'
    facet = facet[idxs]
    words = words[idxs]
    covarep = covarep[idxs]
    keys = ar(keys)[idxs]
    intervals = ar(intervals)[idxs]

    gc['true_bs'] = gc['bs']# bs refers to number of videos processed at once

    total_data = []
    num_skipped = 0
    num_smaller_ints = 0
    new_keys = []
    for i in tqdm(range(words.shape[0])):
        key = keys[i]
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }

        if not ( (len(data['text']) == len(data['audio'])) and (len(data['text']) == len(data['video'])) ): # skip where missing
            num_skipped += 1
            continue

        if gc['zero_out_video']:
            data['video'][:]=0
        if gc['zero_out_audio']:
            data['audio'][:]=0
        if gc['zero_out_text']:
            data['text'][:]=0
        
        if sum([len(v) for v in data.values()]) == 0:
            continue
        
        # split up into multiple subgraphs using VAD info: each word that begins within the utterance boundaries is part of that utterance
        ints = intervals[i]
        if key not in vad_intervals:
            num_skipped += 1
            continue

        vad_ints = vad_intervals[key][:,:-1] # ignore speaker id for now
        datas = []
        offset = 0
        z_offset = 0
        
        if len(ints) < len(vad_ints): # special case where fewer intervals than utterance boundaries; use intervals - one "word" per utterance
            vad_ints = np.arange(ints.shape[0])
            smaller_ints = True
        else:
            smaller_ints = False

        if len(vad_ints)==1:
            vad_ints[0][:] = [ints.min(), ints.max()]

        max_idx = -1
        for interval in vad_ints:
            if smaller_ints:
                idx = ar([interval])
            else:
                idx = np.where((ints[:,0] >= interval[0]) & (ints[:,0] <= interval[1]))[0]
            if len(idx) == 0:
                continue
            else:
                idx = idx[idx>max_idx] # exclude idxs already allocated
                max_idx = idx.max()
        
            _data = {
                'text': data['text'][idx],
                'text_idx': torch.arange(len(idx)) + offset,

                'audio': data['audio'][idx],
                'audio_idx': torch.arange(len(idx)) + offset,

                'video': data['video'][idx],
                'video_idx': torch.arange(len(idx)) + offset,
                
                'z': data['text'][idx].mean(axis=0)[None,:].repeat(gc['num_agg_nodes'],1),
                'z_idx': torch.arange(gc['num_agg_nodes']) + z_offset,
            }

            offset += len(idx)
            z_offset += gc['num_agg_nodes']
            datas.append(_data)
        
        z_idxs = torch.cat([elt['z_idx'] for elt in datas])
        if len(datas) == 0:
            assert len(datas) > 0

        num_utts = len(vad_ints)

        for j,data in enumerate(datas): # for each speaker turn
            for mod in mods:
                ret = build_time_aware_dynamic_graph_uni_modal(data[f'{mod}_idx'],[], [], 0, all_to_all=gc['use_all_to_all'], time_aware=True, type_aware=True)
                
                if len(ret) == 0: # no data for this modality
                    continue
                elif len(ret) == 1:
                    data[mod, 'pres', mod] = ret[0]
                else:
                    data[mod, 'pres', mod], data[mod, 'fut', mod], data[mod, 'past', mod] = ret

                for mod2 in [modx for modx in mods if modx != mod]: # other modalities
                    ret = build_time_aware_dynamic_graph_cross_modal(data[f'{mod}_idx'],data[f'{mod2}_idx'], [], [], 0, time_aware=True, type_aware=True)
                    
                    if len(ret) == 0:
                        continue
                    if len(ret) == 2: # one modality only has one element
                        if len(data[f'{mod}_idx']) > len(data[f'{mod2}_idx']):
                            data[mod2, 'pres', mod], data[mod, 'pres', mod2] = ret
                        else:
                            data[mod, 'pres', mod2], data[mod2, 'pres', mod] = ret
                    
                    else:
                        if len(data[f'{mod}_idx']) > len(data[f'{mod2}_idx']): # the output we care about is the "longer" sequence
                            ret = ret[3:]
                        else:
                            ret = ret[:3]

                        data[mod, 'pres', mod2], data[mod, 'fut', mod2], data[mod, 'past', mod2] = ret
            
            # get z idxs
            for mod in mods:
                ## connect z to nodes via different modality connections
                data[f'{mod}', f'{mod}_z', 'z'] = get_fc_edges(data[f'{mod}_idx'], data['z_idx'])
                # outgoing as well as incoming
                data['z', f'z_{mod}', f'{mod}'] = data[f'{mod}', f'{mod}_z', 'z'].flip(dims=[0])

            datas[j] = data
        
        # aggregate graph with offsets
        data = {
            k: torch.cat([elt[k] for elt in datas if k in elt], dim=(1 if isinstance(k,tuple) else 0)) 
            for k in datas[0].keys()
        }

        # connect z nodes together
        data['z', 'z_z', 'z'] = get_fc_edges(z_idxs, z_idxs)

        hetero_data = {
            **{k: {'x': v} for k,v in data.items() if 'idx' not in k and not isinstance(k, tuple)}, # just get data on mods
            **{k: {'edge_index': v} for k,v in data.items() if isinstance(k, tuple) },
        }

        hetero_data = HeteroData(hetero_data)
        total_data.append(hetero_data)
        
        if i == 12 and (gc['test'] or gc['graph_test']):
            break
        
        new_keys.append(key)
        
    save_pk(join(gc['proc_data'], f'{dsname}_keys_{gc["gran"]}.pk'), new_keys)
    print('Total: ', words.shape[0])
    print('Num skipped: ', num_skipped)
    print('Num smaller: ', num_smaller_ints)
    # save_pk(dsname, total_data)

    loader = DataLoader(total_data, batch_size=gc['true_bs'], shuffle=True)
    return loader,gc

def get_loader_solograph_word(ds, vad_intervals, dsname, gc):
    print(f'Regenerating graphs for {dsname}')
    keys, intervals = ds[-2:]
    new_keys = []

    facet=torch.from_numpy(ds[1][:,:,:].transpose(1,0,2))
    words=torch.from_numpy(ds[2][:,:,:].transpose(1,0,2))
    covarep=torch.from_numpy(ds[3][:,:,:].transpose(1,0,2))
    gc['true_bs'] = gc['bs']# bs refers to number of videos processed at once

    total_data = []
    num_skipped = 0
    num_smaller_ints = 0
    incorrectly_sorted = []
    for i in tqdm(range(words.shape[0])):
        key = keys[i]
        data = {
            'text': get_masked(words[i]),
            'audio': get_masked(covarep[i]),
            'video': get_masked(facet[i]),
        }

        if not ( (len(data['text']) == len(data['audio'])) and (len(data['text']) == len(data['video'])) ): # skip where missing
            num_skipped += 1
            continue

        if gc['zero_out_video']:
            data['video'][:]=0
        if gc['zero_out_audio']:
            data['audio'][:]=0
        if gc['zero_out_text']:
            data['text'][:]=0
        
        if sum([len(v) for v in data.values()]) == 0:
            num_skipped += 1
            continue
        
        # split up into multiple subgraphs using VAD info: each word that begins within the utterance boundaries is part of that utterance
        
        # assert intervals[i].shape[0]==data['text'].shape[0]

        word_ints = npr(intervals[i],2) # trust intervals for word timings passed through pipeline b/c of contraction splitting in feature extraction
        word_ints = word_ints[:gc['seq_len']] # trim to seqlen

        # TODO: temporary solution; ensure that idx is contiguous and increasing in start time - no words should be skipped; have to sort b/c of below problem coming out of MFA
        sorted_idxs = np.argsort(word_ints, axis=0)[:,0]
        if not np.all(sorted_idxs==np.arange(len(word_ints))): # if not consecutive
            incorrectly_sorted.append(key)
            word_ints = word_ints[sorted_idxs]
            data['text'] = data['text'][sorted_idxs]
        
        vad_ints = vad_intervals[key] # trust original all_utterances from process_MFA (MFA unified with diarization) to give us utterance boundaries

        if gc['graph_test']:
            # https://bit.ly/3tRt8z7
            data['text'] = tens(np.tile(np.arange(5), (6,1)))
            data['audio'] = tens(np.tile(np.arange(6)+5, (6,1)))
            data['video'] = tens(np.tile(np.arange(7)+5, (6,1)))

            word_ints = tens(np.sort(np.concatenate([np.arange(7), np.arange(7)[1:-1]])).reshape(-1,2))
            vad_ints = [ {'boundaries': [0,3.5]}, {'boundaries': [4.1,6]} ]

        # preprocess vad_ints so it's in correct order and has no duplicates (problem with MFA output)
        vad_ints = sorted(list(set([tuple(elt['boundaries']) for elt in vad_intervals[key]])), key=lambda x: x[0])

        offset = 0 # word offset as we split up into utterances
        z_offset = 0 # utterance offset, equal to utt_idx * gc['num_agg_nodes']
        datas = [] # to hold subgraph info for processing in next step

        # edge case: remove any utterances that do not have words in them
        idxs_to_keep = []
        vad_ints = ar(lzip(*vad_ints)).transpose()
        for utt_idx in range(len(vad_ints)):
            start,end = vad_ints[utt_idx]
            idx = np.where( ( (word_ints[:,0] >= start) & (word_ints[:,0] <= end) ) | ( (word_ints[:,1] >= start) &  (word_ints[:,1] <= end) ) )[0]
            if len(idx) > 0:
                idxs_to_keep.append(utt_idx)
        
        vad_ints = vad_ints[idxs_to_keep]

        for utt_idx in range(len(vad_ints)):
            start,end = vad_ints[utt_idx]
            
            # idxs of words in word_ints that correspond to this utterance
            idx = np.where( ( (word_ints[:,0] >= start) & (word_ints[:,0] <= end) ) | ( (word_ints[:,1] >= start) &  (word_ints[:,1] <= end) ) )[0]

            # assert that idx is contiguous, no words skipped.  Because of above sorting, this should be the case.
            assert (np.sort(idx) == idx).all() # a weaker condition: assert increasing
            
            _data = {
                'text': data['text'][idx],
                'text_idx': torch.arange(len(idx)) + offset,

                'audio': data['audio'][idx],
                'audio_idx': torch.arange(len(idx)) + offset,

                'video': data['video'][idx],
                'video_idx': torch.arange(len(idx)) + offset,
                
                'z': data['text'][idx].mean(axis=0)[None,:].repeat(gc['num_agg_nodes'],1),
                'z_idx': torch.arange(gc['num_agg_nodes']) + z_offset,
            }

            assert _data['text'].shape[0] == _data['audio'].shape[0] == _data['video'].shape[0] # assuming aligned data

            offset += len(idx)
            z_offset += gc['num_agg_nodes']

            if _data['text'].shape[0] > 0:
                datas.append(_data)
        
        z_idxs = torch.cat([elt['z_idx'] for elt in datas])

        for j,data in enumerate(datas):
            for mod in mods:
                for mod2 in mods:
                    data[mod, 'pres', mod2] = get_fc_edges_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pres_window'])
                    data[mod2, 'pres', mod] = get_fc_edges_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pres_window'])

                    data[mod, 'past', mod2] = get_fc_edges_pastfut_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pastfut_window'], idx1_earlier=False)
                    data[mod, 'fut', mod2] = get_fc_edges_pastfut_window(data[f'{mod}_idx'], data[f'{mod2}_idx'], window=gc['align_pastfut_window'], idx1_earlier=True)

            # get z idxs
            for mod in mods:
                ## connect z to nodes via different modality connections
                data[f'{mod}', f'{mod}_z', 'z'] = get_fc_edges(data[f'{mod}_idx'], data['z_idx'])
                # outgoing as well as incoming
                data['z', f'z_{mod}', f'{mod}'] = data[f'{mod}', f'{mod}_z', 'z'].flip(dims=[0])

            datas[j] = data
        
        # aggregate graph with offsets
        data = {
            k: torch.cat([elt[k] for elt in datas if k in elt], dim=(1 if isinstance(k,tuple) else 0)) 
            for k in datas[0].keys()
        }

        # connect z nodes together
        data['z', 'z_z', 'z'] = get_fc_edges(z_idxs, z_idxs)

        if gc['graph_test']:
            assert gc['align_pres_window'] == 1 and gc['align_pastfut_window'] == 10
            mod_pres = pairs_to_arr([(0,0), (0,1), (1,0), (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,3), (4,4), (4,5), (5,4), (5,5)])
            mod_fut_mod = pairs_to_arr([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (4,5)]) # TODO: this should be the case, but right now we have a window size 1
            mod_past_mod = pairs_to_arr([(1,0), (2,0), (3,0), (2,1), (3,1), (3,2), (5,4)]) # TODO: this should be the case, but right now we have a window size 1
            mod_z = pairs_to_arr([(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1)])
            z_mod = pairs_to_arr([(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])

            for mod in mods:
                arr1 = data[mod, f'{mod}_z', 'z']
                assert pointers_eq(arr1, mod_z)
                arr1 = data['z', f'z_{mod}', mod]
                assert pointers_eq(arr1, z_mod)

                for mod2 in mods:
                    for rel in ['past', 'pres', 'fut']:
                        arr1 = data[mod, rel, mod2]
                        if rel == 'pres':
                            arr2 = mod_pres
                        elif (rel == 'fut'):
                            arr2 = mod_fut_mod
                        elif (rel == 'past'):
                            arr2 = mod_past_mod
                        else:
                            assert False
                        
                        assert pointers_eq(arr1,arr2), pointers_diff(arr1, arr2)

        hetero_data = {
            **{k: {'x': v} for k,v in data.items() if 'idx' not in k and not isinstance(k, tuple)}, # just get data on mods
            **{k: {'edge_index': v} for k,v in data.items() if isinstance(k, tuple) },
        }

        hetero_data = HeteroData(hetero_data)
        total_data.append(hetero_data)
        
        if i == 12 and (gc['test'] or gc['graph_test']):
            break
            
        new_keys.append(key)
        
    save_pk(join(gc['proc_data'], f'{dsname}_keys_{gc["gran"]}.pk'), new_keys)
    print('Total: ', words.shape[0])
    print('Num skipped: ', num_skipped)
    print('Num smaller: ', num_smaller_ints)
    print('Num incorrectly sorted: ', len(incorrectly_sorted))
    # save_pk(dsname, total_data)

    loader = DataLoader(total_data, batch_size=gc['true_bs'], shuffle=True)
    return loader,gc

class Solograph_HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = gc['gat_conv_num_heads']

        assert self.hidden_channels % self.heads == 0, 'Hidden channels must be divisible by number of heads'

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in mods:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for i in range(num_layers):
            conv = HeteroConv({
                conn_type: GATv2Conv(gc['graph_conv_in_dim'], hidden_channels//self.heads, heads=self.heads, dropout=gc['drop_het'])
                for conn_type in all_connections + z_conns
            }, aggr='mean')
            
            self.convs.append(conv)

        self.pes = {k: PositionalEncoding(gc['graph_conv_in_dim']) for k in mods}

    def forward(self, x_dict, edge_index_dict, batch_dict):
        mod_dict = {k: v for k,v in x_dict.items() if k not in ['z']}
        z_dict = {k: v for k,v in x_dict.items() if k in ['z']}
        mod_dict = {key: self.lin_dict[key](x) for key, x in mod_dict.items()} # update modality nodes
        z_dict['z'] = self.lin_dict['text'](z_dict['z']) # update Z node
        # apply positional encoding
        for m, v in mod_dict.items(): # modality, tensor
            idxs = batch_dict[m]
            assert (idxs==(idxs.sort().values)).all()
            _, counts = torch.unique(idxs, return_counts=True)
            mod_dict[m] = self.pes[m](v, counts)
        
        x_dict = {
            **mod_dict,
            **z_dict,
        }
        
        # Graph convolution 
        for i,conv in enumerate(self.convs):
            x_dict, edge_types = conv(x_dict, edge_index_dict, return_attention_weights_dict={elt: True for elt in all_connections + z_conns})

            attn_dict = {
                k: {
                    edge_type: {
                        'edge_index': edge_index,
                        'edge_weight': edge_weight,
                    }
                    for edge_type, (edge_index, edge_weight) in zip(edge_types[k], v[1])
                } 
                for k, v in x_dict.items()
            }

            x_dict = {key: x[0].relu() for key, x in x_dict.items()}

        # get mean scene rep
        if gc['scene_mean']:
            x = torch.cat([v for k,v in x_dict.items() if k not in non_mod_nodes], axis=0)
            batch_dicts = torch.cat([v for k,v in batch_dict.items() if k not in non_mod_nodes], axis=0)
            # text, audio, video: torch.Size([5419, 80]) torch.Size([5419, 80]) torch.Size([5419, 80])
            x = scatter_mean(x, batch_dicts, dim=0) # 216, 80 
            scene_rep = x # 216, 80

            return scene_rep
        else:
            raise NotImplementedError


class Solograph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.projection_head = nn.Sequential(OrderedDict([
            ('fc0',   nn.Linear(gc['graph_conv_in_dim'], gc['graph_conv_in_dim'])),
            ('drop_1', nn.Dropout(p=gc['drop_1'])),
            ('relu0', nn.ReLU()),
            ('fc1',   nn.Linear(gc['graph_conv_in_dim'], gc['graph_conv_in_dim'])),
            ('drop_2', nn.Dropout(p=gc['drop_2'])),
            ('relu1', nn.ReLU())
        ]))

        self.hetero_gnn = Solograph_HeteroGNN(gc['graph_conv_in_dim'], 1, gc['num_gat_layers'])

    def forward(self, batch):
        # batch: HeteroData object from pytorch_geometric, which mimics nested python dict
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
        # print(batch.metadata) for all node and edge info
        # batch has three parts: x_dict, edge_index_dict and batch_dict
        # x_dict.keys(): dict_keys(['text', 'audio', 'video', 'z']) 
        # batch.edge_index_dict.keys(): 
        # dict_keys([('text', 'pres', 'text'), ('text', 'past', 'text'), ('text', 'fut', 'text'), ('text', 'pres', 'audio'), ('audio', 'pres', 'text'), ('text', 'past', 'audio'), ('text', 'fut', 'audio'), ('text', 'pres', 'video'), ('video', 'pres', 'text'), ('text', 'past', 'video'), ('text', 'fut', 'video'), ('audio', 'past', 'text'), ('audio', 'fut', 'text'), ('audio', 'pres', 'audio'), ('audio', 'past', 'audio'), ('audio', 'fut', 'audio'), ('audio', 'pres', 'video'), ('video', 'pres', 'audio'), ('audio', 'past', 'video'), ('audio', 'fut', 'video'), ('video', 'past', 'text'), ('video', 'fut', 'text'), ('video', 'past', 'audio'), ('video', 'fut', 'audio'), ('video', 'pres', 'video'), ('video', 'past', 'video'), ('video', 'fut', 'video'), ('text', 'text_z', 'z'), ('z', 'z_text', 'text'), ('audio', 'audio_z', 'z'), ('z', 'z_audio', 'audio'), ('video', 'video_z', 'z'), ('z', 'z_video', 'video'), ('z', 'z_z', 'z')])
        x_dict = batch.x_dict # batchsize: 77; ptr: 4; z: 7 

        if gc['scene_mean']:
            # A copy of data for data augmentation
            edge_index_dict_aug = copy.deepcopy(batch.edge_index_dict)

            x_dict_aug = copy.deepcopy(x_dict)
            visited_nodes = [] # keep track of visited node types
            visited_edges = [] # keep track of visited edge types

            # You can use any single augmentations, or use a combination of them
            for modality, x in x_dict_aug.items(): # text torch.Size([77, 768]); audio torch.Size([77, 74]); video torch.Size([77, 768]); z torch.Size([7, 768])
                for edge_index_type, edge_index in edge_index_dict_aug.items():
                    # ('text', 'pres', 'text') torch.Size([2, 217]), ('text', 'past', 'text') torch.Size([2, 460]), ('text', 'fut', 'text') torch.Size([2, 460]), ('text', 'pres', 'audio') torch.Size([2, 217]), ('audio', 'pres', 'text') torch.Size([2, 217]), ('text', 'past', 'audio') torch.Size([2, 460]), ('text', 'fut', 'audio') torch.Size([2, 460]), ('text', 'pres', 'video') torch.Size([2, 217]), ('video', 'pres', 'text') torch.Size([2, 217]), ('text', 'past', 'video') torch.Size([2, 460]), ('text', 'fut', 'video') torch.Size([2, 460]), ('audio', 'past', 'text') torch.Size([2, 460]), ('audio', 'fut', 'text') torch.Size([2, 460]), ('audio', 'pres', 'audio') torch.Size([2, 217]), ('audio', 'past', 'audio') torch.Size([2, 460]), ('audio', 'fut', 'audio') torch.Size([2, 460]), ('audio', 'pres', 'video') torch.Size([2, 217]), ('video', 'pres', 'audio') torch.Size([2, 217]), ('audio', 'past', 'video') torch.Size([2, 460]), ('audio', 'fut', 'video') torch.Size([2, 460]), ('video', 'past', 'text') torch.Size([2, 460]), ('video', 'fut', 'text') torch.Size([2, 460]), ('video', 'past', 'audio') torch.Size([2, 460]), ('video', 'fut', 'audio') torch.Size([2, 460]), ('video', 'pres', 'video') torch.Size([2, 217]), ('video', 'past', 'video') torch.Size([2, 460]), ('video', 'fut', 'video') torch.Size([2, 460]), ('text', 'text_z', 'z') torch.Size([2, 77]), ('z', 'z_text', 'text') torch.Size([2, 77]), ('audio', 'audio_z', 'z') torch.Size([2, 77]), ('z', 'z_audio', 'audio') torch.Size([2, 77]), ('video', 'video_z', 'z') torch.Size([2, 77]), ('z', 'z_video', 'video') torch.Size([2, 77]), ('z', 'z_z', 'z') torch.Size([2, 27])

                    # currently not keeping track of visited edges to perform strong augmentation
                    # Selecting subgraphs according to the chosen modality to augment
                    if (modality in edge_index_type) and ("z" not in " ".join(list(edge_index_type))): # modality matches edge end, do not augment z nodes
                        if gc['permute_edges'] and (edge_index_type not in visited_edges):
                            x_dict_aug[modality], edge_index_dict_aug[edge_index_type] = \
                                permute_edges(x, edge_index, gc["edge_perturb_level"])
                            visited_edges.append(edge_index_type)
                        if gc['mask_nodes'] and (modality not in visited_nodes):
                            x_dict_aug[modality] = mask_nodes(x, gc["node_masking_level"])
                            #visited_nodes.append(modality)
                        if gc['subgraph'] and (modality not in visited_nodes) and (edge_index_type not in visited_edges):
                            x_dict_aug[modality], edge_index_dict_aug[edge_index_type] = \
                                subgraph(x, edge_index, gc["subgraph_level"])
                            #visited_nodes.append(modality)
                        if gc['drop_nodes'] and (modality not in visited_nodes):
                            x_dict_aug[modality], edge_index_dict_aug[edge_index_type] = \
                                drop_nodes(x, edge_index, gc["drop_node_level"])
                            #visited_nodes.append(modality)
            scene_rep_1 = self.hetero_gnn(x_dict, batch.edge_index_dict, batch.batch_dict) # 216, 80; 432, 80; 216, 80
            scene_rep_2 = self.hetero_gnn(x_dict_aug, edge_index_dict_aug, batch.batch_dict) # 216, 80; 432, 80; 216, 80
            # scene_rep_1: [bs, 80] same for scene_rep_2

            # InfoNCE contrastive loss from SimCLR implementation
            # https://github.com/google-research/simclr/blob/master/objective.py
            # Projection head
            scene_rep_1 = self.projection_head(scene_rep_1)
            scene_rep_2 = self.projection_head(scene_rep_2)

            # normalize both embeddings
            scene_rep_1 = F.normalize(scene_rep_1, dim=-1)
            scene_rep_2 = F.normalize(scene_rep_2, dim=-1)

            # Actual loss
            LARGE_NUM = 1e9
            batch_size = scene_rep_1.shape[0]
            masks = F.one_hot(torch.arange(batch_size), batch_size).to(scene_rep_1.device)
            labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(scene_rep_1.device)

            logits_aa = torch.matmul(scene_rep_1, scene_rep_1.T) / gc['temperature']
            logits_aa = logits_aa - masks * LARGE_NUM # stablize training
            logits_bb = torch.matmul(scene_rep_2, scene_rep_2.T) / gc['temperature']
            logits_bb = logits_bb - masks * LARGE_NUM
            logits_ab = torch.matmul(scene_rep_1, scene_rep_2.T) / gc['temperature']
            logits_ba = torch.matmul(scene_rep_2, scene_rep_1.T) / gc['temperature']
            loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                    target=torch.argmax(labels, -1), reduction="none")
            loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                    target=torch.argmax(labels, -1), reduction="none")
            loss = loss_a + loss_b
            return torch.mean(loss)
        else:
            raise NotImplementedError
