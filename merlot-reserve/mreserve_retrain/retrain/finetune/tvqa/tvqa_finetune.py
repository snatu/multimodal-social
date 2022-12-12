"""
Finetunes TVQA.

I used a v3-32 for this (it's more expensive than VCR due to the use of video + sound data)
"""

import sys

sys.path.append('../../')
import yaml
from datetime import datetime
import pytz
import jax
import jax.numpy as jnp
from pretrain.dataloader import input_fn_builder, MASK, encoder, AUDIOSPAN, MASKVIDEO
from finetune.common_dataloader import finetune_input_fn_builder, finetune_val_input_fn_builder
from mreserve.modeling import MerlotReserve, one_hot_pool, unit_normalize

from flax.training import train_state
from flax import jax_utils
import flax.linen as nn
from finetune.optimization import construct_finetuning_train_state, finetune_train_step
from mreserve.checkpoint import save_checkpoint, load_checkpoint, bf16_to_f32, f32_to_bf16
import argparse
import pandas as pd
import numpy as np
from flax.core.frozen_dict import freeze
from copy import deepcopy
import clu.parameter_overview
import functools
import time
import os
from dotenv import load_dotenv
import math
import logging
from jax.experimental.host_callback import call, id_print


load_dotenv('../../.env')

jax.config.update('jax_log_compiles', True)
is_on_gpu = any([x.platform == 'gpu' for x in jax.local_devices()])
print('JAX process: {} / {}. Local devices {}. Using {}'.format(
    jax.process_index(), jax.process_count(), jax.local_devices(), 'GPU' if is_on_gpu else 'TPU'), flush=True)

parser = argparse.ArgumentParser(description='Train model!')

parser.add_argument(
    'pretrain_config_file',
    help='Where the config.yaml is located',
    type=str,
    default="../../../pretrain/configs/base.yaml",
)
parser.add_argument(
    'ckpt',
    help='checkpoint to use',
    type=str,
    default="../../../../base_resadapt",
)
parser.add_argument(
    '-lr',
    help='lr',
    type=float,
)
parser.add_argument(
    '-ne',
    help='ne',
    type=int,
    default=5,
)
parser.add_argument(
    '-output_grid_h',
    help='output_grid_h',
    type=int,
    default=12,
)
parser.add_argument(
    '-output_grid_w',
    help='output_grid_w',
    type=int,
    default=20,
)
parser.add_argument(
    '-output_name',
    help='output_name',
    type=str,
    default='',
)
parser.add_argument(
    '-wandb_name',
    help='wandb_name',
    type=str,
    default='merlotreserve-siq-retrain',
)
parser.add_argument(
    '-percent_data',
    help='percent_data',
    type=float,
    default=1.0,
)
parser.add_argument(
    '-wandb_run_name',
    help='wandb_run_name',
    type=str,
    default='tvqa_1.0',
)
parser.add_argument(
    '-output_ext',
    help='output_extension',
    action='store_true',
    default=False,
)
parser.add_argument(
    '-no_wandb',
    help='no_wandb',
    action='store_true',
    default=False,
)
parser.add_argument(
    '-val_batch_size',
    help='val_batch_size -- defaults to 32',
    type=int,
    default=32,
)
parser.add_argument(
    '-scan_minibatch',
    help='scan_minibatch -- basically, if this is true then batch size is 1 but we do gradient accumulation',
    action='store_true',
    default=False,
)
args = parser.parse_args()

# print(f"Loading from {args.config_file}", flush=True)
with open(args.pretrain_config_file, 'r') as f:
    config = yaml.load(f, yaml.FullLoader)

TOTAL_SIZE = 21 * config['data']['num_train_files']
PERCENT_DATASET = args.percent_data
RECORD_PER_FILE = 477

config['data']['train_fns'] = os.path.join(os.environ["TFRECORDS_PATH"], "train{:03d}of833.tfrecord")
config['data']['num_train_files'] = 833 # math.ceil((TOTAL_SIZE * PERCENT_DATASET) / RECORD_PER_FILE) # 256
config['data']['num_answers'] = 4
config['data']['random_scale_max'] = 1.1
config['data']['random_scale_min'] = 1.0
config['data']['num_segments'] = 7

config['device']['batch_size'] = 8
config['device']['prefetch_size'] = 0
config['device']['n_fns_per_cycle'] = 833

NUM_EPOCH = args.ne
# TRAIN_SIZE = 122112

TRAIN_SIZE =  config['data']['num_train_files'] * RECORD_PER_FILE if PERCENT_DATASET < 1.0 else TOTAL_SIZE
steps_per_epoch = TRAIN_SIZE // config['device']['batch_size']

config['optimizer'] = {
    'beta_2': 0.98,
    'eps': 1e-6,
    'learning_rate': args.lr,
    'num_train_steps': NUM_EPOCH * steps_per_epoch,
    'num_warmup_steps': int(0.5 * steps_per_epoch),
    'use_bfloat16_adam': True,
    'weight_decay_rate': 0.1,
    'do_bias_correction': True,
}

config['device']['iterations_per_loop'] = steps_per_epoch
config['data']['lang_seq_len'] = 256
cfg_name = args.pretrain_config_file.split('/')[-1]
seattle_time = pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone('America/Los_Angeles'))
seattle_time = seattle_time.strftime("%Y-%m-%d-%H:%M.%S")

config['device']['output_dir'] = os.path.join(os.environ["OUTPUT_PATH"], cfg_name)
if args.output_name != '':
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], args.output_name)
if args.output_ext:
    ext = "tvqa_"+ str(args.percent_data)+ "_"+str(args.output_grid_h)+'_'+str(args.output_grid_w)
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], ext)
else:
    config['device']['output_dir'] = os.path.join(config['device']['output_dir'], seattle_time)

np.random.seed(123456)
config['model']['output_grid'] = [args.output_grid_h, args.output_grid_w]
ds_train_iter = finetune_input_fn_builder(config, 'tvqa')
# _, dummy_batch = next(ds_train_iter)


config['_ckpt'] = args.ckpt
tags = [cfg_name]
if args.output_name != '':
    tags.append(args.output_name)
if (jax.process_index() == 0) and not args.no_wandb:
    import wandb
    wandb.init(config=config, project=args.wandb_name, entity='snat', name=args.wandb_run_name, notes=f'Loaded from {cfg_name}', tags=tags)
else:
    wandb = None

class MerlotReserveTVQA(MerlotReserve):
    def setup(self):
        super().setup()
        self.proj = nn.Dense(features=1, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(stddev=0.02), name='proj',
                             use_bias=False)

    def __call__(self, batch):

        batch_size, images_per_batch, seq_size, img_dim = batch['images'].shape # TensorShape([1, 7, 540, 768])
        vision_out = self.vision_encoder(batch['images'].reshape(batch_size * images_per_batch, seq_size, img_dim))
        imgs_enc = vision_out['seq_attnpool']
        imgs_enc = imgs_enc.reshape(batch_size, images_per_batch, seq_size // 4, self.hidden_size) # (1, 7, seq_size // 4, 768)
        
        num_ans_per = 1

        batch_size, num_ans_per, joint_seq_len, two_ = batch['textonly_seqs'].shape # (1, 1, 256, 2)
        imgs_enc = imgs_enc.reshape(batch_size, images_per_batch * seq_size // 4, self.hidden_size).repeat(num_ans_per, axis=0) # (1, 1080, 768)

        
        #########################

        # Encode audio
        # Audio clips are provided as [batch_size, num_segments, num_audio_subsegments, audio_seq_len, num_mels]
        batch_size, num_segments, num_audio_subsegments, audio_seq_len, num_mels = batch['audio_clips'].shape # (1, 7, 3, 60, 65)
        audio_enc = self.audio_encoder(batch['audio_clips'].reshape(-1, audio_seq_len, num_mels))['seq_attnpool'] # (21, 6, 768)

        _, audio_token_len, hidden_size = audio_enc.shape # (21, 6, 768)
        num_audio_spans = num_segments * num_audio_subsegments # 21

        audio_enc = audio_enc.reshape(batch_size, num_audio_spans, audio_token_len, hidden_size) # (1, 21, 6, 768)
        audio_enc = audio_enc.repeat(num_ans_per, axis=0) # (1, 21, 6, 768)

        audio_toks = batch['audio_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len) # (1, 256) 
        audio_pointers = (jnp.cumsum((audio_toks == AUDIOSPAN).astype(jnp.int32), -1) - 1) // audio_token_len # (1, 256)
        audio_pointers = audio_pointers % num_audio_spans # (1, 256)
        
        text_toks = batch['textonly_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len)
        
        vision_inputs_text = self.prepare_multimodal_inputs(
            tokens=text_toks,
            token_segment_idx=batch['textonly_seqs'][..., 1].reshape(batch_size * num_ans_per, joint_seq_len),
            vision_input=imgs_enc,
            audio_spans=None,
            audio_pointers=None,
        )
        
        vision_inputs_audio = self.prepare_multimodal_inputs(
            tokens=batch['audio_seqs'][..., 0].reshape(batch_size * num_ans_per, joint_seq_len),
            token_segment_idx=batch['audio_seqs'][..., 1].reshape(batch_size * num_ans_per, joint_seq_len),
            vision_input=imgs_enc,
            audio_spans=audio_enc,
            audio_pointers=audio_pointers,
        )
        
        text_toks = batch['vision_seqs_audio'][..., 0].reshape(batch_size * num_ans_per, -1) # (1,1201)
        audio_toks = batch['vision_seqs_text'][..., 0].reshape(batch_size * num_ans_per, -1) # (1,1201) 
        
        #############################################################################################################
        
        real_bsizes = [vision_inputs_audio['x'].shape[0], vision_inputs_text['x'].shape[0]]
        x = jnp.concatenate([vision_inputs_audio['x'], vision_inputs_text['x']], 0) # (2,1201,768)
        coords = jnp.concatenate([vision_inputs_audio['rotary_coords'], vision_inputs_text['rotary_coords']], 0) # (2,1201,4)
        attnmask = jnp.concatenate([vision_inputs_audio['attention_mask'], vision_inputs_text['attention_mask']], 0) # (2,1201,1201)
        
        keys = ['audio2vision', 'text2vision']
        joint_enc = self.joint_transformer(x, rotary_coords=coords, attention_mask=attnmask)['seq'] # (2,1201,768)
        mm_outputs = {k: z for k, z in zip(keys, jnp.split(joint_enc, np.cumsum(real_bsizes), axis=0))}
        
        mm_outputs['audio2vision'] = mm_outputs['audio2vision'][:, joint_seq_len:] # (1, 945, 768)
        mm_outputs['text2vision'] = mm_outputs['text2vision'][:, joint_seq_len:] # (1, 945, 768)

        # Pool from the right tokens
        is_pool = (audio_toks == MASKVIDEO)[:, joint_seq_len:] # (1,945)
        a2v_cumulative_idx = jnp.cumsum(is_pool.astype(jnp.int32), -1) - 1 # (1,945)
        
        a2v = one_hot_pool(is_pool,
               idx=a2v_cumulative_idx,
               v=mm_outputs['audio2vision'],
               num_segments=images_per_batch * seq_size // 4)['x'].reshape((batch_size * images_per_batch * seq_size // 4, self.hidden_size))
        # 'x': (1, 945, 768), 'idx_oh': (1,945,945), (945, 768)
        
        is_pool_t2v = (text_toks == MASKVIDEO)[:, joint_seq_len:] # (1, 945)
        t2v_cumulative_idx = jnp.cumsum(is_pool_t2v.astype(jnp.int32), -1) - 1 # (1, 945)
        
        t2v = one_hot_pool(is_pool_t2v,
               idx=t2v_cumulative_idx,
               v=mm_outputs['text2vision'],
               num_segments=images_per_batch * seq_size // 4)['x'].reshape((batch_size * images_per_batch * seq_size // 4, self.hidden_size))
        # 'x': (1, 945, 768), 'idx_oh': (1,945,945), (945, 768)
        
        log_scales = jnp.clip(self.scale_params.astype(jnp.float32), a_max=np.log(100.0))
        outputs = {
            'audio2vision': {'x': a2v, 'y': vision_out['target_seq_attnpool'].reshape(batch_size * images_per_batch * seq_size // 4, -1), 'log_scale': log_scales[0]},
            'text2vision': {'x': t2v, 'y': vision_out['target_seq_attnpool'].reshape(batch_size * images_per_batch * seq_size // 4, -1), 'log_scale': log_scales[1]},
        }
        
        for k in outputs:
            temp_to_use = jnp.exp(outputs[k].pop('log_scale') / 2.0)
            for k2 in 'xy':
                outputs[k][k2] = unit_normalize(outputs[k][k2]) * temp_to_use
                if self.use_bfloat16:
                    outputs[k][k2] = outputs[k][k2].astype(jnp.bfloat16)


        return outputs


model = MerlotReserveTVQA.from_config(config)

params = load_checkpoint(args.ckpt)['params']

# Don't need those
for k in ['head', 'span_encoder']:
    params.pop(k, None)
hsz = params['joint_transformer']['final_ln']['bias'].shape[0]
# extra layer
params['proj'] = {'kernel': np.random.randn(hsz, 1).astype(np.float32) * 0.01}
# freeze weights
params = freeze(params)

state, tx_fns = construct_finetuning_train_state(opt_config=config['optimizer'], model=model, params=params)

def train_loss_fn(state, params, batch):
    preds = state.apply_fn({'params': params}, batch)
    loss_info = {}
    dlse = {}

    for c_type, c_dict in preds.items():
        numer_logits = (c_dict['x'] * c_dict['y']).sum(-1) # (945)
        loss_info[c_type] = 0.0

        # For both directions (average the result)
        for k1, k2 in ['xy', 'yx']:
            x = c_dict[k1] # (945,768)
            y = c_dict[k2] # (945,768)

            # Add in extra things that are only valid as targets
            y_allgather = jax.lax.all_gather(y, 'batch').reshape(-1, x.shape[-1]) # (7560, 768)

            print(f"{c_type} {k1}->{k2} dot product sim:  shaped [{x.shape}] -> [{y_allgather.shape}]", flush=True) 
            denom_logits = jnp.einsum('lh,vh->lv', x, y_allgather) # (945,7560)
            
            dlse_key = '_'+c_type+'_'+k1+'_'+k2+'_'+'denom_logits'
            loss_info[dlse_key] = denom_logits
            
            denom_lse = jax.nn.logsumexp(denom_logits.astype(jnp.float32), axis=-1, return_sign=False) # (945)
            loss_info[c_type] += (denom_lse - numer_logits).mean() / 2.0 
            
            dlse_key = '_'+c_type+'_'+k1+'_'+k2+'_'+'denom_lse'
            loss_info[dlse_key] = denom_lse
            dlse_key = '_'+c_type+'_'+k1+'_'+k2+'_'+'numer_logits'
            loss_info[dlse_key] = numer_logits
            
            
    loss = sum([v for k, v in loss_info.items() if not k.startswith('_')])
    return loss, loss_info



p_train_step = jax.pmap(functools.partial(finetune_train_step, loss_fn=train_loss_fn, tx_fns=tx_fns, scan_minibatch=args.scan_minibatch),
                                          axis_name='batch', donate_argnums=(0,1))

def pred_step(state: train_state.TrainState, batch):
    logits_from_audio, logits_from_text = state.apply_fn({'params': state.params}, batch)

    out = {'logprobs_audio': jax.nn.log_softmax(logits_from_audio, axis=-1),
            'preds_audio': jnp.argmax(logits_from_audio, -1),
            'logprobs_text': jax.nn.log_softmax(logits_from_text, axis=-1),
            'preds_text': jnp.argmax(logits_from_text, -1),
            }
    softmax_joint = jax.nn.softmax(logits_from_audio, axis=-1) + jax.nn.softmax(logits_from_text, axis=-1)
    out['preds_joint'] = jnp.argmax(softmax_joint, -1)
    return out


p_pred_step = jax.pmap(pred_step, axis_name='batch', donate_argnums=(1,))


def val_epoch(state: train_state.TrainState):
    """
    perform a validation epoch
    :param state:
    :return:
    """
    val_config = deepcopy(config)
    val_config['data']['val_fns'] = os.path.join(os.environ["TFRECORDS_PATH"], "val{:03d}of008.tfrecord")
    val_config['data']['num_val_files'] = 8
    val_config['data']['do_random_scale'] = False
    val_config['data']['batch_size'] = args.val_batch_size

    val_iter = finetune_val_input_fn_builder(val_config, 'tvqa')

    text_preds = []
    audio_preds = []
    joint_preds = []

    for ids, batch in val_iter:
        val_pred = p_pred_step(state, batch)
        preds_joint = val_pred['preds_joint'].reshape(-1)
        preds_audio = val_pred['preds_audio'].reshape(-1)
        preds_text = val_pred['preds_text'].reshape(-1)

        labels = batch['labels'].reshape(-1)
        for (p_j, p_a, p_t, id_i, label_i) in zip(val_pred['preds_joint'].reshape(-1),
                                                  val_pred['preds_audio'].reshape(-1),
                                                  val_pred['preds_text'].reshape(-1), ids, labels):
            if id_i == 'pad':
                continue
            text_preds.append({'pred': p_t, 'label': label_i, 'id': id_i})
            audio_preds.append({'pred': p_a, 'label': label_i, 'id': id_i})
            joint_preds.append({'pred': p_j, 'label': label_i, 'id': id_i})

    text_preds = pd.DataFrame(text_preds)
    text_preds['is_right'] = text_preds['pred'] == text_preds['label']
    text_acc = text_preds['is_right'].mean()

    audio_preds = pd.DataFrame(audio_preds)
    audio_preds['is_right'] = audio_preds['pred'] == audio_preds['label']
    audio_acc = audio_preds['is_right'].mean()

    joint_preds = pd.DataFrame(joint_preds)
    joint_preds['is_right'] = joint_preds['pred'] == joint_preds['label']
    joint_acc = joint_preds['is_right'].mean()
    return {'text_acc': text_acc, 'audio_acc': audio_acc, 'joint_acc': joint_acc}

train_metrics = []
log_every = config['device'].get('commit_every_nsteps', 50)
time_elapsed = []

# the + 1 is because for some reason it crashes at the end otherwise. why? idk/
for n in range(config['optimizer']['num_train_steps']+100):
    st = time.time()
    id_, batch = next(ds_train_iter)
    state, loss_info, loss = p_train_step(state, batch)

    if jax.process_index() == 0:
        train_metrics.append(jax.tree_map(lambda x: x[0], loss_info))
        jax.tree_map(lambda x: x.copy_to_host_async(), train_metrics[-1])
        
        step_for_logging = n - log_every
        if step_for_logging >= 0:
            train_metrics[step_for_logging] = {k: float(v) for k, v in train_metrics[step_for_logging].items()}
            if not args.no_wandb:
                wandb.log(train_metrics[step_for_logging], step=step_for_logging, commit=(n + 1) % log_every == 0)

        if (n + 1) % config['device']['iterations_per_loop'] == 0:
            print("Done 1 epoch", flush=True)

            save_checkpoint(state, path=config['device']['output_dir'], no_optimizer=True)
            # val_info = val_epoch(state)
            # print(f"Saving @iter {n:03d}.\nInfo: {pd.Series(val_info)}\n~\n", flush=True)
            # if wandb is not None:
            #     wandb.log({k + '_val': v for k, v in val_info.items()}, step=step_for_logging, commit=True)

        time_elapsed.append(time.time() - st)
        if len(time_elapsed) >= 100:
            tsum = sum(time_elapsed)
            print("Completed 100 batches in {:.3f}sec, avg {:.3f} it/sec".format(tsum, 100.0 / tsum), flush=True)
            time_elapsed = []
            

