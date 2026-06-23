"""Debug: compare target vs flow parameters."""
import torch, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
from loom.core import SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.training.dataset import params_to_vector, vector_to_params, CONTINUOUS_KEYS, CATEGORICAL_KEYS
from loom.training.losses import param_loss, multi_resolution_stft_loss
from loom.flow.conditioner import Conditioner
from loom.flow.dit import FlowNetwork
from loom.flow.flow_matching import sample_euler

n_audio = 44100
synth = SubtractiveSynth(SAMPLE_RATE, n_audio).eval()

state = torch.load('data_flow_mr/best_flow.pt', map_location='cpu', weights_only=True)
cond = Conditioner(frontend='multires', n_bins=192, d_model=256, d_cond=512, n_layers=4)
cond.load_state_dict(state['conditioner']); cond.eval()
flow = FlowNetwork(d_model=256, n_dit_blocks=4, nhead=8, d_cond=512)
flow.load_state_dict(state['flow_net']); flow.eval()

torch.manual_seed(999)
params = random_params(1, stage=1)
params.pop('fx_routing', None)
with torch.no_grad(): target_a = synth(params)
pv = params_to_vector(params)

with torch.no_grad():
    cv, al = cond(target_a)
    pred_vec = sample_euler(flow, cv, al, n_steps=20, stage=1)

pred_p = vector_to_params(pred_vec)

# Stage 1: only osc + filter vary
varying_keys = {'osc_pitch', 'osc_detune', 'osc_waveform', 'osc_type',
                'filter_cutoff', 'filter_q', 'filter_type', 'filter_mix'}

print(f"{'Param':<22} {'Target':>8} {'Flow':>8} {'Delta':>8} {'Status':>8}")
print('-' * 60)
for key in CONTINUOUS_KEYS:
    t = params[key].item()
    p = pred_p[key].item()
    d = abs(t - p)
    status = 'VARY' if key in varying_keys else 'FIXED'
    ok = '✓' if d < 0.15 else ('~' if d < 0.3 else 'X')
    print(f'{key:<22} {t:>8.3f} {p:>8.3f} {d:>8.3f} {status:>8} {ok}')

print()
for key, n in CATEGORICAL_KEYS:
    t_cat = params[key].argmax().item()
    p_cat = pred_p[key].argmax().item()
    status = 'VARY' if key in varying_keys else 'FIXED'
    ok = '✓' if t_cat == p_cat else 'X'
    print(f'{key:<22} class={t_cat}    class={p_cat}      {status:>8} {ok}')

# Check audio
pred_p_render = {k: v for k, v in pred_p.items()}
pred_p_render.pop('fx_routing', None)
with torch.no_grad(): pred_a = synth(pred_p_render)
s_loss = multi_resolution_stft_loss(pred_a, target_a).item()
p_loss = param_loss(pred_vec, pv).item()
print(f'\nSpectral loss: {s_loss:.4f}  |  Param loss: {p_loss:.4f}')
print(f'Target RMS: {target_a.norm().item():.4f}  |  Flow RMS: {pred_a.norm().item():.4f}')
