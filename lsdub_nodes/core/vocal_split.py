#!/usr/bin/env python3
"""vocal_split.py — ML vocal separation (Mel-Band RoFormer) for the lead-vocal drive.

The mouth must follow the LEAD singer only. A DSP centre-extract leaves backing harmonies and reverb in the
"lead" stem, and the loudness drive (sung_dyn) then opens the mouth on lines she is not singing. This module
runs a band-split RoFormer over the mix and returns the vocal stem (and the instrumental as mix - vocals).

Model: Mel-Band RoFormer (Wang et al. 2023), 60 mel bands, 6 x (time transformer, freq transformer), dim 384,
stereo 44.1 kHz. The weights are the MIT-licensed vocal checkpoint; this file is our own torch implementation
of the architecture, written to load that state_dict. Torch + numpy only (no librosa / einops): it runs in the
ComfyUI venv as-is.

  python -m lsdub.core.vocal_split --in mix.wav --out-vocals vox.wav [--out-inst inst.wav] [--ckpt PATH] [--device cuda]

The ComfyUI node (comfyui/nodes_vocal.py) wraps `separate()` on ComfyUI AUDIO dicts.
"""
import argparse, math, os, subprocess
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SR = 44100
CFG = dict(dim=384, depth=6, num_bands=60, dim_head=64, heads=8, n_fft=2048, hop=441, win=2048,
           mask_depth=2, chunk=352800, overlap=2)
DEFAULT_CKPT = os.path.expanduser('~/ComfyUI/models/audio_separation/MelBandRoformer_kim_vocals.ckpt')
CKPT_SHA256 = '87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e'


# --- mel filter bank (Slaney scale + Slaney area norm), only its non-zero pattern decides the bands ---
def _hz_to_mel(f):
    f = np.asanyarray(f, float); f_sp = 200.0 / 3; mel = f / f_sp
    min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp; logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mel)


def _mel_to_hz(m):
    m = np.asanyarray(m, float); f_sp = 200.0 / 3; f = f_sp * m
    min_log_hz = 1000.0; min_log_mel = min_log_hz / f_sp; logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)


def mel_bank(sr, n_fft, n_mels):
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f); ramps = mel_f[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    w[0, 0] = w[0, 1] * 0.25; w[-1, -1] = w[-1, -2] * 0.25          # cover DC and Nyquist
    return w.astype(np.float32)


# --- the network ---
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.scale = dim ** 0.5; self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rotary(nn.Module):
    def __init__(self, dim_head):
        super().__init__()
        self.freqs = nn.Parameter(1.0 / 10000 ** (torch.arange(0, dim_head, 2).float() / dim_head), requires_grad=False)

    def forward(self, x):                                               # x: (..., n, d), pairs interleaved
        n = x.shape[-2]
        ang = torch.arange(n, device=x.device, dtype=torch.float32)[:, None] * self.freqs.float()[None]
        ang = ang.repeat_interleave(2, dim=-1)
        x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return (x.float() * ang.cos() + rot * ang.sin()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__(); inner = heads * dim_head; self.h = heads
        self.rotary_embed = Rotary(dim_head); self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False); self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False))

    def forward(self, x):
        x = self.norm(x); b, n, _ = x.shape
        q, k, v = self.to_qkv(x).view(b, n, 3, self.h, -1).permute(2, 0, 3, 1, 4)
        q, k = self.rotary_embed(q), self.rotary_embed(k)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o * self.to_gates(x).transpose(1, 2).unsqueeze(-1).sigmoid()
        return self.to_out(o.transpose(1, 2).reshape(b, n, -1))


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(dim * mult, dim), nn.Dropout(0.0))

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, heads, dim_head), FeedForward(dim)])])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x; x = ff(x) + x
        return self.norm(x)


def _mlp(dim_in, dim_out, hidden, depth):
    dims = (dim_in, *([hidden] * depth), dim_out); net = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        net.append(nn.Linear(a, b))
        if i < len(dims) - 2: net.append(nn.Tanh())
    return nn.Sequential(*net)


class MelBandRoformer(nn.Module):
    def __init__(self, dim=384, depth=6, num_bands=60, dim_head=64, heads=8, n_fft=2048, hop=441, win=2048,
                 mask_depth=2, **_):
        super().__init__()
        self.stft = dict(n_fft=n_fft, hop_length=hop, win_length=win, normalized=False); self.win = win
        bank = torch.from_numpy(mel_bank(SR, n_fft, num_bands)) > 0                   # (bands, freqs)
        fidx = torch.arange(n_fft // 2 + 1).expand(num_bands, -1)[bank]
        fidx = (fidx[:, None] * 2 + torch.arange(2)).flatten()                          # stereo interleave
        self.register_buffer('freq_indices', fidx, persistent=False)
        self.register_buffer('bands_per_freq', bank.sum(0), persistent=False)
        dims_in = tuple(4 * int(c) for c in bank.sum(1))                                # 2 ch x complex
        self.dims_in = dims_in
        self.layers = nn.ModuleList([nn.ModuleList([Transformer(dim, heads, dim_head), Transformer(dim, heads, dim_head)])
                                     for _ in range(depth)])
        self.band_split = nn.Module()
        self.band_split.to_features = nn.ModuleList([nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in dims_in])
        me = nn.Module(); me.to_freqs = nn.ModuleList([nn.Sequential(_mlp(dim, d * 2, dim * 4, mask_depth), nn.GLU(-1))
                                                       for d in dims_in])
        self.mask_estimators = nn.ModuleList([me])

    def forward(self, audio):                                           # audio: (b, 2, t) -> vocals (b, 2, t)
        b, s, t = audio.shape; audio = audio.float(); wdt = self.band_split.to_features[0][1].weight.dtype
        win = torch.hann_window(self.win, device=audio.device)
        spec = torch.stft(audio.reshape(b * s, t), **self.stft, window=win, return_complex=True)
        spec = torch.view_as_real(spec).reshape(b, s, *spec.shape[-2:], 2)                # b s f t c
        spec = spec.permute(0, 2, 1, 3, 4).reshape(b, -1, spec.shape[-2], 2)             # b (f s) t c
        x = spec[:, self.freq_indices].permute(0, 2, 1, 3).reshape(b, spec.shape[2], -1).to(wdt)  # b t (f c)
        x = torch.stack([m(c) for c, m in zip(x.split(self.dims_in, -1), self.band_split.to_features)], -2)
        T, Fb, D = x.shape[1:]
        for tt, ft in self.layers:
            x = tt(x.transpose(1, 2).reshape(b * Fb, T, D)).reshape(b, Fb, T, D).transpose(1, 2)
            x = ft(x.reshape(b * T, Fb, D)).reshape(b, T, Fb, D)
        mask = torch.cat([m(xb) for xb, m in zip(x.unbind(-2), self.mask_estimators[0].to_freqs)], -1)
        mask = torch.view_as_complex(mask.float().reshape(b, T, -1, 2).permute(0, 2, 1, 3).contiguous())  # b f' t
        cspec = torch.view_as_complex(spec.contiguous())                                   # b (f s) t
        summed = torch.zeros_like(cspec).index_add_(1, self.freq_indices, mask)
        denom = self.bands_per_freq.repeat_interleave(2).clamp(min=1).to(cspec.real.dtype)[None, :, None]
        out = (cspec * summed / denom).reshape(b, -1, s, T).transpose(1, 2).reshape(b * s, -1, T)
        y = torch.istft(out, **self.stft, window=win, length=t)
        return y.reshape(b, s, t)


_MODEL = {}


def load(ckpt=DEFAULT_CKPT, device='cuda', dtype=torch.float32):   # fp16 overflows in this net; bf16 or fp32
    key = (ckpt, str(device), dtype)
    if key in _MODEL: return _MODEL[key]
    m = MelBandRoformer(**CFG)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)                         # tensors only, no pickle code
    m.load_state_dict(sd, strict=True)
    m = m.to(device).eval()
    if dtype != torch.float32: m = m.to(dtype)
    _MODEL.clear(); _MODEL[key] = m
    return m


@torch.inference_mode()
def separate(mix, model, chunk=CFG['chunk'], overlap=CFG['overlap'], batch=4, progress=None):
    """mix: float tensor (2, n) at 44.1 kHz -> vocals (2, n). Overlap-add with a fade window."""
    dev = next(model.parameters()).device
    n = mix.shape[-1]; step = chunk // overlap; fade = chunk // 10
    w = torch.ones(chunk); r = torch.linspace(0, 1, fade); w[:fade] = r; w[-fade:] = r.flip(0)
    pad = chunk - step
    x = F.pad(mix.float(), (pad, pad + chunk))
    starts = list(range(0, x.shape[-1] - chunk + 1, step))
    acc = torch.zeros_like(x); cnt = torch.zeros(x.shape[-1])
    for i in range(0, len(starts), batch):
        ss = starts[i:i + batch]
        seg = torch.stack([x[:, s0:s0 + chunk] for s0 in ss]).to(dev)
        y = model(seg).float().cpu()
        for j, s0 in enumerate(ss):
            acc[:, s0:s0 + chunk] += y[j] * w; cnt[s0:s0 + chunk] += w
        if progress: progress(min(i + batch, len(starts)), len(starts))
    return (acc / cnt.clamp(min=1e-6))[:, pad:pad + n]


def _read(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, np.float32).reshape(-1, 2).T.copy())


def _write(path, x):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', str(x.shape[0]), '-i', '-',
                    '-c:a', 'pcm_s16le', path], input=x.T.contiguous().numpy().astype(np.float32).tobytes(), check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out-vocals', required=True)
    ap.add_argument('--out-inst'); ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bf16', action='store_true', help='half the VRAM; fp32 is the reference')
    a = ap.parse_args(argv)
    mix = _read(a.inp)
    m = load(a.ckpt, a.device, torch.bfloat16 if (a.bf16 and a.device != 'cpu') else torch.float32)
    v = separate(mix, m, progress=lambda i, n: print(f'\r  chunk {i}/{n}', end='', flush=True)); print()
    _write(a.out_vocals, v)
    if a.out_inst: _write(a.out_inst, mix - v)
    print(f'vocals -> {a.out_vocals}' + (f'; instrumental -> {a.out_inst}' if a.out_inst else ''))


if __name__ == '__main__':
    main()
