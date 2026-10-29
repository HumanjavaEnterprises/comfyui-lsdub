"""Vocal nodes. AUDIO is ComfyUI's {"waveform": (B, C, N), "sample_rate": int}."""
import os
import torch
import torchaudio.functional as AF
import folder_paths
import comfy.model_management as mm
from .lsdub_nodes.core import vocal_split as vs, lead_vocal as lv

folder_paths.add_model_folder_path('audio_separation', os.path.join(folder_paths.models_dir, 'audio_separation'))


def _to_stereo44(audio):
    w, sr = audio['waveform'][0].float(), int(audio['sample_rate'])
    if w.shape[0] == 1: w = w.repeat(2, 1)
    w = w[:2]
    return AF.resample(w, sr, vs.SR) if sr != vs.SR else w


def _audio(x):
    return {'waveform': x[None].float().cpu(), 'sample_rate': vs.SR}


class LSDubVocalSplit:
    """mix -> vocals + instrumental (Mel-Band RoFormer, MIT vocal weights)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'audio': ('AUDIO',),
                             'model': (folder_paths.get_filename_list('audio_separation'),),
                             'precision': (['fp32', 'bf16'], {'default': 'fp32'})}}
    RETURN_TYPES = ('AUDIO', 'AUDIO'); RETURN_NAMES = ('vocals', 'instrumental')
    FUNCTION = 'run'; CATEGORY = 'LSDub/audio'

    def run(self, audio, model, precision):
        dev = mm.get_torch_device()
        m = vs.load(folder_paths.get_full_path('audio_separation', model), dev,
                    torch.bfloat16 if precision == 'bf16' else torch.float32)
        mix = _to_stereo44(audio); v = vs.separate(mix, m)
        return (_audio(v), _audio(mix - v))


class LSDubLeadVocal:
    """vocals -> lead + backing + lead activity CSV (time_s, lead_active, f0_hz) for the mouth drive."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'vocals': ('AUDIO',),
                             'share': ('FLOAT', {'default': 0.35, 'min': 0.05, 'max': 1.0, 'step': 0.01}),
                             'floor_db': ('FLOAT', {'default': -40.0, 'min': -80.0, 'max': -10.0, 'step': 1.0})}}
    RETURN_TYPES = ('AUDIO', 'AUDIO', 'STRING'); RETURN_NAMES = ('lead', 'backing', 'lead_activity_csv')
    FUNCTION = 'run'; CATEGORY = 'LSDub/audio'

    def run(self, vocals, share, floor_db):
        d = lv.extract(_to_stereo44(vocals).numpy(), share, floor_db)
        rows = ['time_s,lead_active,f0_hz'] + [f"{t:.3f},{a:.4f},{f:.1f}" for t, a, f in zip(d['t'], d['active'], d['f0'])]
        return (_audio(torch.from_numpy(d['lead'])), _audio(torch.from_numpy(d['backing'])), '\n'.join(rows))


NODE_CLASS_MAPPINGS = {'LSDubVocalSplit': LSDubVocalSplit, 'LSDubLeadVocal': LSDubLeadVocal}
NODE_DISPLAY_NAME_MAPPINGS = {'LSDubVocalSplit': 'LSDub Vocal Split (RoFormer)', 'LSDubLeadVocal': 'LSDub Lead Vocal'}
