# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""Vocal nodes. AUDIO is ComfyUI's {"waveform": (B, C, N), "sample_rate": int}. Torch is imported on execute."""
import os

import folder_paths
from comfy_api.v0_0_2 import io

folder_paths.add_model_folder_path('audio_separation', os.path.join(folder_paths.models_dir, 'audio_separation'))


def _to_stereo44(audio):
    import torchaudio.functional as AF
    from .lsdub_engine.core import vocal_split as vs
    w, sr = audio['waveform'][0].float(), int(audio['sample_rate'])
    if w.shape[0] == 1: w = w.repeat(2, 1)
    w = w[:2]
    return AF.resample(w, sr, vs.SR) if sr != vs.SR else w


def _audio(x):
    from .lsdub_engine.core import vocal_split as vs
    return {'waveform': x[None].float().cpu(), 'sample_rate': vs.SR}


class LSDubVocalSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubVocalSplit', display_name='LSDub Vocal Split (RoFormer)', category='LSDub/audio',
            description='Split a song into vocals and instrumental with Mel-Band RoFormer (MIT vocal weights in '
                        'models/audio_separation).',
            inputs=[
                io.Audio.Input('audio', tooltip='The song mix (any sample rate; mono is doubled to stereo).'),
                io.Combo.Input('model', options=folder_paths.get_filename_list('audio_separation'),
                               tooltip='Mel-Band RoFormer vocal checkpoint in models/audio_separation '
                                       '(MelBandRoformer_kim_vocals.ckpt).'),
                io.Combo.Input('precision', options=['fp32', 'bf16'], default='fp32',
                               tooltip='fp32 is exact; bf16 is faster with a small quality loss. fp16 overflows in '
                                       'this network, so it is not offered.'),
            ],
            outputs=[io.Audio.Output(display_name='vocals', tooltip='Vocals, stereo 44.1 kHz.'),
                     io.Audio.Output(display_name='instrumental', tooltip='Mix minus vocals, stereo 44.1 kHz.')],
        )

    @classmethod
    def execute(cls, audio, model, precision):
        import torch
        import comfy.model_management as mm
        from .lsdub_engine.core import vocal_split as vs
        m = vs.load(folder_paths.get_full_path('audio_separation', model), mm.get_torch_device(),
                    torch.bfloat16 if precision == 'bf16' else torch.float32)
        mix = _to_stereo44(audio)
        v = vs.separate(mix, m)
        m.to('cpu')                                                     # free the card for the render
        del m
        if hasattr(mm, 'soft_empty_cache'): mm.soft_empty_cache()
        return io.NodeOutput(_audio(v), _audio(mix - v))


class LSDubLeadVocal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='LSDubLeadVocal', display_name='LSDub Lead Vocal', category='LSDub/audio',
            description='Keep only the lead singer: vocals -> lead + backing, and a lead-activity curve that drives '
                        'the mouth (so it does not open on backing lines).',
            inputs=[
                io.Audio.Input('vocals', tooltip='The vocal stem (from LSDub Vocal Split).'),
                io.Float.Input('share', default=0.35, min=0.05, max=1.0, step=0.01,
                               tooltip='How much of the vocal energy must be the lead for a frame to count as lead. '
                                       'Lower keeps more; higher drops more backing.'),
                io.Float.Input('floor_db', default=-40.0, min=-80.0, max=-10.0, step=1.0,
                               tooltip='Quieter than this (dB) counts as silence.'),
            ],
            outputs=[io.Audio.Output(display_name='lead', tooltip='The lead vocal.'),
                     io.Audio.Output(display_name='backing', tooltip='Everything else in the vocal stem.'),
                     io.String.Output(display_name='lead_activity_csv',
                                      tooltip='CSV time_s,lead_active,f0_hz per analysis frame: the mouth drive.')],
        )

    @classmethod
    def execute(cls, vocals, share, floor_db):
        import torch
        from .lsdub_engine.core import lead_vocal as lv
        d = lv.extract(_to_stereo44(vocals).numpy(), share, floor_db)
        rows = ['time_s,lead_active,f0_hz'] + [f"{t:.3f},{a:.4f},{f:.1f}" for t, a, f in zip(d['t'], d['active'], d['f0'])]
        return io.NodeOutput(_audio(torch.from_numpy(d['lead'])), _audio(torch.from_numpy(d['backing'])), '\n'.join(rows))


NODES = [LSDubVocalSplit, LSDubLeadVocal]
