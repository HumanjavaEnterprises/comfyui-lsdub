# LSDub for ComfyUI

Match the lip-sync to the audio, without re-rendering the whole shot. Built for music videos.
More: https://lsdub.vidbnch.app

## Nodes (0.0.1)

- **LSDub Vocal Split (RoFormer)**: mix -> vocals + instrumental. Weights: Mel-Band RoFormer vocal checkpoint
  (MIT), placed at `models/audio_separation/MelBandRoformer_kim_vocals.ckpt`.
- **LSDub Lead Vocal**: vocals -> lead singer only + backing + a lead-activity curve that drives the mouth.

The lip-sync nodes follow in the next release.

Apache-2.0.

## Links

- Comfy Registry: https://registry.comfy.org/publishers/vveerrgg/nodes/lsdub (`comfy node install lsdub`)
- Project page: https://lsdub.vidbnch.app

## License and attribution

Apache-2.0. If you use this code in your project, keep the `NOTICE` file with it, keep the copyright headers,
note what you changed, and credit LSDub with a link to this repository. To cite it, use `CITATION.cff`
(GitHub's "Cite this repository" button).
