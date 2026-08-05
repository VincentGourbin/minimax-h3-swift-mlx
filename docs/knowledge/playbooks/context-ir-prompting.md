---
okf_version: "0.1"
kind: playbook
created: 2026-08-05
---

# Prompting H3 like the official editor (Context-IR format)

H3-Base is trained to consume **H3-Context-IR output**, not free-form prompts. The official
reproducible cases (`scripts/readme/reproducible-768p-*-request.sh` in the HF repo) reveal the
format — a single string with labeled sections:

```
integrated_multimodal_description: [Shot 1] <camera, subject, setting, lighting, motion…>
[Shot 2] At 00:04.500, <cut description…>
overall_soundscape: <explicit sound design: which sources, their character AND their level —
"loud", "deafening", "close-miked", "dominates the mix"…>
non_diegetic_music: <music description, or "none">
```

## Audio loudness: scene-faithful, not a knob (investigated 2026-08-05)

H3 mixes at the loudness the SCENE implies, verified with content probes (all in-regime 768):
a "deafening drum solo" peaks at **0 dBFS**, MiniMax's hyperspace showcase at −3.6 dB, a fox
trotting in snow at −29.9 dB. Quiet scenes genuinely come out whisper-quiet; the whole pipeline
was numerically cleared (parity) before reaching this conclusion. Secondary factor: a short edge
below the trained 768 costs ~6 dB (960×544 → −36 dB for the same fox).

Practice: describe the desired sound and its character in `overall_soundscape:` (that sets the
*content*), and use `--normalize-audio` (generate/mux) to bring quiet ambience scenes to a
comfortable −3 dBFS peak at listening time — the raw dump keeps the faithful mix.

Rules of thumb:
- Always include `overall_soundscape:` with explicit loudness words for anything that should be
  audible; name each sound source.
- `[Shot N]` timecodes (`At 00:04.500`) drive cuts and synchronized audio events.
- `non_diegetic_music: none` if no score is wanted — omitting the section leaves it to chance.
- MiniMax's full prompt guides are local: `/Volumes/Lexar/models/MiniMax-H3/docs/
  VIDEO_PROMPT_WRITING_GUIDE_base_en.md` (and `_ref_en.md`).
- Longer term: build a local Context-IR substitute (e.g. gemma-4-swift-mlx rewriting user
  prompts into this format), like the hosted H3-Context-IR does.
