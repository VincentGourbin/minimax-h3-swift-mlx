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
- MiniMax's full prompt guides ship with the checkpoint:
  `$H3_MODELS_DIR/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md` (and `_ref_en.md`).
- Longer term: build a local Context-IR substitute (e.g. gemma-4-swift-mlx rewriting user
  prompts into this format), like the hosted H3-Context-IR does.

---

## Checked against MiniMax's official skill (2026-08-18)

MiniMax has since published prompt-writing skills in the H3 repo
([`skills/h3-prompt-writing`](https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills), plus
eight genre-specific ones: product ads, 3D animation shorts, papercraft explainers, brand promos,
music-video subtitles, co-op game intros, paper collage, hand-drawn/live-action blends). Comparing
it to what this playbook reverse-engineered from the reproducible cases:

**Confirmed, no change needed.** The three core fields and their exact order
(`integrated_multimodal_description` → `overall_soundscape` → `non_diegetic_music`), the five modes
(T2VA / I2VA / FL2VA / L2VA / Ref2VA), `[Shot N]` structure, and the rule that reference labels
(`<Picture 1>`, `<Video 1>`, `<Audio 1>`) must stay identical across every section — which is
exactly the failure the `--enhance-prompt` keyframe path was fixed to prevent.

**New rules worth adopting:**

- **Duration alignment**: the description's own timeline must match the requested length, stated as
  4–15 s. Our CLI floor is 5 s (`--allow-short-video` below that), so the usable band agrees.
- **Language preservation**: sections in English, but dialogue, lyrics and on-screen text stay in
  their original language.
- **Keyframe connection** (I2VA / FL2VA / L2VA): explicitly state *how* the supplied frame joins
  the timeline, rather than describing the frame and the action separately.
- **Avoid**: plot summaries, unresolved reference labels, misaligned timing — and *abstract
  descriptors*.

**And one direct conflict with our own practice.** The skill's avoid-list names **"cinematic"** as
an abstract descriptor to keep out of prompts. Every published example in this repo opens with it
("Cinematic, medium side profile shot, …"), and `ContextIREnhancer.systemPrompt` instructs Gemma to
open `[Shot 1]` with a style word, offering "Cinematic" first. Those runs came out well, so this is
not a bug — but it is an untested habit contradicting the model author's guidance, and the honest
resolution is a same-seed A/B (identical prompt with and without the opening descriptor) rather
than editing the enhancer on authority. Until that runs, treat the style opener as unvalidated.

**For whoever picks up Ref2VA** (out of scope today): it takes six sections in this order —
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`,
`non_diegetic_music`.
