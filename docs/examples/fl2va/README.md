# fl2va — keyframe-conditioned generation

The video starts from an image you provide. The canvas adopts that image's aspect ratio, the
keyframe goes through the Qwen3-VL vision tower (as a `<Picture 1>` reference in the prompt) and
through the video VAE encoder (as conditioning rows that anchor the denoising loop), and the
prompt says what happens next.

---

## The flying 2CV — does the prompt still win over the image?

The interesting question about keyframe conditioning is not whether the first frame matches. It
is whether the model still *obeys the prompt* when the prompt contradicts the image. So: a real
photograph the model had never produced, of a car that is unambiguously **parked** — and an
instruction to make it **fly**.

### Input

![input](2cv-input.jpg)

A 1920×1280 photograph (borrowed from [ltx-video-swift-mlx](https://github.com/VincentGourbin/ltx-video-swift-mlx)'s
image-to-video example). Nothing in it suggests motion: the car sits on gravel, wheels straight,
against a hedge.

### Prompt

Written in one French sentence, then rewritten into H3's Context-IR format by the local Gemma 4
stage — the image analysis is what lets it say "the medium side profile shot from the reference
image holds":

```bash
minimax-h3 enhance "La 2CV rouge décolle lentement du sol, s'élève au-dessus des arbres et \
  s'envole dans le ciel, les roues tournant dans le vide." --image 2cv-input.jpg -f 124
```

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Cinematic, medium side profile shot, the vibrant red 2CV
lifts slowly off the ground, rises above the trees, and flies into the sky, the wheels spinning in
mid-air. The camera tracks the vehicle's ascent.

overall_soundscape: The sound of the tires spinning on the gravel fades as the vehicle gains
altitude, replaced by the rush of wind as it flies.

non_diegetic_music: A sweeping orchestral score begins as the car lifts, building tension with
strings and brass as it ascends into the frame.
```

```bash
minimax-h3 generate "$(cat 2cv-fly.prompt.txt)" --image 2cv-input.jpg \
  -W 576 -H 384 -f 124 -s 25 --seed 0 \
  --transformer-quant qint8 --text-encoder-quant qint8 -o 2cv-fly.mp4
```

### Result

▶ [2cv-fly-576x384.mp4](2cv-fly-576x384.mp4) — 576×384, 124 frames (5.2 s), 24 sigma steps,
qint8 throughout, 1 h 01 on an M3 Max.

![frames 0 / 30 / 60 / 123](2cv-fly-contact-sheet.png)

Reading the four frames (0, 30, 60, 123): the photograph is reproduced — stickers on the rear
quarter window, chrome hubcaps, gravel — then the car leaves the ground, then it is above the
treeline with the background blurred by motion, then it is in open sky.

**What this demonstrates**: conditioning anchors the scene without freezing it. A weaker
conditioning would have lost the 2CV's identity somewhere along the climb; a stronger one would
have kept the car parked.

**Where it fails, honestly**: in the last second, against a featureless white sky, the body loses
coherence — proportions drift and spurious wheels appear. The model has no scene left to hold on
to. More sigma steps and a larger canvas are the obvious levers; this run used 24 steps at
576×384 to keep it under the hour.

The soundtrack follows the action too: peak −4.7 dBFS here versus −27 dBFS for the snowy fox
scene below. H3 mixes faithfully to what the prompt describes — quiet scenes decode quiet.

---

## Fox — quality run at the full canvas

![frame](fox-i2va-frame.png)

▶ [fox-i2va-1344x768.mp4](fox-i2va-1344x768.mp4) — 1344×768 (H3's maximum canvas), 124 frames,
29 sigma steps, qint8, 8 h 50 on an M3 Max.

Keyframe: [`../t2va/fox-frame.png`](../t2va/fox-frame.png), itself a frame from an earlier t2va
run — which makes it an *easy* keyframe, already inside the model's own distribution. That is
exactly why the 2CV test above matters more.

Note: this run predates the switch of the vision tower to bf16 (see the `conditioner-fl2va`
parity probe), so re-running it today would give a slightly different — and more
release-faithful — result.
