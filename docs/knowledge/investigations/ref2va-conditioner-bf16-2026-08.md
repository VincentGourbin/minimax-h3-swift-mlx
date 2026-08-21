---
okf_version: "0.1"
kind: investigation
created: 2026-08-21
status: CLOS — port juste, écart = arrondi bf16 amplifié x10, et la revue humaine du 2026-08-21 valide le clip à cosinus 0,911
---

# Le conditionneur ref2va diverge en bf16 (cosinus 0,911 à la profondeur 50)

## Le constat

La sonde `conditioner-ref2va` pleine profondeur échoue là où fl2va passe :

| | fl2va (138 tokens) | ref2va (6092 tokens) |
| --- | --- | --- |
| `hidden_states[50]` rel RMS | 0,0236 | **0,412** |
| cosinus | 0,99972 | **0,911** |

Profil par profondeur de ref2va — et c'est lui qui oriente tout :

```
depth   rel RMS      cosine       max|Δ|      RMS ours/ref
    1   0.35947      0.942238     42.12       1.198 / 1.117
    2   0.76121      0.841271     17.5        1.942 / 1.41
    5   0.028162     0.999606     8.75        2.09  / 2.086
   10   0.015501     0.999880     13          3.926 / 3.924
   40   0.052994     0.998596     14.5        3.596 / 3.596
   45   0.41738      0.908741     1.476e+04   7.371 / 8.073
   50   0.41164      0.911365     1.506e+04   7.735 / 8.435
```

L'explosion à la couche 45 (max|Δ| 1,5e4) est la signature documentée en août pour fl2va —
l'activation massive qui amplifie une dérive d'entrée.

> **Piège de lecture, payé sur place.** J'ai d'abord lu les 36 % de la couche 1 comme « l'entrée
> diffère déjà ». C'est FAUX : le rejeu ci-dessous, avec des features vision *identiques*, affiche
> toujours 0,357 à la couche 1 et 0,758 à la couche 2 avant de retomber à 0,0026 à la couche 5.
> Les deux premières couches sont dominées par quelques activations aberrantes et leur RMS relatif
> ne veut rien dire. **Ne pas diagnostiquer sur les profondeurs 1-2** ; le signal commence à 5.

## Ce que la structure n'explique pas

Tout ce qui a été écrit pour ref2va est EXACT, mesuré :

- ids de tokens de la présentation, `mm_token_type_ids`, tags de lignes H3 : **0,0** sur 6092 tokens
  (donc numérotation par modalité, `<Audio 2>` avant `<Video 1>`, timestamps en round-half-even,
  un pad par token fusionné, double étiquetage AdaLN/mrope — tout juste) ;
- normalisation des références : trois contrats **bit-exacts**, les deux autres à 7e-6 et 2e-7 ;
- tour vision aux DEUX géométries ref2va, **en fp32** : 2,2e-4 à 48×80 et 4,2e-5 à 128×128 ;
- `grid_t=N` ≡ N appels `grid_t=1` : **0,0**.

Et `hidden_states[0]` — les embeddings avec les features vision injectées, AUCUNE couche —
reproduit *exactement* l'erreur des features image (max 2,875 / moyenne 0,0275 / échelle 26,5).
**L'injection, le layout et la présentation n'ajoutent rien.** L'entrée est fausse parce que la
tour vision en bf16 l'est déjà.

## Les quatre coins

Même image, même presentation, tour comparée dans les deux dtypes des deux côtés :

| features image | max\|Δ\| | moyenne\|Δ\| | échelle |
| --- | --- | --- | --- |
| **nous fp32 vs eux fp32** | 0,0017 | **9,3e-6** | 26,8 |
| eux bf16 vs eux fp32 | 1,596 | 0,0192 | 26,8 |
| nous bf16 vs eux fp32 | 2,330 | 0,0251 | 26,8 |
| nous bf16 vs eux bf16 | 2,875 | 0,0278 | 26,8 |

Lecture :

1. **Notre arithmétique est juste** — en fp32 on colle à 9,3e-6 de moyenne.
2. Notre chemin bf16 est **~1,3× plus lossy que celui de torch** (0,0251 contre 0,0192 d'écart à
   la vérité fp32). Suspect naturel : `nn.LayerNorm` de torch remonte le bf16 en fp32 en interne,
   et SDPA accumule en fp32 — à vérifier côté MLX.
3. Mais surtout : **deux implémentations bf16 CORRECTES diffèrent l'une de l'autre d'environ deux
   fois leur erreur individuelle**. 0,0192 + 0,0251 ≈ 0,044 en pire cas, on mesure 0,0278. L'écart
   observé est donc de l'ordre de l'arrondi inévitable, pas d'un défaut catégoriel.

L'erreur croît avec le nombre de tokens par appel de tour — 1,625 sur des blocs vidéo de 960
tokens, 2,875 sur l'image de 4096 — ce qui est la signature d'une accumulation. fl2va n'appelait
la tour que sur 112 tokens : le défaut y était invisible.

## Pourquoi ref2va amplifie ce que fl2va absorbait

La séquence ref2va est à **98,8 % des lignes vision** (6016 sur 6092), contre 112 sur 138 pour
fl2va, et surtout deux ordres de grandeur de plus en absolu. Le modèle développe ses activations
massives aux couches 43-45 et amplifie toute perturbation de l'entrée — mécanisme établi en août.

## Le test qui a tranché

Rejeu calqué sur celui qui avait tranché fl2va : **piloter notre stack texte avec les features
vision bf16 de la RÉFÉRENCE**, plus ses propres taps deepstack, de sorte qu'il ne reste rien de
notre tour dans l'entrée (`parity conditioner-ref2va --quant reference-vision`).

| conditionnement | rel RMS à 50 | cosinus | max\|Δ\| à 45 |
| --- | --- | --- | --- |
| notre tour, bf16 (dtype de la release) | 0,412 | 0,911 | 1,5e4 |
| notre tour, fp32 | 0,251 | 0,968 | 1,1e4 |
| **features de la référence** | **0,0215** | **0,99977** | 512 |

**Notre stack texte est EXACT à 6092 tokens** — 0,99977, soit un poil mieux que les 0,99972 de
fl2va à 138 tokens. Toute la divergence vient de la tour vision.

## Conclusion

1. **Le port est juste.** Structure ref2va exacte (0,0), stack texte exact (0,99977), arithmétique
   de la tour exacte (9,3e-6 en fp32).
2. **La limite est un arrondi irréductible, amplifié ~10×.** Un écart de features de 0,019-0,028 en
   moyenne ressort à 0,25-0,41 de rel RMS à la profondeur 50, concentré sur les cellules
   d'activation massive (max|Δ| ~1e4 à la couche 45). Le modèle est chaotique sur cette entrée.
3. **La barre de 0,03 de fl2va ne se transporte pas.** fl2va : 138 tokens dont 112 vision, un seul
   appel de tour sur 112 tokens. ref2va : 6092 tokens dont **6016 vision** (98,8 %), des appels de
   tour à 960 et 4096 tokens. Aucune implémentation qui ne reproduit pas l'arrondi bf16 de torch
   bit pour bit n'atteindra 0,03 ici.

## Et la règle « toujours bf16 » ?

Elle ne généralise PAS. En fl2va, la tour fp32 donnait cosinus 0,67 contre 0,9997 en bf16, d'où la
règle « fidélité au dtype de la release avant précision par composant ». **Ici c'est l'inverse
mesuré** : fp32 donne 0,968 contre 0,911. Ce qui est cohérent avec les distances — notre fp32 est à
9,3e-6 de la vérité alors que le bf16 de la référence en est à 0,019, donc notre fp32 est plus
proche de ce que la release produit réellement que notre bf16 ne l'est. Le verdict de fl2va était
donc un fait de CE cas (une activation en équilibre instable, que le nudge d'une seule cellule
reproduisait), pas une loi.

**Tranché le 2026-08-21 : le défaut RESTE bf16**, et le clip produit avec a été jugé « impec » sur
le timbre et la synchro labiale — donc 0,911 suffit en pratique et l'A/B fp32 devient une curiosité,
plus un préalable.

**Décision initiale, conservée : le défaut RESTE bf16.** C'est le dtype de la release, c'est ce qui est validé
E2E pour fl2va, et un changement de défaut sur la foi d'une seule métrique est exactement l'erreur
que ce repo a déjà commise quatre fois (métriques qui adoubent des clips que l'oreille ou l'œil
rejettent). Le test qui décide est une comparaison humaine de deux clips ref2va générés, tour bf16
contre tour fp32 — non faite à ce jour.

## Piste secondaire, réelle mais marginale

Notre bf16 est ~1,3× plus lossy que celui de torch (0,0251 contre 0,0192 d'écart à la vérité
fp32). Suspect : `nn.LayerNorm` de torch remonte le bf16 en fp32 en interne, et SDPA accumule en
fp32. Corriger cela rapprocherait 0,0251 de 0,0192 — mais l'écart ours-vs-theirs resterait de
l'ordre de la somme des deux arrondis, donc le gain à la profondeur 50 serait faible (~0,911 vers
~0,92 en ordre de grandeur). À faire pour la propreté, pas comme remède.

## Leçons de méthode

- **Ne pas diagnostiquer sur les profondeurs 1-2.** J'ai lu les 36 % de la couche 1 comme « l'entrée
  diffère » ; le rejeu, à entrée IDENTIQUE, affiche les mêmes 0,357. Ces couches sont dominées par
  quelques activations aberrantes. Le signal commence à la profondeur 5.
- **Les sondes d'isolement de ce repo tournent en fp32 par convention** — précisément le régime où
  un défaut d'accumulation de dtype est INVISIBLE. Les quatre sondes composants de ref2va étaient
  vertes pendant que le conditionneur complet était à 0,911. Une sonde qui isole l'arithmétique ne
  peut pas voir un problème de dtype ; c'est la sonde pleine profondeur qui existe pour ça.
- **Le rejeu « depuis l'état de la référence » tranche là où le raisonnement patine.** Deuxième fois
  qu'il ferme une enquête de conditionneur dans ce repo.
