---
okf_version: "0.1"
kind: pitfall
created: 2026-08-20
---

# MLXNN déplie les clés numériques contiguës en tableau

## Symptôme

Un module Swift dont l'arbre de paramètres reproduit **exactement** les clés du checkpoint charge
quand même en erreur, avec un message qui montre les deux côtés parfaitement d'accord :

```
MLXNN.UpdateError.incompatibleItems(
  path: ["encoder", "block"],
  modules: ["H3AudioVAE", "H3AudioEncoder"],
  item: "module(Block { 0: H3AudioConv1d(...), 1: H3AudioEncoderBlock {...}, ... })",
  value:  "[ [bias: [64], weight: [64, 7, 1]], [block: [...]], ... ]")
```

Les formes correspondent une à une. Ce n'est pas un problème de poids.

## Cause

Un `nn.Sequential` PyTorch nomme ses enfants par leur indice : `encoder.block.0.weight`,
`encoder.block.1.block.0.alpha`, … La transcription naturelle en Swift est une struct dont les
propriétés portent ces indices comme clés :

```swift
final class Block: Module {
    @ModuleInfo(key: "0") var convIn: H3AudioConv1d
    @ModuleInfo(key: "1") var stage1: H3AudioEncoderBlock
    // …
}
```

Elle produit le bon arbre — mais `ModuleParameters.unflattened` reconstruit un **tableau** dès que
les clés d'un niveau sont des entiers contigus partant de 0. Le dictionnaire de poids arrive donc
comme `NestedItem.array`, face à un `NestedItem.module` : `update(parameters:)` refuse, et le
message d'erreur est trompeur parce qu'il imprime les deux structures comme si elles se
correspondaient.

## Correctif

Utiliser un vrai tableau Swift, ce que MLXNN sait mettre à jour. Les éléments d'un
`nn.Sequential` sont hétérogènes, donc il leur faut une classe de base commune :

```swift
class H3AudioOp: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

final class H3AudioEncoder: H3AudioOp {
    @ModuleInfo(key: "block") var block: [H3AudioOp]
    override func callAsFunction(_ x: MLXArray) -> MLXArray { block.reduce(x) { $1($0) } }
}
```

Bonus : la construction redevient pilotée par la config (`for stride in config.encoderRates`) au
lieu d'être figée sur le nombre d'étages du checkpoint publié.

## Portée

Vaut pour tout `nn.Sequential`, `nn.ModuleList` ou `nn.ModuleDict` à clés `"0"`, `"1"`, … Une
suite numérique **non** contiguë ou ne commençant pas à 0 ne déclenche pas le dépliage — donc le
piège n'apparaît que sur la transcription la plus naturelle.

Voir `Sources/MiniMaxH3/Models/AudioVAE/H3AudioVAEEncoder.swift`.
