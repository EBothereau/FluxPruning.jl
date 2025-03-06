# PruningForFlux.jl

[![Build Status](https://github.com/EBothereau/FluxPruning/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EBothereau/FluxPruning/actions/workflows/CI.yml?query=branch%3Amain)

This package allows to prune Flux.jl neural networks. 
For now, this package only allows typical Dense and Convolutional Layers. Behaviour on Residual networks has not been implemented yet. 

This code includes the following Pruning methods: 


Structured

- local
  - L1
  - L2
  - geomedian [Link](https://arxiv.org/abs/1811.00250)
  - mean
  - median
  - cor (to be fixed)
  - random
- global
  - cop_0_0 (to be fixed) [Link](https://arxiv.org/abs/1906.10337)

Unstructured

- local
  - L1
  - random 
- global
  - lamp [Link](https://arxiv.org/abs/2010.07611)
  - L1
  - SynFlow [Link](https://arxiv.org/abs/2006.05467)


``` 
using Flux
using FluxPruning

model = Chain(
  Conv((7,), 2 => 128, pad=SamePad(), relu),
  MaxPool((4,)),
  Conv((5,), 128 => 128, pad=SamePad(), relu),
  MaxPool((2,)),
  Flux.flatten,
  Dense(4096, 256, relu), 
  Dropout(0.5),
  Dense(256, 128, relu),
  Dense(128,6),
)
```

## Structured Pruning
When appliying structured pruning: 
```
input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0.5 # ratio of weights or elements to be removed
localite_pruning = "local" # local or global
norme_pruning = "geomedian" # see the following table to find the corresponding pruning methods 


r = RatioPrune(ratio,norme_pruning,localite_pruning,input_data_size)
model = filterPruning(model,r)
```

The returned model will be the pruned model. It is a smaller netwotk, but still a typical Flux network as the simplifications due to the removal of the filters and the neurons have been applied. 



It is also possible to designate the number of filters/neurons to remove manually: 

```
input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0# ratio of weights or elements to be removed
localite_pruning = "local" # local or global
norme_pruning = "geomedian" # see the following table to find the corresponding pruning methods 
elements_to_prune = [120,10,200,5,0]

r = RatioPrune(ratio,norme_pruning,localite_pruning, input_data_size)
model = filterPruning(model,r,elements_to_prune)
```
Will return the folowwing network: 

```
Chain([Conv((7,), 2 => 8, relu, pad=3), MaxPool((4,)), Conv((5,), 8 => 118, relu, pad=2), MaxPool((2,)), flatten, Dense(3776 => 56, relu), Dropout(0.5), Dense(56 => 123, relu), Dense(123 => 6)])

```

## Unstructured Pruning
When appliying unstructured pruning:

```
input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0.5 # ratio of weights or elements to be removed
localite_pruning = "global" # local or global
norme_pruning = "lamp" # see the following table to find the corresponding pruning methods 


r = RatioPrune(ratio,norme_pruning,localite_pruning, input_data_size)
model,masks = weightPruning(model,r)
apply_masks(model,masks)
```

mask will return a matrix indicating which weights are to be set to 0. It is necessary to use apply_mask to actually set the weights to 0. 



