# PruningForFlux.jl

This package allows to prune Flux.jl neural networks. 
For now, this package only allows typical Dense and Convolutional Layers. Behaviour on Residual networks has not been implemented yet. 

This code includes the following Pruning methods: 


Structured

- local
  - L1
  - L2
  - geomedian
  - mean
  - median
  - cor (à corriger)
  - random
- global
  - cop_0_0 (a corriger)

Unstructured

- local
  - L1
  - random 
- global
  - lamp
  - L1
  - SynFlow


``` 
using Flux
using PruningForFlux

m = Chain(
  Conv((7,), 2 => 128, pad=SamePad(), relu),
  Conv((5,), 128 => 128, pad=SamePad(), relu),
  MaxPool((2,)),
  Flux.flatten,
  Dense(2048, 256, relu), 
  Dropout(0.5),
  Dense(256, 128, relu),
  Dropout(0.5),
  Dense(128,x),
)

input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0.5 # ratio of weights or elements to be removed
localite_pruning = "local" # local or global
norme_pruning = "L1" # see the following table to find the corresponding pruning methods 
```

## Structured Pruning
When appliying structured pruning: 
```
input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0.5 # ratio of weights or elements to be removed
localite_pruning = "local" # local or global
norme_pruning = "geomedian" # see the following table to find the corresponding pruning methods 


r = RatioPrune(ratio,norme_pruning,localite_pruning, (window_size,2,1))
model = filterPruning(model,r)
```

The returned model will be the pruned model. It is a smaller netwotk, but still a typical Flux network as the simplifications due to the removal of the filters and the neurons have been applied. 



It is also possible to designate the number of filters/neurons to remove manually: 

```
input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0.5 # ratio of weights or elements to be removed
localite_pruning = "local" # local or global
norme_pruning = "geomedian" # see the following table to find the corresponding pruning methods 
elements_to_prune = [1,1,0,0,0]

r = RatioPrune(ratio,norme_pruning,localite_pruning, (window_size,2,1))
model = filterPruning(model,r,elements_to_prune)
```



## Unstructured Pruning
When appliying unstructured pruning:

```
input_data_size = (256,2,1) # Dimensions of the typical batch size of the network.
ratio = 0.5 # ratio of weights or elements to be removed
localite_pruning = "global" # local or global
norme_pruning = "lamp" # see the following table to find the corresponding pruning methods 


r = RatioPrune(ratio,norme_pruning,localite_pruning, (window_size,2,1))
model,masks = weightPruning(model,r)
apply_masks(model,masks)
```

mask will return a matrix indicating which weights are to be set to 0. It is necessary to use apply_mask to actually set the weights to 0. 




[![Build Status](https://github.com/EBothereau/PruningForFlux.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/EBothereau/PruningForFlux.jl/actions/workflows/CI.yml?query=branch%3Amain)
