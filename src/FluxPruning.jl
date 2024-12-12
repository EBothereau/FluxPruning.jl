module FluxPruning

using Flux
import BSON
import Random
export prune
using MaskedArrays
using MaskedArrays: mask, freeze, MaskedArray, MaskedSliceArray
using Functors: isleaf, functor
using LinearAlgebra
using ChainRulesCore
using Zygote
using CUDA
using RandomNumbers
using Statistics

#Pruning Structures 

include("structures.jl")
export RatioPrune, Threshold

#Structured pruning 
include("FilterPruning/FilterCriteria.jl")
include("FilterPruning/FilterPruning.jl")
include("FilterPruning/utils.jl")

#Unstructured pruning 
include("WeightsPruning/WeightsCriteria.jl")
include("WeightsPruning/WeightsPruning.jl")
include("WeightsPruning/utils.jl")


export apply_masks, weightPruning,filterPruning



function apply_masks(model,masks)
    ps = Flux.params(model)  
    for (index, mask) in enumerate(masks)   
        ps[index][mask].=0
    end
end

function weightPruning(model,r::RatioPrune)
    list_e,_= get_struct_Network_weights(model,r.data_in, r.locality)
    #println(list_e)
    list_to_prune = number_pruned_weigth(r,list_e)
    #println(list_to_prune)

    masks,_ = get_pruning_mask_weight(model, list_to_prune[:,1],r)
    #apply_masks(model,masks)   
    return model, masks
end

function weightPruning(model,r::RatioPrune,list_to_prune)
    masks,_ = get_pruning_mask_weight(model, list_to_prune,r)
    #apply_masks(model,masks)   
    return model, masks
end


function filterPruning(model,r::RatioPrune)
    list_e,_= get_struct_Network_filter(model,r.data_in,r.locality)
    list_to_prune = number_pruned_filter(r,list_e)
    #println(list_to_prune[:,1])
    #println(list_e)
    masks,_ = get_pruning_mask_filter(model, list_to_prune[:,1],r)
    masks = verif_network(model,masks, r.data_in)
    #for e in masks
        ##println(size(e))
    #end
    model = reduce_filters(model,masks,r.data_in)
    return model
end

function filterPruning(model,r::RatioPrune,list_to_prune)
    masks,_ = get_pruning_mask_filter(model, list_to_prune,r)
    masks = verif_network(model,masks, r.data_in)
    #for e in masks
    #    #println(size(e))
    #end
    model = reduce_filters(model,masks,r.data_in)
    return model
end



function weightPruning(model,r::Threshold)
    masks,_ = get_pruning_mask_weight(model,r)
    #apply_masks(model,masks)   
    return model, masks
end


function filterPruning(model,r::Threshold)
    masks = get_pruning_mask_filter(model, r)
    masks = verif_network(model,masks, r.data_in)
    model = reduce_filters(model,masks,r.data_in)
    return model
end


end # end of module PruningForFlux
