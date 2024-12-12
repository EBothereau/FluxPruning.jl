function create_masks_iter_loc_l1(model, list_to_prune)
    masks = []
    index_layer = 0
    for layer in model.layers
        if isa(layer, Conv) || isa(layer, Dense)
            index_layer = index_layer+1
            weights = Flux.params(layer)[1]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(Flux.params(layer)[2])) #bias
            if isa(layer, Conv) 
                nb_filters = size(weights)[end]
                if nb_filters > 1 #On conserve au moins un filtre
                    mask = L1_criterion(weights,index_layer,list_to_prune)
                end
            elseif   isa(layer, Dense)
                nb_neurons = size(weights)[1]
                if nb_neurons > 1  # On conserve au moins 1 neurones
                    mask = L1_criterion(weights,index_layer,list_to_prune)
                end    
            end
            push!(masks, mask)
            push!(masks, mask_b)
        end
    end
    return masks
end

function L1_criterion(weights,index_layer,list_to_prune)
    sorted_weights = sort(abs.(vec(weights)))
    num_pruned = list_to_prune[index_layer]
    threshold = sorted_weights[num_pruned]
    mask = abs.(weights) .<= threshold          
    return mask
end

function create_masks_l1(model, r::Threshold)
    masks = []
    index_layer = 0
    for layer in model.layers
        if isa(layer, Conv) || isa(layer, Dense)
            index_layer = index_layer+1
            weights = Flux.params(layer)[1]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(Flux.params(layer)[2])) #bias
            if isa(layer, Conv) 
                nb_filters = size(weights)[end]
                if nb_filters > 1 #On conserve au moins un filtre
                    mask = L1_criterion_threshold(weights,r.threshold)
                end
            elseif   isa(layer, Dense)
                nb_neurons = size(weights)[1]
                if nb_neurons > 1  # On conserve au moins 1 neurones
                    mask = L1_criterion_threshold(weights,r.threshold)
                end    
            end
            push!(masks, mask)
            push!(masks, mask_b)
        end
    end
    return masks
end

function L1_criterion_threshold(weights,threshold)
    mask = abs.(weights) .<= threshold          
    return mask
end

#------------------------------------------------------------------------------------------
#                              Global Itératif Non structuré
#------------------------------------------------------------------------------------------

#random
function create_masks_random(model, list_to_prune)
    index_ltp = 1
    masks = []  
    #println(list_to_prune)
    for layer in model.layers
        index_removed = Int64[]
        if isa(layer, Conv) || isa(layer, Dense)
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 
            num_pruned = list_to_prune[index_ltp]
            num_params = length(weights)
            for (i, w) in enumerate(weights)
                if w==0
                    push!(index_removed,i)
                end
            end   
            all_index_filtered = [x for x in 1:num_params if !(x in index_removed)]
            indices = Random.randperm(length(all_index_filtered))[1:num_pruned]
            mask[index_removed].= 1
            mask[indices].= 1
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1
        end
    end
    return masks
end



#par magnitude
function create_masks_global_iter_l1(model, list_to_prune)
    masks = []
    weights_list = []
    for layer in model.layers
        if isa(layer, Conv) || isa(layer, Dense)
            weights = Flux.params(layer)[1]
            push!(weights_list, vec(weights))
        end
    end
    all_weights = cat(weights_list..., dims=1)
    sorted_weights = sort(abs.(all_weights))
    num_pruned = list_to_prune[1]
    if num_pruned > 0
        threshold = sorted_weights[num_pruned]
        for layer in model.layers
            if isa(layer, Conv) || isa(layer, Dense)
                weights = Flux.params(layer)[1]
                mask = abs.(weights) .<= threshold
                push!(masks, mask)
                mask =   [] #CuArray{Float32}(undef, 0)
                push!(masks, [] )#CuArray{Float32}(undef, 0)
            end
        end
    else 
        for layer in model.layers
            if isa(layer, Conv) || isa(layer, Dense)
                mask =   [] #CuArray{Float32}(undef, 0)
                push!(masks, mask)
                push!(masks, [] )#CuArray{Float32}(undef, 0)
            end
        end
    end
    return masks
    weights = nothing
    sorted_weights = nothing
    mask = nothing
end


#------------------------------------------------------------------------------------------
#                              Global LAMP
#------------------------------------------------------------------------------------------

#par magnitude
function create_masks_lamp(model, list_to_prune)
    masks = []
    weights_list = []
    for layer in model.layers
        if isa(layer, Conv) || isa(layer, Dense)
            weights = Flux.params(layer)[1]
            weights_sorted = sort(vec(weights).^2)
            weight_score = weights_sorted./reverse(cumsum(reverse(weights_sorted)))
            push!(weights_list, weight_score)
        end
    end
    all_weights = cat(weights_list..., dims=1)
    sorted_weights = sort(abs.(all_weights))
    num_pruned = list_to_prune[1]
    if num_pruned>0
        if num_pruned != 0
            threshold = sorted_weights[num_pruned]
            #println("le threshold vaut :$(threshold)")
        else
            threshold = Inf
        end    
        index = 1 
        for layer in model.layers
            if isa(layer, Conv) || isa(layer, Dense)
                weights = Flux.params(layer)[1]
                weight_score_2 = weights_list[index][sortperm(sortperm(abs.(vec(weights))))]
                mask = weight_score_2 .<= threshold
                push!(masks, mask)
                mask =   [] #CuArray{Float32}(undef, 0)
                push!(masks, mask)
                index = index+1
            end
        end
    else
        for layer in model.layers
            if isa(layer, Conv) || isa(layer, Dense)
                mask =   [] #CuArray{Float32}(undef, 0)
                push!(masks, mask)
                push!(masks, mask)
            end
        end
    end
    weights = nothing
    mask = nothing
    sorted_weights = nothing
    return masks, threshold
end


function create_masks_lamp(model, r::Threshold)
    masks = []
    weights_list = []
            threshold = r.threshold
        index = 1 
        for layer in model.layers
            if isa(layer, Conv) || isa(layer, Dense)
                weights = Flux.params(layer)[1]
                weight_score_2 = weights_list[index][sortperm(sortperm(abs.(vec(weights))))]
                mask = weight_score_2 .<= threshold
                push!(masks, mask)
                mask =   [] #CuArray{Float32}(undef, 0)
                push!(masks, mask)
                index = index+1
            end
        end
    weights = nothing
    mask = nothing
    sorted_weights = nothing
    return masks, threshold
end

#------------------------------------------------------------------------------------------
#                              Global SynFlow
#------------------------------------------------------------------------------------------

#par magnitude
function create_masks_SynFlow(model, list_to_prune, r)
    masks = []
    data_ones = ones(r.data_in)#Dimensions entrée
    model2 = deepcopy(model)
    ps = Flux.params(model2)  
    testmode!(model2, true)  # We are in test mode, with no dropout 
    for x in ps
        x .= abs.(x)
    end
    gs = Flux.gradient(ps) do
        ŷ = model2[1:end-1](Float32.(data_ones)) 
        ##println( (ŷ))
        sum(ŷ)
    end
    for x in ps
        isnothing(gs[x]) && continue
        x̄r = gs[x]
        x .= x̄r.*x        
    end
    masks = create_masks_global_iter_l1(model2, list_to_prune)
    testmode!(model2, false)  # We are in test mode, with no dropout 
    return masks
end


function create_masks_SynFlow(model, r::Threshold)
    masks = []
    data_ones = ones(r.data_in)
    model2 = deepcopy(model)
    ps = Flux.params(model2)  
    testmode!(model2, true)  # We are in test mode, with no dropout 
    for x in ps
        x .= abs.(x)
    end
    gs = Flux.gradient(ps) do
        ŷ = model2[1:end-1](Float32.(data_ones)) 
        #println( (ŷ))
        sum(ŷ)
    end
    for x in ps
        isnothing(gs[x]) && continue
        x̄r = gs[x]
        x .= x̄r.*x 
    end
    masks = create_masks_global_iter_l1(model2, r)
    testmode!(model2, false)  # We are in test mode, with no dropout 
    return masks
end


