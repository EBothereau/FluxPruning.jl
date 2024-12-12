#------------------------------------------------------------------------------------------
#                              Local Itératif Structuré (Filtre/Neurones)
#------------------------------------------------------------------------------------------

#Norme L1
function create_masks_iter_filter_l1(model, list_to_prune)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1
    for layer in model.layers
        if isa(layer, Conv) 
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask = zeros(Bool,size(weights)) 
            mask_b = zeros(Bool,size(biais)) 
            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            if num_filter_pruned>0
                for i in 1:1:nb_filters
                    push!(list_index, sum(abs.(selectdim(weights,ndims(weights),i))))
                end
                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(list_index)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 
            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1

        elseif  isa(layer, Dense)
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            nb_neurons = size(weights)[1]
            num_neurons_pruned = list_to_prune[index_ltp]
            if num_neurons_pruned>0
                for i in 1:1:nb_neurons
                    push!(list_index, sum(abs.(weights[i,:])))
                end
                mask = zeros(Bool,size(weights)) 
                mask_b = zeros(Bool,size(biais)) 
                if flag == 1
                    flag = 2
                        #for index in pruned_previous_layer
                        #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                        #        #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                        #    end
                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_neurons_indices = sortperm(list_index)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)
            else 
                mask = zeros(Bool,size(weights)) 
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #end
                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                push!(masks, mask)
                mask_b = zeros(Bool,size(biais)) 
                push!(masks, mask_b)
            end
            index_ltp += 1
        end
    end
    return masks
end

#norme L2 
function create_masks_iter_filter_l2(model, list_to_prune)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1
    for layer in model.layers

        if isa(layer, Conv) 
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask = zeros(Bool,size(weights)) 
            mask_b = zeros(Bool,size(biais)) 

            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            if num_filter_pruned>0

                for i in 1:1:nb_filters
                    push!(list_index, sqrt(sum(abs.(selectdim(weights,ndims(weights),i)).^2)))
                end

                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(list_index)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 
            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1

        elseif  isa(layer, Dense)
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            nb_neurons = size(weights)[1]
            num_neurons_pruned = list_to_prune[index_ltp]
            if num_neurons_pruned>0
                for i in 1:1:nb_neurons
                    push!(list_index, sqrt(sum(abs.(weights[i,:]).^2))) 
                end
                mask = zeros(Bool,size(weights)) 
                mask_b = zeros(Bool,size(biais)) 
                if flag == 1
                    flag = 2
                        #for index in pruned_previous_layer
                        #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                        #        #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                        #    end
                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_neurons_indices = sortperm(list_index)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)

            else 
                mask = zeros(Bool,size(weights)) 
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #end
                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                push!(masks, mask)
                mask_b = zeros(Bool,size(biais)) 
                push!(masks, mask_b)
            end
            index_ltp += 1
        end
    end
    return masks
end



#geometric median
function create_masks_iter_filter_geomedian(model, list_to_prune)
    masks = []
    index_ltp = 1
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    for layer in model.layers
        if isa(layer, Conv) 
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 

            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            if num_filter_pruned>0
                geomedian_filter = geometric_median(weights)
                for i in 1:1:nb_filters
                    if sum(abs.(selectdim(weights,ndims(weights),i)))==0
                        push!(list_index, 0)
                    else 
                        a = selectdim(weights,ndims(weights),i)-geomedian_filter
                        a = sum(abs.(a)).^2
                        push!(list_index, a)
                    end
                end

                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(list_index)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 
            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1

        elseif  isa(layer, Dense)
            list_index = []

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 

            nb_neurons = size(weights)[1]
            num_neurons_pruned = list_to_prune[index_ltp] 

            if num_neurons_pruned > 0

                for i in 1:1:nb_neurons
                    push!(list_index, sqrt(sum(abs.(weights[i,:]).^2))) 
                end

                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #    end
                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                
                ##println(pruned_previous_layer)
                sorted_neurons_indices = sortperm(list_index)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)
            else 
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #end
                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                push!(masks, mask)
                push!(masks, mask_b)
            end
            index_ltp += 1
        end
    end
    return masks
end

function create_masks_iter_filter_median(model, list_to_prune)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1

    for layer in model.layers
        if isa(layer, Conv) 
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 

            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            if num_filter_pruned>0

                median_filter = median(weights, dims=length(size(weights)))
                
                for i in 1:1:nb_filters
                    if sum(abs.(selectdim(weights,ndims(weights),i)))==0
                        push!(list_index, 0)
                    else 
                        push!(list_index, sum(abs.(selectdim(weights,ndims(weights),i)-median_filter).^2))
                    end
                end

                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(list_index)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 
            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1

        elseif  isa(layer, Dense)
            list_index = []

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]

            nb_neurons = size(weights)[1]

            num_neurons_pruned = list_to_prune[index_ltp]
            if num_neurons_pruned>0
                median_neuron = median(weights, dims=1)[1]

                for i in 1:1:nb_neurons
                    if sum(abs.(weights[i,:]))==0
                        push!(list_index, 0)
                    else 
                        push!(list_index, sum(abs.(weights[i,:].-median_neuron).^2))
                    end
                end

                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 
                
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #    end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                
                ##println(pruned_previous_layer)
                sorted_neurons_indices = sortperm(list_index)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)

            else 
                mask =  zeros(Bool,size(weights)) 

                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                mask_b =  zeros(Bool,size(biais)) 

                push!(masks, mask)
                push!(masks, mask_b)
            end
            index_ltp += 1
        end
    end
    return masks

end



function create_masks_iter_filter_cop_0_0(model, r::Threshold)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1
    score_list = []
    importance_list = []
    to_keep = []
    sum_liste_to_prune = 0

   
    
        threshold = r.threshold
        index_ltp = 1

        for (index, layer) in enumerate(model.layers)
            if isa(layer, Conv) 
                list_index = []
                weights = Flux.params(layer)[1]
                biais = Flux.params(layer)[2]

                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 
                
                nb_filters = size(weights)[end]
                num_filter_pruned = list_to_prune[index_ltp]
                val_flag = nb_filters
                if num_filter_pruned>0 
                    for i in 1:nb_filters
                        if score_list[index_ltp][i]<=threshold && val_flag >=4
                            selectdim(mask,ndims(mask),i).=1
                            mask_b[i]=1
                        end
                        val_flag = val_flag-1
                    end
                end
                push!(masks, mask)
                push!(masks, mask_b)
                index_ltp += 1

            elseif  isa(layer, Dense)            
                list_index = []

                weights = Flux.params(layer)[1]
                biais = Flux.params(layer)[2]
                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 

                nb_neurons = size(weights)[1]

                num_neurons_pruned = list_to_prune[index_ltp]

                val_flag = nb_neurons
                if num_neurons_pruned>0 
                    for i in 1:nb_neurons
                        if score_list[index_ltp][i]<=threshold && val_flag >=6
                            mask[i,:].=1
                            mask_b[i]=1
                        end
                        val_flag = val_flag-1
                    end

                    push!(masks, mask)
                    push!(masks, mask_b)

                else 
                    push!(masks, mask)
                    push!(masks, mask_b)
                end
                index_ltp += 1
            end
        end

    return masks
end


function create_masks_iter_filter_cop_0_0(model, list_to_prune,data_in)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1
    score_list = []
    importance_list = []
    to_keep = []
    sum_liste_to_prune = 0

    for layer in model.layers
        importance_list = []

        if isa(layer, Conv) 
                
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            sum_liste_to_prune +=num_filter_pruned
            corr_matrix = zeros(nb_filters,nb_filters)
            if num_filter_pruned>0 && nb_filters >=4
                for i in 1:nb_filters
                    for j in 1:i
                        if i!=j
                            corr_matrix[i,j]= sum(cor(selectdim(weights,ndims(weights),i),selectdim(weights,ndims(weights),j)))/(size(weights)[1])
                            corr_matrix[j,i]=corr_matrix[i,j]
                        end
                    end
                end
                for n in 1:nb_filters
                    m = maximum(corr_matrix[n,:])
                    imp = 1-sum(sort(corr_matrix[n,:])[end-2:end])/(m*3)
                    push!(importance_list,imp)
                end
                liste_indices = sortperm(importance_list,rev=true)[1:4]
                importance_list[liste_indices] .= Inf
                push!(score_list,importance_list)
            else
                push!(score_list, [])
            end
            index_ltp += 1

        elseif  isa(layer, Dense)
            list_index = []

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]

            nb_neurons = size(weights)[1]

            num_neurons_pruned = list_to_prune[index_ltp]
            sum_liste_to_prune +=num_neurons_pruned

            if num_neurons_pruned>0 && nb_neurons>=6
                corr_matrix = zeros(nb_neurons,nb_neurons)
                for i in 1:nb_neurons
                    for j in 1:i
                        if i!=j
                            corr_matrix[i,j]=cor(weights[i,:],weights[j,:])
                            corr_matrix[j,i]=corr_matrix[i,j]
                        end
                    end
                end

                for n in 1:nb_neurons
                    m = maximum(corr_matrix[n,:])
                    imp = 1-sum(sort(corr_matrix[n,:])[end-2:end])/(m*3)
                    push!(importance_list,imp)
                end
                liste_indices = sortperm(importance_list,rev=true)[1:6]
                importance_list[liste_indices] .= Inf
                push!(score_list,importance_list)
            else
                push!(score_list, [])
            end

            index_ltp += 1
        else
            push!(to_keep, [])
        end
    end
    
    if sum_liste_to_prune >0
        score_list2 = sort(collect(Iterators.flatten(score_list)))
        threshold = score_list2[sum_liste_to_prune]
        index_ltp = 1

        for (index, layer) in enumerate(model.layers)
            if isa(layer, Conv) 
                list_index = []
                weights = Flux.params(layer)[1]
                biais = Flux.params(layer)[2]

                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 
                
                nb_filters = size(weights)[end]
                num_filter_pruned = list_to_prune[index_ltp]
                val_flag = nb_filters
                if num_filter_pruned>0 
                    for i in 1:nb_filters
                        if score_list[index_ltp][i]<=threshold && val_flag >=4
                            selectdim(mask,ndims(mask),i).=1
                            mask_b[i]=1
                        end
                        val_flag = val_flag-1
                    end
                end
                push!(masks, mask)
                push!(masks, mask_b)
                index_ltp += 1

            elseif  isa(layer, Dense)            
                list_index = []

                weights = Flux.params(layer)[1]
                biais = Flux.params(layer)[2]
                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 

                nb_neurons = size(weights)[1]

                num_neurons_pruned = list_to_prune[index_ltp]

                val_flag = nb_neurons
                if num_neurons_pruned>0 
                    for i in 1:nb_neurons
                        if score_list[index_ltp][i]<=threshold && val_flag >=6
                            mask[i,:].=1
                            mask_b[i]=1
                        end
                        val_flag = val_flag-1
                    end

                    push!(masks, mask)
                    push!(masks, mask_b)

                else 
                    push!(masks, mask)
                    push!(masks, mask_b)
                end
                index_ltp += 1
            end
        end
    else 
        for layer in model.layers
            if isa(layer, Conv) ||isa(layer, Dense)       
                weights = Flux.params(layer)[1]
                biais = Flux.params(layer)[2]
                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 
                push!(masks, mask)
                push!(masks, mask_b)
            end
        end

    end

    return masks
end




function create_masks_iter_filter_cor(model, list_to_prune)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1

    for layer in model.layers
        if isa(layer, Conv) 
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]

            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 
            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            if num_filter_pruned>0
                corr_matrix = zeros(nb_filters,nb_filters)
                for i in 1:nb_filters
                    for j in 1:i
                        if i!=j
                            corr_matrix[i,j]= sum(cor(selectdim(weights,ndims(weights),i),selectdim(weights,ndims(weights),j)))/(size(weights)[1])
                            corr_matrix[j,i]=corr_matrix[i,j]
                        end
                    end
                end

                importance_list = []
                for n in 1:nb_filters
                    m = maximum(corr_matrix[n,:])
                    imp = 1-sum(sort(corr_matrix[n,:])[end-2:end])/(m*3)
                    push!(importance_list,imp)
                end

                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(importance_list)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 

            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1

        elseif  isa(layer, Dense)

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]

            nb_neurons = size(weights)[1]

            num_neurons_pruned = list_to_prune[index_ltp]

            if num_neurons_pruned>0
                corr_matrix = zeros(nb_neurons,nb_neurons)
                for i in 1:nb_neurons
                    for j in 1:i
                        if i!=j
                            corr_matrix[i,j]=cor(weights[i,:],weights[j,:])
                            corr_matrix[j,i]=corr_matrix[i,j]
                        end
                    end
                end

                importance_list = []
                for n in 1:nb_neurons
                    m = maximum(corr_matrix[n,:])
                    imp = 1-sum(sort(corr_matrix[n,:])[end-2:end])/(m*3)
                    push!(importance_list,imp)
                end


                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 
                
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #    end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                
                ##println(pruned_previous_layer)
                sorted_neurons_indices = sortperm(importance_list)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)


            else 
                mask =  zeros(Bool,size(weights)) 

                if flag == 1
                    flag = 2

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                push!(masks, mask)
                mask_b = zeros(Bool,size(biais)) 
                push!(masks, mask_b)
            end
            index_ltp += 1
        end
    end
    return masks

end


function create_masks_iter_filter_mean(model, list_to_prune)
    masks = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    index_ltp = 1

    for layer in model.layers
        if isa(layer, Conv) 
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 

            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]
            if num_filter_pruned>0

                mean_filter = mean(weights, dims=length(size(weights)))
                for i in 1:1:nb_filters
                    if sum(abs.(selectdim(weights,ndims(weights),i)))==0
                        push!(list_index, 0)
                    else 
                        a = selectdim(weights,ndims(weights),i)-mean_filter
                        a = sum(abs.(a)).^2
                        push!(list_index, a)
                    end
                end

                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(list_index)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 
            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp += 1

        elseif  isa(layer, Dense)
            list_index = []

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]

            nb_neurons = size(weights)[1]

            num_neurons_pruned = list_to_prune[index_ltp]
            if num_neurons_pruned>0
                mean_neuron = mean(weights, dims=1)[1]

                for i in 1:1:nb_neurons
                    if sum(abs.(weights[i,:]))==0
                        push!(list_index, 0)
                    else 
                        push!(list_index, sum(abs.(weights[i,:].-mean_neuron).^2))
                    end
                end

                mask =  zeros(Bool,size(weights)) 
                mask_b =  zeros(Bool,size(biais)) 
                
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #    end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                
                ##println(pruned_previous_layer)
                sorted_neurons_indices = sortperm(list_index)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)

            else 
                mask =  zeros(Bool,size(weights)) 

                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                push!(masks, mask)
                mask_b = zeros(Bool,size(biais)) 
                push!(masks, mask_b)
            end
            index_ltp += 1
        end
    end
    return masks
end

function create_masks_iter_filter_random(model, list_to_prune)
    masks = []
    index_ltp = 1
    al = []
    pruned_previous_layer = []
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    for layer in model.layers
        if isa(layer, Conv) 
            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 

            nb_filters = size(weights)[end]
            num_filter_pruned = list_to_prune[index_ltp]

            if num_filter_pruned>0
                for i in 1:1:nb_filters
                    if sum(abs.(selectdim(weights,ndims(weights),i)))==0
                        push!(list_index, 0)
                    else 
                        push!(list_index, i)
                    end
                end
                al = rand(nb_filters)
                al[list_index .== 0] .= 0 # On met une valeur très grande à tous les éléments déjà masqués dans le masque précédent

                if flag == 0
                    flag = 1
                else
                    for index in pruned_previous_layer
                        selectdim(mask,ndims(mask)-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                sorted_filters_indices = sortperm(al)
                pruned_filter_indices = sorted_filters_indices[1:(num_filter_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_filter_indices
                selectdim(mask,ndims(mask),pruned_filter_indices) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_filter_indices].= 1 
            end
            push!(masks, mask)
            push!(masks, mask_b)
            index_ltp +=1
        elseif  isa(layer, Dense)
            list_index = []

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            mask =  zeros(Bool,size(weights)) 
            mask_b =  zeros(Bool,size(biais)) 

            nb_neurons = size(weights)[1]
            num_neurons_pruned = list_to_prune[index_ltp]

            if num_neurons_pruned > 0
                num_neurons_pruned = list_to_prune[index_ltp]
                for i in 1:1:nb_neurons
                    if sum(abs.(weights[i,:]))==0
                        push!(list_index, 0)
                    else 
                        push!(list_index, i)
                    end
                end
                al = rand(nb_neurons)
                al[list_index .== 0] .= 0 # On met une valeur très grande à tous les éléments déjà masqués dans le masque précédent
    
                
                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #    end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                
                ##println(pruned_previous_layer)
                sorted_neurons_indices = sortperm(al)
                pruned_neurons_indices = sorted_neurons_indices[1:(num_neurons_pruned)]  # Exclure les indices des filtres à masquer
                pruned_previous_layer = pruned_neurons_indices
                mask[pruned_neurons_indices,:] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                mask_b[pruned_neurons_indices].= 1 
                push!(masks, mask)
                push!(masks, mask_b)

            else 
                mask =  zeros(Bool,size(weights)) 

                if flag == 1
                    flag = 2
                    #for index in pruned_previous_layer
                    #        mask[:,(index-1)*16 + 1 : index*16] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                            #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                    #end

                else
                    for index in pruned_previous_layer
                        mask[:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                    end
                end
                pruned_previous_layer = []
                push!(masks, mask)
                mask_b = zeros(Bool,size(biais)) 
                push!(masks, mask_b)
            end
            index_ltp +=1
        end
    end
    return masks
end


#------------------------------------------------------------------------------------------
#                              Par couche
#------------------------------------------------------------------------------------------

function create_masks_iter_layer_l1(model,  ratio, num_layer)
    masks = []
    s=0
    id_layer = 1
    for layer in model.layers
        if isa(layer, Conv) || isa(layer, Dense)
            if id_layer == num_layer
                weights = Flux.params(layer)[1]
                num_params = length(weights)
                sorted_weights = sort(abs.(vec(weights))) #.^2 L2
                num_pruned = max(1, round(Int, num_params*ratio))
                threshold = sorted_weights[num_pruned]
                mask = abs.(weights) .<= threshold  #.^2 L2           
                push!(masks, mask)
                s=s+sum(mask)
            else
                mask =   [] #CuArray{Float32}(undef, 0)
                push!(masks, mask)
            end
            mask =   [] #CuArray{Float32}(undef, 0)
            push!(masks, mask)
            id_layer = id_layer+1
        end
    end
    weights = nothing
    sorted_weights = nothing
    mask = nothing

    return s, masks
end


function geometric_median(filters)
    n = size(filters)[end]  # Nombre de filtres
    if length(size(filters))==3
        m = size(filters, 1)  # Dimension du filtre en x
        p = size(filters, 2)  # Dimension du filtre en y
        median =  zeros(Float32, m,p)

    else
        m = size(filters, 1)  # Dimension du filtre en x
        p = size(filters, 2)  # Dimension du filtre en y
        q = size(filters, 3)
        median =  zeros(Float32, m,p,q)
    end
    # Initialisation de la médiane géométrique
    liste_non_nuls = [x for x in 1:n if sum(abs.(selectdim(filters,ndims(filters),x)))!= 0]    # Calcul de la distance euclidienne
    n_non_nuls = size(liste_non_nuls,1)
    liste_filtre = selectdim(filters,ndims(filters),liste_non_nuls)
    function distance(x, y)
        return sqrt(sum(abs2, x .- y))
    end
    # Calcul de la médiane géométrique en utilisant l'algorithme de Weiszfeld
    function weiszfeld_median(points, epsilon=1e-5, max_iterations=100)
        y = mean(points)
        
        for _ in 1:max_iterations
            dist = [distance(points[i], y) for i in 1:n_non_nuls]
            if maximum(dist) < epsilon
                break
            end
            weights = 1 ./ dist
            weights ./= sum(weights)
            y = sum(points[i] .* weights[i] for i in 1:n_non_nuls)
        end
        return y
    end
    
    # Calcul de la médiane géométrique selon la troisième dimension du tableau
    if length(size(filters))==3

        for i in 1:p
            for j in 1:m
                median[j,i] = weiszfeld_median(liste_filtre[j,i,:])
            end
        end
    else 
        for i in 1:p
            for j in 1:m
                for k in 1:q
                    median[j,i,k] = weiszfeld_median(liste_filtre[j,i,k,:])
                end
            end
        end

    end
    return median
end