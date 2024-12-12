function number_pruned_filter(r::RatioPrune, e_per_layer)
    list_to_prune = zeros(Int64,length(e_per_layer),1)
    for p in 1:(length(e_per_layer)-1)
        if e_per_layer[p] >= 4
            to_remove = round(e_per_layer[p]*(r.pruningRatio))
            if e_per_layer[p]-to_remove<4
                to_remove=e_per_layer[p]-4
            end
            Q = (e_per_layer[p]-to_remove)/e_per_layer[p]
            Q = Q^(1/1)
            liste = [round(e_per_layer[p]*(Q^(i-1))-e_per_layer[p]*(Q^(i)) ) for i in 1]
            
            if sum(liste)!= to_remove
                liste[1] += to_remove-sum(liste)
            end

            list_to_prune[p,:]=liste
        else 
            list_to_prune[p,:].=0
        end
    end
    list_to_prune[length(e_per_layer),:].=0
return list_to_prune
liste = nothing
to_remove= nothing
end

function verif_network(model,masks, size_data_input)
    pruned_previous_layer = []
    index_mask = 1
    taille = size_data_input[1]
    taille_precedent = []
    data_type = zeros(size_data_input)
    flag = 0 #0 == premier layer, 1 == conv 2 == dense
    ps = Flux.params(model)  
    for (index, mask) in enumerate(masks)   
        ps[index][mask].=0
    end
    for (l, layer) in enumerate(model)
        if isa(layer, Conv) 

            list_index = []
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]

            nb_filters = size(weights)[end]
            for i in 1:1:nb_filters
                if sum(abs.(selectdim(weights,ndims(weights),i)))==0
                    push!(list_index, i)
                    masks[index_mask+1][i]=1
                    selectdim(masks[index_mask],ndims(masks[index_mask]),i) .= 1   
                end
            end

            if flag == 0
                flag = 1
            else
                for index in pruned_previous_layer
                    selectdim(masks[index_mask],ndims(masks[index_mask])-1,index) .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)  selectdim(masks[index_mask],ndims(masks[index_mask])-1,index)
                end
            end
            pruned_previous_layer = list_index
            index_mask = index_mask+2
        elseif  isa(layer, Dense)
            list_index = []

            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            nb_neurons = size(weights)[1]

            for i in 1:1:nb_neurons
                if sum(abs.(weights[i,:]))==0
                    push!(list_index, i)
                    masks[index_mask+1][i]=1
                    masks[index_mask][i,:] .= 1
                end
            end
            if flag == 1
                flag = 2
                data_type = zeros(taille_precedent)

                selectdim(data_type,ndims(data_type)-1,pruned_previous_layer).=1  
                data_type = Flux.flatten(data_type) 
                masks[index_mask][:,findall(x -> x == 1, data_type)] .=1 
                #for index in pruned_previous_layer
                #    masks[index_mask] = Array(masks[index_mask])
                #    masks[index_mask][:,((index-1)*taille + 1):(index*taille)] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                        #16 == dilension en x pour la dimension de couche en entrée de flatten. A modifier
                #end
            else
                for index in pruned_previous_layer
                    masks[index_mask][:,index] .= 1  # Masquer les filtres à partir des indices calculés            push!(masks, mask)
                end
            end
            pruned_previous_layer = list_index
            index_mask = index_mask+2

        end
        new_layer = layer 
        taille_precedent = taille
        taille = size(new_layer(data_type))
        data_type = zeros(taille)
        ##println(size(data_type))

    end
    data_type = nothing
    weights = nothing
    biais = nothing
    nb_neurons = nothing
    pruned_previous_layer= nothing

    return masks
end


function reduce_filters(model, masks, size_data_input)
    list_layers = []
    flag = 0
    ps = Flux.params(model)  
    nb_filter_last_conv = 0
    taille = size_data_input
    matrice_test = zeros(size_data_input)
    taille_pre = size_data_input
    pruned_previous_layer = []
    for (index, mask) in enumerate(masks)   
        ps[index][mask].=0
    end
    new_mask = []
    index = 1
    # Création du modèle réduit
    for (index, layer) in enumerate(model.layers)
        ##println(typeof(layer))
        if isa(layer, Conv)
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            

            nb_filter_last_conv = size(weights)[end]
            
            liste_a = []
            for i in 1:size(weights)[1]
                if sum(abs.(selectdim(weights,1,i)))==0
                    push!(liste_a,i)
                end
            end

            liste_abis = []

            if length(size(weights))==4
                for i in 1:size(weights)[2]
                    if sum(abs.(selectdim(weights,2,i)))==0
                        push!(liste_abis,i)
                    end
                end
            end

            liste_b = []
            for i in 1:size(weights)[end-1]
                if sum(abs.(selectdim(weights,ndims(weights)-1,i)))==0     
                    push!(liste_b,i)
                end
            end

            liste_c = []
            for i in 1:size(weights)[end]
                if sum(abs.(selectdim(weights,ndims(weights),i)))==0
                    push!(liste_c,i)
                end
            end

            if length(size(weights))==3
                weights2 =  weights[setdiff(1:end, liste_a),setdiff(1:end, liste_b),setdiff(1:end, liste_c)]
            else
                weights2 =  weights[setdiff(1:end, liste_a),setdiff(1:end, liste_abis),setdiff(1:end, liste_b),setdiff(1:end, liste_c)]
            end

            biais2 = biais[setdiff(1:end, liste_c)]
            pruned_previous_layer = liste_c

            
            if length(size(weights))==3
                masks[index] = masks[index][setdiff(1:end, liste_a), setdiff(1:end, liste_b), setdiff(1:end, liste_c)]
            else
                masks[index] = masks[index][setdiff(1:end, liste_a),setdiff(1:end, liste_abis), setdiff(1:end, liste_b), setdiff(1:end, liste_c)]
            end


            masks[index+1] = masks[index+1][setdiff(1:end, liste_c)]

            if length(size(weights2))==3
                a,b,c = size(weights2)

            # Copie des filtres non nuls dans une nouvelle couche
                new_layer = Conv((a,), b => c, pad=layer.pad, stride=layer.stride, layer.σ)
            else 
                a,b,c,d = size(weights2)
                ##println(size(weights2))
            # Copie des filtres non nuls dans une nouvelle couche
                new_layer = Conv((a,b), c => d, pad=layer.pad, stride=layer.stride, layer.σ)

            end
            # Combiner new_layer avec m

            weights_new = Flux.params(new_layer)[1]
            biais_new = Flux.params(new_layer)[2]
            
            weights_new .= weights2
            biais_new .= biais2
            index += 2
            flag = 1
            push!(list_layers, new_layer)
        elseif isa(layer, Dense)
            weights = Flux.params(layer)[1]
            biais = Flux.params(layer)[2]
            liste = []
            if flag ==1
                flag =2
                matrice_test = zeros(taille_pre)

                selectdim(matrice_test,ndims(matrice_test)-1,pruned_previous_layer).=1  
                matrice_test = Flux.flatten(matrice_test)
                liste = [i for (i, x) in enumerate(matrice_test) if x[1] == 1]
                #for index in 1:nb_filter_last_conv
                #    if sum(abs.(weights[:,(((index-1)*taille + 1):(index*taille))] )) ==0
                #        push!(liste, index)
                #        #println("ici")
                #    end
                #end
                masks[index] = masks[index][:,setdiff(1:end, liste)]
                weights =  weights[:,setdiff(1:end, liste)]
            end


            

            b,a = size(weights)

            liste_a = []
            for i in 1:a
                if sum(abs.(weights[:,i]))==0
                    push!(liste_a,i)
                end
            end
            
            b,a = size(weights)
            liste_b = []
            for i in 1:b
                if sum(abs.(weights[i,:]))==0
                    push!(liste_b,i)
                end
            end

            b,a = size(weights)

            
            weights2 =  weights[setdiff(1:end, liste_b),setdiff(1:end, liste_a)]
            
            biais2 = biais[setdiff(1:end, liste_b)]
            masks[index] = masks[index][setdiff(1:end, liste_b),setdiff(1:end, liste_a)]
            masks[index+1] = masks[index+1][setdiff(1:end, liste_b)]
            in_dim, out_dim = size(weights2)
            # Copie des neurones non nuls dans une nouvelle couche
            new_layer = Dense(out_dim, in_dim, layer.σ) 

            weights_new = Flux.params(new_layer)[1]
            biais_new = Flux.params(new_layer)[2]

            weights_new .= weights2
            biais_new .= biais2
            push!(list_layers, new_layer)

            index += 2
        else
            new_layer = layer 
            push!(list_layers, new_layer)
        end 
        taille_pre = taille
        taille = size(layer(matrice_test))
        matrice_test = zeros(taille)

    end
    matrice_test = nothing
    weights = nothing
    biais= nothing
    weights2 = nothing
    biais2= nothing
    m = Chain(list_layers)
    return m
end

function get_struct_Network_filter(model,size_data_input,locality)
    data_type = zeros(size_data_input)
    taille = size_data_input

    nb_w = 0
    nb_b = 0
    nb_filters = 0
    nb_neurons = 0
    list_e = []

    for (l, layer) in enumerate(model)
        taille = size(layer(data_type))
        data_type = zeros(taille)
        if isa(layer, Dense) || isa(layer, Conv)
            weights = Flux.params(layer)[1]
            biases = Flux.params(layer)[2]

            nb_w += length(weights)
            nb_b += length(biases)
        end

        if isa(layer, Dense)
            neurons = size(weights)
            nb_neurons += neurons[1]

            push!(list_e,neurons[1])
        elseif isa(layer, Conv)
            filters = size(weights)[end]
            filter_dims = size(weights)[1:end-1]
                push!(list_e,filters)
            nb_filters += filters
        end
    end

    
    if locality == "global" 
        list_e = [(nb_w + nb_b)]
    end
    return list_e, (nb_w + nb_b)
end
