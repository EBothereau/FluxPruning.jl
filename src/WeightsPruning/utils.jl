function number_pruned_weigth(r::RatioPrune,e_per_layer)
    list_to_prune = zeros(Int64,length(e_per_layer),1)
    for p in 1:(length(e_per_layer))
        if e_per_layer[p] >= 4
            to_remove = round(e_per_layer[p]*r.pruningRatio)
            if e_per_layer[p]-to_remove<4
                to_remove=e_per_layer[p]-4
            end
            quotient = div(to_remove, 1)
            remainder = mod(to_remove, 1)
            liste = [quotient for _ in 1:1]
            if remainder>0
               for i in 1:Int(remainder)
                    liste[i] += 1
                end
            end
            list_to_prune[p,:]=liste
        else 
            list_to_prune[p,:].=0
        end
    end
return list_to_prune
liste = nothing
quotient= nothing
remainder= nothing
to_remove= nothing
end


function get_struct_Network_weight(model,n)
    data_type = zeros(n.size_data_input)
    taille = n.size_data_input


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

            push!(list_e,length(weights))
        elseif isa(layer, Conv)
            filters = size(weights)[end]
            filter_dims = size(weights)[1:end-1]
            push!(list_e,length(weights))
            nb_filters += filters
        end
    end

    
    if n.typePruning == "global" 
        list_e = [(nb_w + nb_b)]
    end
    #println("nombre parametres du réseau $(nb_w + nb_b)")
    return list_e, (nb_w + nb_b)
end


function get_struct_Network_weights(model,size_data_input, locality)
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

            push!(list_e,length(weights))
        elseif isa(layer, Conv)
            filters = size(weights)[end]
            filter_dims = size(weights)[1:end-1]
            push!(list_e,length(weights))
            nb_filters += filters
        end
    end

    
    if locality == "global" 
        list_e = [(nb_w + nb_b)]
    end
    #println("nombre parametres du réseau $(nb_w + nb_b)")
    return list_e, (nb_w + nb_b)
end
