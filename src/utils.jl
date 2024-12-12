


#------------------------------------------------------------------------------------------------------------------------------------


function get_S_C_penalty(model, input_size) 
    S_list = []
    C_list = []
    taille = input_size
    taille_precedent = input_size
    data_type = zeros(taille)
    flag = 0 
    S_p = 0
    C_p = 0
    for (l, layer) in enumerate(model)
        S = 0
        C = 0
        taille_precedent = taille
        taille = size(layer(data_type))
        data_type = zeros(taille)

        if isa(layer, Conv) 
            dim_weights = size(Flux.params(layer)[1])
            nb_filtre = dim_weights[3]
            dim_filtre = dim_weights[1]
            nb_channels = dim_weights[2] 

            if flag ==0
                flag = 1
            else 
                S = S_p +  dim_filtre*nb_filtre
                C = C_p + dim_filtre*nb_filtre*taille[1]*taille[2]
                push!(S_list,S)
                push!(C_list,C)    
            end

            S_p = dim_filtre*nb_channels
            C_p =  dim_filtre*nb_channels*taille[1]*taille[2]

        elseif  isa(layer, Dense)
            dim_weights = size(Flux.params(layer)[1])
            nb_neurones = dim_weights[2]
            nb_connection_neurone = dim_weights[1]

            S = S_p + round((taille[1]*taille[2])/nb_filtre)*nb_neurones
            C = C_p + round((taille[1]*taille[2])/nb_filtre)*nb_neurones
            push!(S_list,S)
            push!(C_list,C) 

            S_p = (taille[1]*taille[2])*nb_neurones
            C_p = (taille[1]*taille[2])*nb_neurones

        end

        S = S_p + round((taille[1]*taille[2])/nb_filtre)*nb_neurones
        C = C_p + round((taille[1]*taille[2])/nb_filtre)*nb_neurones
        push!(S_list,S)
        push!(C_list,C)

    end
    return S_list, C_list
end

#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
