function get_pruning_mask_weight(model, list_to_prune, r::RatioPrune)
    threshold = []
    if r.locality =="local"
        if r.criterion == "L1"
            masks = create_masks_iter_loc_l1(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "random"
            masks = create_masks_random(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        else
            error("Non valable")
        end
    elseif r.locality =="global"
        if r.criterion == "lamp"
                masks, threshold = create_masks_lamp(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "L1"
                masks = create_masks_global_iter_l1(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "SynFlow"
                masks = create_masks_SynFlow(model, list_to_prune,r)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "random"
            masks = create_masks_random(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        else 
            error("Non valable")
        end
    else 
        error("Non valable")
    end
    return masks,threshold
end

function get_pruning_mask_weight(model, r::Threshold)
    threshold = []
        if r.criterion == "L1"
            masks = create_masks_l1(model, r)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "lamp"
                masks, threshold = create_masks_lamp(model, r)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "SynFlow"
                masks = create_masks_SynFlow(model, r)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        else 
            error("Non valable")
        end
    return masks,threshold
end
