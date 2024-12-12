function get_pruning_mask_filter(model, list_to_prune, r::RatioPrune)
    threshold = []
    if r.locality=="local"
        if r.criterion == "L1"
            masks = create_masks_iter_filter_l1(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "L2"
            masks = create_masks_iter_filter_l2(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "geomedian"
            masks = create_masks_iter_filter_geomedian(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "mean"
            masks = create_masks_iter_filter_mean(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "median"
            masks = create_masks_iter_filter_median(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "cor"
            masks = create_masks_iter_filter_cor(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        elseif r.criterion == "random"
            masks = create_masks_iter_filter_random(model, list_to_prune)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        else
            error("Non valable")
        end
    elseif r.locality=="global"
        if r.criterion == "cop_0_0"
            masks = create_masks_iter_filter_cop_0_0(model, list_to_prune,r.data_in)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        else
            error("Non valable")
        end
    else 
        error("Non valable")
    end
    return masks,threshold
end
function get_pruning_mask_filter(model, r::Threshold)  # A finir d'implémeneter
        if r.criterion == "cop_0_0"
            masks = create_masks_iter_filter_cop_0_0(model, r)  #Modifier ici le % de réseau retiré et le nombre d'itérations
        else
            error("Non valable")
        end
    return masks
end
