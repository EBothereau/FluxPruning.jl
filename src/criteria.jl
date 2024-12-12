#------------------------------------------------------------------------------------------
#                              Local Itératif Non structuré
#------------------------------------------------------------------------------------------

#sélection par magnitude



#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#RED++ functions

function RED_d(w,layer, input, output)
    Δl = maximum(layer)-minimum(layer)
    in = produit_liste(input)
    out = produit_liste(output)
    s = 0
    for i in 1:size(layer)[3]
        s+= GaussianKernel(w./Δl,layer[:,:,i]./Δl)
    end
    s = s./(in*out*Δl)
    return s
end



function produit_liste(liste)
    produit = 1
    for element in liste
        produit *= element
    end
    return produit
end
