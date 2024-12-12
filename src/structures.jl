
# using Ratio to prune
struct RatioPrune
    pruningRatio::Real
    criterion::String
    locality::String
    data_in::Tuple{Vararg{Int}}
end

function RatioPrune(r::Real)
    r = RatioPrune(r,"L1","local", (256,2,1))
    return r
end
function RatioPrune(r::Real,norme::String,locality::String)
    r = RatioPrune(r,norme,locality, (256,2,1))
    return r
end

#pruning all elements under a threshold -> to be implemented !!
struct Threshold
    threshold::Real
    criterion::String
    data_in::Tuple{Vararg{Int}}
end

function Threshold(k::Real)
    m = Threshold(k,"L1",(256,2,1))
    return m
end
function Threshold(k::Real, norme)
    m = Threshold(k,norme,(256,2,1))
    return m
end

