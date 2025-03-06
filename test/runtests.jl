using FluxPruning
using Flux

Structured_Local_criterions = ["L1", "L2", "geomedian","mean","median","random"]
Structured_Global_criterions = ["cop_0_0"]


Unstructured_Local_criterions = ["L1", "random"]
Unstructured_Global_criterions = ["lamp", "L1", "SynFlow"]

@testset "Structured_Local_2d" begin
    for criterion in Structured_Local_criterions
        m = Chain(Conv((3, ), 1 => 32), Conv((5, ), 32 => 32),Flux.flatten,  Dense(8000, 10))
        r = RatioPrune(0.5,criterion,"local", (256,1,1))
        mpruned = filterPruning(m,r)
        @test mpruned isa Chain
        @test mpruned != m
    end
end

#cop_0_0 est a corriger
#=@testset "Structured_Global_2d" begin
    for criterion in Structured_Global_criterions
        m = Chain(Conv((3, ), 1 => 32), Conv((5, ), 32 => 32),Flux.flatten,  Dense(8000, 10))
        r = RatioPrune(0.5,criterion,"global", (256,1,1))
        mpruned = filterPruning(m,r)
        @test mpruned isa Chain
        @test mpruned != m
    end
end=#





@testset "Unstructured_Local_2d" begin
    for criterion in Unstructured_Local_criterions
        m = Chain(Conv((3, ), 1 => 32), Conv((5, ), 32 => 32),Flux.flatten,  Dense(8000, 10))
        r = RatioPrune(0.5,criterion,"local", (256,1,1))
        m,masks = weightPruning(m,r)
        apply_masks(m,masks)
        @test m isa Chain
        #@test Flux.params(m) != Flux.params(mpruned)
    end
end

@testset "Unstructured_Global_2d" begin
    for criterion in Unstructured_Global_criterions
        m = Chain(Conv((3, ), 1 => 32), Conv((5, ), 32 => 32),Flux.flatten,  Dense(8000, 10))
        r = RatioPrune(0.5,criterion,"global", (256,1,1))
        m,masks = weightPruning(m,r)
        apply_masks(m,masks)
        @test m isa Chain
        #@test Flux.params(m) != Flux.params(mpruned)
    end
end

@testset "Structured_Local_3d" begin
    for criterion in Structured_Local_criterions
       m = Chain(Conv((3,1 ), 1 => 32), Conv((5,1 ), 32 => 32),Flux.flatten,  Dense(8000, 10))
       r = RatioPrune(0.5,criterion,"local", (256,1,1,1))
       mpruned = filterPruning(m,r)
       @test mpruned isa Chain
       @test Flux.params(mpruned) != Flux.params(m)
    end
end

#@testset "Structured_Global_3d" begin
#     for criterion in Structured_Global_criterions
#        m = Chain(Conv((3,1 ), 1 => 32), Conv((5,1 ), 32 => 32),Flux.flatten,  Dense(8000, 10))
#        r = RatioPrune(0.5,criterion,"global", (256,1,1,1))
#        mpruned = filterPruning(m,r)
#        @test mpruned isa Chain
#        @test Flux.params(mpruned) != Flux.params(m)
#    end
#end
@testset "Unstructured_Local_3d" begin
     for criterion in Unstructured_Local_criterions
        m = Chain(Conv((3,1 ), 1 => 32), Conv((5,1 ), 32 => 32),Flux.flatten,  Dense(8000, 10))
        r = RatioPrune(0.5,criterion,"local", (256,1,1,1))
        m,masks = weightPruning(m,r)
        apply_masks(m,masks)
        @test m isa Chain
        #@test Flux.params(m) != Flux.params(mpruned)
    end
end

@testset "Unstructured_Global_3d" begin
     for criterion in Unstructured_Global_criterions
        m = Chain(Conv((3,1 ), 1 => 32), Conv((5,1 ), 32 => 32),Flux.flatten,  Dense(8000, 10))
        r = RatioPrune(0.5,criterion,"global", (256,1,1,1))
        m,masks = weightPruning(m,r)
        apply_masks(m,masks)
        @test m isa Chain
        #@test Flux.params(m) != Flux.params(mpruned)
    end
end
