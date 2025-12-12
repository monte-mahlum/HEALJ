module HEALJ

include("DataLoader.jl")

include("Layers/GCN.jl")
include("Layers/HGT.jl")
include("Layers/MLP.jl")

include("Losses.jl")
include("Train.jl")

export DataLoader
export GCNLayer, GCNEncoder, gcn_forward, perturb_nodes
export HGT
export MLPHead

export supervised_bce_loss, contrastive_loss, total_loss
export HEALModel, graph_embeddings, model_forward,
       train_step!, train!

end # module HEALJ