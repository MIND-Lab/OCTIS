from octis.models.base_etm import BaseETM



class DETM(BaseETM):
    def __init__(self=num_topics=50, 
                 rho_size=300, 
                 emb_size=300, 
                 t_hidden_size=800, 
                 theta_act=relu, 
                 train_embeddings=1,
                 eta_nlayers=3,
                 eta_hidden_size=200,
                 delta=0.005)
