from .vit_pytorch import ViT
from .vit_pytorch.t2t import T2TViT
from .vit_pytorch.cross_vit import CrossViT
from .vit_pytorch.crossformer import CrossFormer



def m_Vit(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']
    model = ViT(image_size = image_size,
                patch_size = 32,
                channels = in_chans,
                num_classes = num_classes,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
            )
    return model

def m_T2TViT(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']
    model = T2TViT(dim = 512,
                image_size = image_size,
                channels =in_chans,
                depth = 5,
                heads = 8,
                mlp_dim = 512,
                num_classes = num_classes,
                t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
                )
    return model

def m_CrossViT(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']
    model = CrossViT(
                    image_size = image_size,
                    num_classes = num_classes,
                    depth = 4,               # number of multi-scale encoding blocks
                    sm_dim = 192,            # high res dimension
                    sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
                    sm_enc_depth = 2,        # high res depth
                    sm_enc_heads = 8,        # high res heads
                    sm_enc_mlp_dim = 2048,   # high res feedforward dimension
                    lg_dim = 384,            # low res dimension
                    lg_patch_size = 64,      # low res patch size
                    lg_enc_depth = 3,        # low res depth
                    lg_enc_heads = 8,        # low res heads
                    lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
                    cross_attn_depth = 2,    # cross attention rounds
                    cross_attn_heads = 8,    # cross attention heads
                    dropout = 0.1,
                    emb_dropout = 0.1
    )
    return model


def m_CrossFormer(model_params):
    """Constructs a large TResnet model.
    """
    in_chans = model_params['args'].in_chans
    num_classes = model_params['num_classes']
    image_size =   model_params['image_size']
    model = CrossFormer(
                    image_size = image_size,
                    num_classes = num_classes,                # number of output classes
                    dim = (64, 128, 256, 512),         # dimension at each stage
                    depth = (2, 2, 8, 2),              # depth of transformer at each stage
                    global_window_size = (8, 4, 2, 1), # global window sizes at each stage
                    local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
    )
    return model