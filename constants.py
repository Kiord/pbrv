import numpy as np

UP = np.array([0,1,0], dtype=np.float32)
FRONT = np.array([0,0,-1], dtype=np.float32)
EPSILON = 1e-8

class TexUnit:
    GBUFFER_POSITION = 0
    GBUFFER_NORMAL   = 1
    GBUFFER_ALBEDO   = 2
    GBUFFER_RMAO     = 3

    SSAO_NOISE       = 4
    SSAO             = 5
    SSAO_BLUR        = 6

    ALBEDO_MAP       = 7
    NORMAL_MAP       = 8 
    ROUGHNESS_MAP    = 9
    METALNESS_MAP  = 10
    AO_MAP           = 11