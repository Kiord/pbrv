import numpy as np

UP = np.array([0,1,0], dtype=np.float32)
FRONT = np.array([0,0,-1], dtype=np.float32)
EPSILON = 1e-8
MAX_LUMINANCE = 100.0
REL_LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

TONE_MAPPING_IDS = { 'simple':0, 'aces':1, 'reinhard':2, 'uncharted2':3, 'none':4}

class TexUnit:
    GBUFFER_POSITION = 0
    GBUFFER_NORMAL   = 1
    GBUFFER_ALBEDO   = 2
    GBUFFER_RMAOS    = 3
    GBUFFER_EMISSIVE = 4

    SSAO_NOISE       = 5
    SSAO             = 6
    SSAO_BLUR        = 7

    SHADOW_MAP       = 8

    ALBEDO_MAP       = 9
    NORMAL_MAP       = 10
    ROUGHNESS_MAP    = 11
    METALLIC_MAP     = 12
    EMISSIVE_MAP     = 13
    SPECULAR_MAP     = 14
    AO_MAP           = 15

    ENV_BACKGROUND   = 16
    ENV_IRRADIANCE   = 17
    ENV_SPECULAR     = 18


