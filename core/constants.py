import numpy as np

UP = np.array([0,1,0], dtype=np.float32)
FRONT = np.array([0,0,-1], dtype=np.float32)
EPSILON = 1e-8
MAX_LUMINANCE = 100.0
REL_LUMINANCE = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

TONE_MAPPING_IDS = { 'simple':0, 'aces':1, 'reinhard':2, 'uncharted2':3, 'none':4}

