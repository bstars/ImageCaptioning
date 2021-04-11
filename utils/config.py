class CONFIG:
    WORD2VEC_WINDOW_SIZE = 2
    WORD2VEC_BATCH_SIZE = 256
    WORD2VEC_LEARNING_RATE = 5e-3
    VOCAB_SIZE = 1004
    WORD2VEC_EMBED_DIM = 128


    MASK_WEIGHT = {
        "<NULL>" : 0.,
        "<UNK>" : 0.5,
        "with" : 0.95,
        'a' : 0.95
    }
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_CHANNEL = 3
    RNN_DIM = 512
    TIME_SPAN = 17
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 64
    LEARNING_RATE_DECAY = 0.999

    CKPT_PATH = '../ckpt1/checkpoint1'
    COCO_PATH = '../data'
