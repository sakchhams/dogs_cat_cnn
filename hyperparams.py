class Hyperparameters:
    """tunable hyperparameters and misc. data"""
    #misc data
    train_dir = 'train_01'

    #anything bigger than that would give me an OOM straight away
    filter_size = [5,5,5]
    num_filters = [32,32,64]
    #Dense layer
    dense_layer_size = 128

    #Image dimensions
    img_w, img_h = 400, 300
    img_size_flat = img_w * img_h
    img_shape = (img_w, img_h)
    color_channels = 3

    num_classes = 2 #cat 0 or dog 1
    batch_size = 20
    train_iters = 2