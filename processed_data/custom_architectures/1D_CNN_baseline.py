from model_factory import MODEL_BUILDERS

def build_model(input_shape, num_classes, params):
    # Auto-generated wrapper for 1D_CNN
    builder = MODEL_BUILDERS['1D_CNN']
    return builder(input_shape, num_classes, params)
