AUTOMATION_MODEL_DEFAULTS = {
    "hidden_features": 128,
    "nonlinearity": "relu",
    "p_dropout": 0.0,
    "conv_layer": "GCN",
    "pre_gnn": [128],
    "post_gnn": [256, 128],
    "variant": "sum",
}

AUTOMATION_TRAIN_DEFAULTS = {
    "runs": 10,
    "lr": 0.0005,
    "batch_size": 32,
    "patience": 50,
    "epochs": 500,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "seed_mode": "auto",
    "seed_base": 20260320,
    "seed_list": None,
    "allow_duplicate_seeds": False,
    "activation_checkpoint": False,
}
