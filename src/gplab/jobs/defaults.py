AUTOMATION_MODEL_DEFAULTS = {
    "hidden_features": 128,
    "nonlinearity": "relu",
    "p_dropout": 0.0,
    "conv_layer": "GCN",
    "pre_gnn": [128],
    "post_gnn": [256, 128],
    "variant": "sum",
}

AUTOMATION_TRAINING_DEFAULTS = {
    "runs": 10,
    "lr": 0.0005,
    "batch_size": 32,
    "patience": 50,
    "epochs": 500,
    "split": {
        "train": 0.8,
        "val": 0.1,
    },
    "seeds": {
        "mode": "auto",
        "base": 20260320,
        "values": None,
        "allow_duplicates": False,
    },
}

AUTOMATION_EXECUTION_DEFAULTS = {
    "log_file": None,
    "tag": None,
    "activation_checkpoint": False,
}
