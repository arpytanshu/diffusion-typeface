
config = {
    "run_name" : "DDPM_cond1",
    "epochs" : 500,
    "batch_size" : 16,
    "img_size" : 64,
    "dataset_path" : "/shared/datasets/english_typeface_classes/",
    "device" : "cuda",
    "lr" : 3e-4,
    "num_classes": 26,

    "ema_beta": 0.95,

    "perc_uncond_train": 0.1, # ratio of samples to train unconditionally
    "log_interval": 10,
    "log_batch": 10,

    # generation config
    "cfg_scale": 5
}