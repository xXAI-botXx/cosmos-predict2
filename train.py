from cosmos_predict2.configs.base.utils import cs, L, Dataset, DataLoader, get_sampler


# --------------------------------
# Define the dataset in the config
# --------------------------------
example_video_dataset = L(Dataset)(
    dataset_dir="datasets/custom_video2world_dataset",
    num_frames=93,                     # Number of frames per sample
    video_size=(704, 1280),            # Resolution (H, W) of your videos
)

dataloader_video_train = L(DataLoader)(
    dataset=example_video_dataset,
    sampler=L(get_sampler)(dataset=example_video_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# -------------------------------
# Register your experiment config
# -------------------------------
predict2_video2world_training_2b_custom_data = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_custom_data",
    ),
    ...
    dataloader_train=dataloader_video_train,
    ...
)

cs.store(
    group="experiment",
    package="_global_",
    name="2b_custom_data",
    node=predict2_video2world_training_2b_custom_data,
)


