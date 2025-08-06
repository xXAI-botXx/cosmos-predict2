# -------
# Imports
# -------
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L

def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()



# -----------
# Data Config
# -----------
physgen_train_dataset = L(Dataset)(
    dataset_dir="datasets/physgen_train",
    num_frames=93, # 1 Input Frame + 1 Target Frame -> 1 F/S -> 2 Seconds Video ==> 1 Input Frame
                    # BUT during tokenization buffer got added with 93 frames, so 93
    video_size=(256, 256),  # 256 resolution, 1:1 aspect ratio
)

dataloader_physgen_train = L(DataLoader)(
    dataset=physgen_train_dataset,
    sampler=L(get_sampler)(dataset=physgen_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

physgen_val_dataset = L(Dataset)(
    dataset_dir="datasets/physgen_train",
    num_frames=93, # 1 Input Frame + 1 Target Frame -> 1 F/S -> 2 Seconds Video ==> 1 Input Frame
                    # BUT during tokenization buffer got added with 93 frames, so 93
    video_size=(256, 256),  # 256 resolution, 1:1 aspect ratio
)

dataloader_physgen_val = L(DataLoader)(
    dataset=physgen_val_dataset,
    sampler=L(get_sampler)(dataset=physgen_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)



# -----------------
# Experiment Config
# -----------------
predict2_video2world_training_1a_physgen = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        # {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="1a_physgen",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),     # enable EMA during training
                prompt_refiner_config=dict(enabled=False),  # disable prompt refiner during training
                guardrail_config=dict(enabled=False),   # disable guardrail during training
                max_num_conditional_frames=1,
                min_num_conditional_frames=1,
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,            # context parallelism size
    ),
    dataloader_train=dataloader_physgen_train,
    dataloader_val=dataloader_physgen_val,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=19000*10,                      # maximum number of iterations
    ),
    checkpoint=dict(
        save_iter=1000,                      # checkpoints will be saved every 500 iterations.
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[20_000],              # adjust considering max_iter
        f_max=[0.99],
        f_min=[0.0],
    ),
)



# ----------------
# Save Data Config
# ----------------
for _item in [
    # 2b, custom data
    predict2_video2world_training_1a_physgen,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
