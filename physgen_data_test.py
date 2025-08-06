
from cosmos_predict2.data.dataset_video import Dataset

dataset = Dataset(
    dataset_dir="datasets/physgen_train",
    num_frames=93,
    video_size=[256, 256],
)

indices = [0, 13, -1]
for idx in indices:
    data = dataset[idx]
    info_str = f"{idx=}\n"
    info_str += f"{data['video'].sum()=}\n"
    info_str += f"{data['video'].shape=}\n"
    info_str += f"{data['video_name']=}\n"
    info_str += f"{data['t5_text_embeddings'].shape=}\n"
    info_str += "---"
    print(info_str)