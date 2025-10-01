import torch
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.
pipe = Video2WorldPipeline.from_config(
    config=get_cosmos_predict2_video2world_pipeline(model_size="2B"),
    dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
)

# Specify the input image path and text prompt.
image_path = "./datasets/physgen_test_raw/input_physgen_0.png"
prompt = "Make a sound_reflection propagation with the sound source in the center of the image in one step."

# Run the video generation pipeline.
video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.
save_image_or_video(video, "./output/test.mp4", fps=16)


