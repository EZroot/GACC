import torch
import cv2
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

# load pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# optimize for GPU memory
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# generate
prompt = "Spiderman is surfing. Darth Vader is also surfing and following Spiderman"
video_frames = pipe(prompt, num_inference_steps=25, num_frames=10).frames

# convert to video
video_path = export_to_video(video_frames)

# save the video
output_path = "./vids/video.mp4"  # specify the desired output path and filename
frame_rate = 15  # specify the frame rate of the output video

# OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # specify the codec (e.g., mp4v, XVID, etc.)
video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (video_frames[0].shape[1], video_frames[0].shape[0]))

# write frames to the video file
for frame in video_frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert frame color format
    video_writer.write(frame)

# release the video writer
video_writer.release()

print("Video saved successfully!")
