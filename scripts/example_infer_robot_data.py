"""
Usage:
python example_infer_robot_data.py --model-path deepseek-ai/deepseek-vl2-tiny
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.srt.server_args import ServerArgs
from sglang.utils import async_stream_and_merge, stream_and_merge

from PIL import Image
import requests
import cv2
import time


def main(
    server_args: ServerArgs,
    video_path: str = "scripts/top_camera-images-rgb_low_res.mp4",
    stereo: bool = False,
):
    vlm = sgl.Engine(**dataclasses.asdict(server_args))

    image_token = "<image>"

    assert video_path.endswith(".mp4"), "Video path must be a mp4 file"
    video_capture = cv2.VideoCapture(video_path)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Read video failed, check video path")
            break

        prompt = f"{image_token}\n What is in the image?" 

        sampling_params = {
            "temperature": 0.001,
            "max_new_tokens": 512,
        }

        # numpy to PIL image
        frame = Image.fromarray(frame)
        start_time = time.time()

        output = vlm.generate(
            prompt=prompt,
            image_data=frame,
            sampling_params=sampling_params,
        )
        end_time = time.time()
        print(f"Time taken to generate the response: {end_time - start_time} seconds")
        print(f"Number of tokens generated: {output['meta_info']['completion_tokens']}")
        print(f"Tokens per second: {output['meta_info']['completion_tokens'] / (end_time - start_time)}")

        print("===============================")
        print(f"Prompt: {prompt}")
        print(f"Generated text: {output['text']}")  
        import pdb; pdb.set_trace()
        
    vlm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    main(server_args)