"""
Usage:
python example_infer.py --model-path deepseek-ai/deepseek-vl2-tiny
"""

import argparse
import dataclasses

import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.srt.server_args import ServerArgs
from sglang.utils import async_stream_and_merge, stream_and_merge

from PIL import Image
import requests


def main(
    server_args: ServerArgs,
):
    vlm = sgl.Engine(**dataclasses.asdict(server_args))

    image_token = "<image>"

    image2 = Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)
    image = Image.open(requests.get("https://raw.githubusercontent.com/deepseek-ai/DeepSeek-VL2/refs/heads/main/images/visual_grounding_1.jpeg", stream=True).raw)


    prompt = f"{image_token}\n What is in the image?" 

    sampling_params = {
        "temperature": 0.001,
        "max_new_tokens": 512,
    }

    import time
    start_time = time.time()
    output = vlm.generate(
        prompt=prompt,
        image_data=image,
        sampling_params=sampling_params,
    )
    end_time = time.time()
    print(f"Time taken to generate the response: {end_time - start_time} seconds")
    print(f"Number of tokens generated: {output['meta_info']['completion_tokens']}")
    print(f"Tokens per second: {output['meta_info']['completion_tokens'] / (end_time - start_time)}")

    print("===============================")
    print(f"Prompt: {prompt}")
    print(f"Generated text: {output['text']}")  
    
    for i in range(5):
        prompt = f"<image>\n Describe the image in detail." 
        start_time = time.time()
        output = vlm.generate(
            prompt=prompt,
            image_data=image2,
            sampling_params=sampling_params,
        )
        end_time = time.time()
        print(f"Time taken to generate the response: {end_time - start_time} seconds")
        print(f"Number of tokens generated: {output['meta_info']['completion_tokens']}")
        print(f"Tokens per second: {output['meta_info']['completion_tokens'] / (end_time - start_time)}")

        print("===============================")
        print(f"Prompt: {prompt}")
        print(f"Generated text: {output['text']}")

    vlm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    main(server_args)