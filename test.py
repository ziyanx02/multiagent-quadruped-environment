
import numpy as np
import os

import imageio
import isaacgym
from mqe.utils import get_args
import torch

from mqe.envs.utils import make_mqe_env, custom_cfg

def save_gif(frames, fps):

    # Assuming your ndarray is named 'frames'
    # frames.shape = (134, 4, 240, 360)

    # Define the output GIF file name
    output_gif_path = 'output_animation.gif'

    # Convert the frames to uint8 (assuming it's in range 0-1)
    frames = np.transpose(frames, (0, 2, 3, 1))
    frames_uint8 = frames.astype(np.uint8)

    frames = [frames_uint8[i] for i in range(len(frames_uint8))]

    # Save frames as GIF
    imageio.mimsave(output_gif_path, frames, fps=fps)

    print("GIF created successfully.")

if __name__ == '__main__':
    args = get_args()

    # task_name = "go1plane"
    # task_name = "go1gate"
    # task_name = "go1football-defender"
    # task_name = "go1football-1vs1"
    # task_name = "go1football-2vs2"
    # task_name = "go1sheep-easy"
    # task_name = "go1sheep-hard"
    task_name = "go1seesaw"
    # task_name = "go1door"
    # task_name = "go1pushbox"

    args.num_envs = 1
    args.headless = False
    args.record_video = True

    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))
    env.start_recording()
    env.reset()
    import time
    while True:
        # obs, _, _, _ = env.step(0 * torch.tensor([[[1, 0, 0],[1, 0, 0],[1, 0, 0],[1, 0, 0],],],
        #                         dtype=torch.float32, device="cuda").repeat(env.num_envs, 1, 1))
        obs, _, done, _ = env.step(0 * torch.tensor([[[1, 0, 0],[1, 0, 0],],],
                                dtype=torch.float32, device="cuda").repeat(env.num_envs, 1, 1))
        if done.tolist()[0]:
            print("done")
            frames = env.get_complete_frames()
            if len(frames) == 0:
                continue
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            print(video_array.shape)
            save_gif(video_array, 1 / env.dt)
            exit()
