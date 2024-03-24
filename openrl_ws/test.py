import isaacgym
from openrl_ws.utils import make_env, get_args, custom_cfg, MATWrapper

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent

import cv2
import imageio
import numpy as np

def save_video(frames, fps):
    # Assuming your ndarray is named 'frames'
    # frames.shape = (134, 4, 240, 360)

    # Define the output video file name
    output_video_path = 'output_video.mp4'

    # Define the video codec and frame rate
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # Get the shape of a single frame
    frame_shape = frames.shape[2], frames.shape[3]

    frames = frames[:, :3, :, :]
    frames = np.transpose(frames, (0, 2, 3, 1))

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, codec, fps, frame_shape)

    # Iterate through each frame
    for i in range(len(frames)):
        # Convert frame to uint8 (assuming it's in range 0-255)
        frame = frames[i]
        frame = frame.astype(np.uint8)
        
        # Transpose frame from (4, 240, 360) to (240, 360, 3) if needed
        # frame = np.transpose(frame, (1, 2, 0))
        
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print("Video created successfully.")

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

args = get_args()
env, _ = make_env(args, custom_cfg(args))
net = PPONet(env, device="cuda")  # Create neural network.
agent = PPOAgent(net)  # Initialize the agent.

if args.algo == "jrpo" or args.algo == "ppo":
    from openrl.modules.common import PPONet
    from openrl.runners.common import PPOAgent
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
else:
    from openrl.modules.common import MATNet
    from openrl.runners.common import MATAgent
    env = MATWrapper(env)
    net = MATNet(env, cfg=args, device=args.rl_device)
    agent = MATAgent(net, use_wandb=args.use_wandb)

if getattr(args, "checkpoint") is not None:
    agent.load(args.checkpoint)

# env.start_recording()
agent.set_env(env)  # The agent requires an interactive environment.
obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.
while True:
    action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)
    # if done[0, 0]:
    #     frames = env.get_complete_frames()
    #     video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
    #     print(video_array.shape)
    #     print(np.mean(video_array))
    #     save_gif(video_array, 1 / env.dt)
