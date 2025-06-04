import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import cv2
import os
import matplotlib.pyplot as plt
from collections import deque
from tqdm import trange

# ==== Hyperparameters ====
ENV_NAME = "PongNoFrameskip-v4"
NUM_ENVS = 32
TOTAL_TIMESTEPS = 10000000#10000000
ROLLOUT_STEPS = 128
MINI_BATCH_SIZE = 256
PPO_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
LR = 2.5e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Frame preprocessing ====
def preprocess(obs):
    obs = obs[35:195]
    obs = cv2.resize(obs, (84, 84))
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return obs.astype(np.float32) / 255.0

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        for _ in range(self.k):
            self.frames.append(preprocess(obs))
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(preprocess(obs))
        return np.stack(self.frames, axis=0)

# ==== Make vectorized environments ====
def make_env():
    def _thunk():
        env = gym.make(ENV_NAME)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False)
        env = gym.wrappers.FrameStack(env, 4)
        return env
    return _thunk

from gym.vector import AsyncVectorEnv
envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

# ==== Neural Network ====
class PPOAgent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = self.net(x)
        return self.policy(x), self.value(x)

model = PPOAgent(envs.single_action_space.n).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

# ==== Storage ====
obs_shape = envs.single_observation_space.shape
obs = envs.reset()

obs = np.asarray(obs)
obs = torch.from_numpy(obs).float().to(DEVICE)

reward_history = []
episode_rewards = np.zeros(NUM_ENVS)
max_reward_achieved = -float("inf")

num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_STEPS)

for update in trange(num_updates, desc="Training"):
    storage = {
        'obs': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': []
    }

    for step in range(ROLLOUT_STEPS):
        with torch.no_grad():
            logits, value = model(obs)
            probs = Categorical(logits=logits)
            actions = probs.sample()
            logprobs = probs.log_prob(actions)

        next_obs, reward, done, info = envs.step(actions.cpu().numpy())
        storage['obs'].append(obs.cpu().numpy())
        storage['actions'].append(actions.cpu().numpy())
        storage['logprobs'].append(logprobs.cpu().numpy())
        storage['values'].append(value.cpu().numpy())
        storage['rewards'].append(reward)
        storage['dones'].append(done)

        obs = torch.from_numpy(next_obs).float().to(DEVICE)

        episode_rewards += reward
        for i, d in enumerate(done):
            if d:
                reward_history.append(episode_rewards[i])
                if episode_rewards[i] > max_reward_achieved:
                    max_reward_achieved = episode_rewards[i]
                    print(f"New max reward: {max_reward_achieved}")
                episode_rewards[i] = 0

    for k in storage:
        storage[k] = np.asarray(storage[k])

    obs_batch = torch.tensor(storage['obs'], dtype=torch.float32, device=DEVICE)
    actions_batch = torch.tensor(storage['actions'], device=DEVICE)
    logprobs_batch = torch.tensor(storage['logprobs'], device=DEVICE)
    values_batch = torch.tensor(storage['values'], device=DEVICE)
    rewards_batch = torch.tensor(storage['rewards'], device=DEVICE)
    dones_batch = torch.tensor(storage['dones'], device=DEVICE)

    advantages = torch.zeros_like(rewards_batch, device=DEVICE)
    last_advantage = 0
    with torch.no_grad():
        _, next_value = model(obs)
    for t in reversed(range(ROLLOUT_STEPS)):
        mask = 1.0 - dones_batch[t].float()
        delta = rewards_batch[t] + GAMMA * next_value.squeeze() * mask - values_batch[t].squeeze()
        advantages[t] = last_advantage = delta + GAMMA * GAE_LAMBDA * mask * last_advantage
        next_value = values_batch[t]
    returns = advantages + values_batch.squeeze()

    b_obs = obs_batch.reshape(-1, 4, 84, 84)
    b_actions = actions_batch.reshape(-1)
    b_logprobs = logprobs_batch.reshape(-1)
    b_returns = returns.reshape(-1)
    b_advantages = advantages.reshape(-1)

    inds = np.arange(b_obs.shape[0])
    for epoch in range(PPO_EPOCHS):
        np.random.shuffle(inds)
        for start in range(0, len(inds), MINI_BATCH_SIZE):
            end = start + MINI_BATCH_SIZE
            mb_inds = inds[start:end]

            logits, value = model(b_obs[mb_inds])
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_logprobs = dist.log_prob(b_actions[mb_inds])

            ratio = (new_logprobs - b_logprobs[mb_inds]).exp()
            surr1 = ratio * b_advantages[mb_inds]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages[mb_inds]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value.squeeze() - b_returns[mb_inds]) ** 2).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    if update % 10 == 0:
        torch.save(model.state_dict(), f"ppo_pong_checkpoint.pt")
        # print("Model saved.")
print("Training complete.")
reward_file_name = f"reward_curve_{str(TOTAL_TIMESTEPS)}.png"
video_name = f"ppo_pong_result_{str(TOTAL_TIMESTEPS)}.mp4"
# ==== Plot reward curve and save ====
plt.figure(figsize=(10,5))
plt.plot(reward_history)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("PPO Pong Training Reward")
plt.grid()
plt.savefig(reward_file_name)
plt.close()

# ==== Play with trained model and save video ====
from gym.wrappers import AtariPreprocessing, FrameStack

model.load_state_dict(torch.load("ppo_pong_checkpoint.pt"))

env = gym.make(ENV_NAME, render_mode="rgb_array")
env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
env = FrameStack(env, 4)

obs = env.reset()
done = False
frames = []

while not done:
    obs_tensor = torch.from_numpy(np.array(obs)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = model(obs_tensor)
        action = torch.argmax(logits, dim=1).item()
    obs, reward, done, info = env.step(action)

    frame = env.render()
    frames.append(frame)

env.close()

height, width, _ = frames[0].shape
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print(f"Video saved as {video_name}")