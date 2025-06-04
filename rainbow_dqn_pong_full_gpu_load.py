import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import math
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from gym.wrappers import AtariPreprocessing, FrameStack
from gym.vector import AsyncVectorEnv
from collections import deque
torch.set_float32_matmul_precision('high')
ENV_NAME = "PongNoFrameskip-v4"
GAMMA = 0.99
N_STEPS = 3
BATCH_SIZE = 512
BUFFER_SIZE = 1000000
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 8000
V_MIN = -10
V_MAX = 10
ATOM_SIZE = 51
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 10000000
TOTAL_FRAMES = 10000000
EPSILON = 1e-6
NUM_ENVS = 64

os.makedirs("videos", exist_ok=True)
os.makedirs("models", exist_ok=True)

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.017)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.017)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(input, weight, bias)

class NStepReplayBuffer:
    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_info(self):
        R = 0
        for idx, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * reward
        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        return (state, action, R, next_state, done)

    def push(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return

        n_step_transition = self._get_n_step_info()

        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(n_step_transition)
            self.priorities.append(max_prio)
        else:
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = n_step_transition
            self.priorities[idx] = max_prio

    def sample(self, batch_size, beta):
        prios = np.array(self.priorities)
        probs = prios ** ALPHA
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + EPSILON

    def __len__(self):
        return len(self.buffer)
    
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()
        self.n_actions = n_actions
        self.atom_size = ATOM_SIZE
        self.support = torch.linspace(V_MIN, V_MAX, ATOM_SIZE).to("cuda")
        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            n_linear = self.feature(dummy_input).shape[1]
        self.value_stream = nn.Sequential(
            NoisyLinear(n_linear, 512), nn.ReLU(),
            NoisyLinear(512, self.atom_size)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_linear, 512), nn.ReLU(),
            NoisyLinear(512, n_actions * self.atom_size)
        )

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        feature = self.feature(x / 255.0)
        value = self.value_stream(feature).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(feature).view(-1, self.n_actions, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = nn.functional.softmax(q_atoms, dim=2).clamp(min=1e-3)
        return dist

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []

    def push(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            idx = len(self.buffer) % self.capacity
            self.buffer[idx] = transition
            self.priorities[idx] = max_prio

    def sample(self, batch_size, beta):
        prios = np.array(self.priorities)
        probs = prios ** ALPHA
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + EPSILON

    def __len__(self):
        return len(self.buffer)

def projection(next_dist, rewards, dones, gamma):
    batch_size = rewards.size(0)
    delta_z = (V_MAX - V_MIN) / (ATOM_SIZE - 1)
    support = torch.linspace(V_MIN, V_MAX, ATOM_SIZE).to("cuda")
    Tz = rewards.unsqueeze(1) + gamma * support * (1 - dones.unsqueeze(1))
    Tz = Tz.clamp(V_MIN, V_MAX)
    b = (Tz - V_MIN) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    offset = torch.linspace(0, (batch_size - 1) * ATOM_SIZE, batch_size).long().unsqueeze(1).expand(batch_size, ATOM_SIZE).to("cuda")
    m = torch.zeros_like(next_dist)
    m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return m

 
    
if __name__ == "__main__":
    def make_env():
        def _thunk():
            env = gym.make(ENV_NAME, render_mode="rgb_array")
            env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
            env = FrameStack(env, 4)
            return env
        return _thunk
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    state = envs.reset()
    input_shape = state[0].shape
    n_actions = envs.single_action_space.n

    current_model = torch.compile(RainbowDQN(input_shape, n_actions).to("cuda"))
    target_model = torch.compile(RainbowDQN(input_shape, n_actions).to("cuda"))
    target_model.load_state_dict(current_model.state_dict())
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    # replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
    replay_buffer = NStepReplayBuffer(BUFFER_SIZE, n_step=N_STEPS, gamma=GAMMA)

    episode_rewards = []
    episode_reward = np.zeros(NUM_ENVS)

    for frame_idx in tqdm(range(1, TOTAL_FRAMES + 1, NUM_ENVS)):
        state_v = torch.tensor(state, dtype=torch.float32).to("cuda")
        current_model.reset_noise()
        
        with torch.no_grad():
            q_values = current_model(state_v)
            actions = q_values.argmax(1).cpu().numpy()

        next_state, reward, done, _ = envs.step(actions)

        # 안정성: numpy 타입 확실히 고정
        reward = reward.astype(np.float32)
        done = done.astype(bool)
        actions = actions.astype(np.int64)

        # PER 저장 (index 보호 적용)
        for idx in range(NUM_ENVS):
            if actions[idx] < 0 or actions[idx] >= n_actions:
                continue  # invalid action index 방지
            replay_buffer.push((state[idx], actions[idx], reward[idx], next_state[idx], done[idx]))

        state = next_state

        # episode reward 기록 (옵션)
        for idx, d in enumerate(done):
            if d:
                episode_rewards.append(episode_reward[idx])
                episode_reward[idx] = 0
            else:
                episode_reward[idx] += reward[idx]

        # PER warmup 보호
        if len(replay_buffer) < 20000:#BATCH_SIZE:
            continue

        # PER 샘플링
        beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)
        states, actions_b, rewards_b, next_states, dones_b, indices, weights = replay_buffer.sample(BATCH_SIZE, beta)

        # Tensor 변환
        states = torch.tensor(states, dtype=torch.float32).to("cuda")
        next_states = torch.tensor(next_states, dtype=torch.float32).to("cuda")
        actions_b = torch.tensor(actions_b).to("cuda")
        rewards_b = torch.tensor(rewards_b, dtype=torch.float32).to("cuda")
        dones_b = torch.tensor(dones_b, dtype=torch.float32).to("cuda")
        weights = torch.tensor(weights, dtype=torch.float32).to("cuda")

        # 모델 업데이트
        current_model.reset_noise()
        target_model.reset_noise()

        dist = current_model.dist(states)
        dist = dist.gather(1, actions_b.unsqueeze(1).unsqueeze(1).expand(-1, 1, ATOM_SIZE)).squeeze(1)

        next_dist = target_model.dist(next_states)
        next_action = torch.sum(next_dist * current_model.support, dim=2).argmax(1)
        next_dist = next_dist[range(BATCH_SIZE), next_action]
        target_dist = projection(next_dist, rewards_b, dones_b, GAMMA)

        loss = -(target_dist * dist.log()).sum(1)
        prios = loss + EPSILON
        loss = (loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        replay_buffer.update_priorities(indices, prios.detach().cpu().numpy())

        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(current_model.state_dict())

    envs.close()

    # 저장 및 그래프
    video_file_name = f"rainbow_pong_play_{str(TOTAL_FRAMES)}.mp4"
    graph_file_name = f"rainbow_pong_reward_{str(TOTAL_FRAMES)}.png"
    torch.save(current_model.state_dict(), "models/rainbow_pong.pt")
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(graph_file_name)
    plt.close()

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = AtariPreprocessing(env, frame_skip=4, scale_obs=False)
    env = FrameStack(env, 4)
    state = env.reset()
    frames = []

    done = False
    while not done:
        frame = env.render()
        frames.append(frame)
        state_v = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to("cuda")
        q_values = current_model(state_v)
        action = q_values.argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state

    env.close()

    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()