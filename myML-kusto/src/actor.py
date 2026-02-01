import os
import time
import pickle
import redis
import gymnasium as gym
import numpy as np
import torch
import io

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

def run_actor():
    r_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    env = gym.make("CartPole-v1")

    local_model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )
    print(f"Actor started. Connecting to Redis at {REDIS_HOST}")
    state, _ = env.reset()
    
    episode = 0
    while True:
        if episode % 100 == 0:
            model_bytes = r_conn.get("global_model_weights")
        if model_bytes:
            buffer = io.BytesIO(model_bytes)
            state_dict = torch.load(buffer)
            local_model.load_state_dict(state_dict)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = local_model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        transition = (state, action, reward, next_state, done)
        serialized_data = pickle.dumps(transition)
        
        r_conn.lpush("replay_buffer", serialized_data)

        state = next_state
        if done:
            state, _ = env.reset()
            episode += 1

if __name__ == "__main__":
    run_actor()