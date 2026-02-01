import os
import time
import pickle
import redis
import torch
import torch.optim as optim
import io
import time
import signal # 종료 신호 처리를 위해 추가
import sys # 종료 처리를 위해 추가
from torch.utils.tensorboard import SummaryWriter # TensorBoard 추가

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
BATCH_SIZE = 64
GAMMA = 0.99
LOG_DIR = "/app/logs" # 로그 저장 경로 (도커 마운트 필요)

# 종료 플래그
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down Learner... Saving final model.")
    running = False

def run_learner():
    global running
    # 종료 신호 등록 (Docker stop 등)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    r_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    writer = SummaryWriter(LOG_DIR) # TensorBoard Writer 초기화
    
    # 모델 정의
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    print(f"Learner started. Connecting to Redis at {REDIS_HOST}")
    step = 0

    while running: # running 플래그 확인
        # 버퍼에 충분한 데이터가 쌓일 때까지 대기
        if r_conn.llen("replay_buffer") < BATCH_SIZE:
            time.sleep(0.1)
            continue
            
        transitions = []
        for _ in range(BATCH_SIZE):
            data = r_conn.rpop("replay_buffer")
            if data:
                transitions.append(pickle.loads(data))
        
        if not transitions:
            continue
            
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # 텐서 변환
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        q_values = model(states) 
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = model(next_states)
            max_next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + GAMMA * max_next_q_value * (1 - dones)
            
        loss = loss_fn(q_value, expected_q_value)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # TensorBoard 기록
        writer.add_scalar("Training/Loss", loss.item(), step)
        step += 1

        # 학습된 모델 가중치를 직렬화하여 Redis에 업로드 (주기적으로 업데이트)
        if step % 10 == 0: # 매 스텝마다 하면 느리니까 10번마다
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            r_conn.set("global_model_weights", buffer.getvalue())
            print(f"Step {step}: Model updated. Loss: {loss.item():.4f}")

    # 종료 시 최종 저장
    writer.close()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    r_conn.set("global_model_weights", buffer.getvalue())
    print("Final model saved to Redis. Bye!")

if __name__ == "__main__":
    import numpy as np
    run_learner()