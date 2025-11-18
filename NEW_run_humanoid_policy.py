import time
import numpy as np

try:
    import gymnasium as gym
    GYM_IS_GYMNASIUM = True
except ImportError:
    import gym
    GYM_IS_GYMNASIUM = False

from vlmrm.contrib.sb3.clip_rewarded_sac import CLIPRewardedSAC

ENV_NAME = "Humanoid-v4"
MODEL_PATH = "/home/han/vlmrm_runs/training/Humanoid_CLIP_2025-11-18_16-27-18_b773a1e9/checkpoints/final_model.zip"


def make_env():
    """Humanoid 환경을 human 렌더 모드로 생성."""
    if GYM_IS_GYMNASIUM:
        env = gym.make(ENV_NAME, render_mode="human")
    else:
        # 옛 gym 버전일 경우
        env = gym.make(ENV_NAME)
    return env


def main():
    # 1) 환경 생성
    env = make_env()

    # 2) 학습된 SAC 모델 로드 (이 안에서 Monitor/DummyVecEnv로 래핑될 수 있음)
    model = CLIPRewardedSAC.load(MODEL_PATH, env=env)

    # stable-baselines3 계열이면 내부 VecEnv 를 이렇게 꺼낼 수 있음
    if hasattr(model, "get_env"):
        wrapped_env = model.get_env()
        if wrapped_env is not None:
            env = wrapped_env

    # 3) 실시간 롤아웃 루프
    obs = env.reset()
    # Gymnasium 원시 env일 경우 reset이 (obs, info)를 줄 수 있으므로 처리
    if isinstance(obs, tuple):
        obs, _ = obs

    episode_idx = 1
    episode_return = 0.0
    episode_steps = 0

    while True:
        # 정책에서 액션 뽑기 (deterministic=True로 noise 제거)
        action, _ = model.predict(obs, deterministic=True)

        # VecEnv 이면 obs, reward, done, info가 array/list 형태로 나옴
        obs, reward, done, info = env.step(action)

        # 첫 번째 환경만 사용한다고 가정
        if isinstance(reward, (list, np.ndarray)):
            r0 = float(reward[0])
            d0 = bool(done[0])
        else:
            r0 = float(reward)
            d0 = bool(done)

        episode_return += r0
        episode_steps += 1

        # 관측도 VecEnv / Gymnasium 모두 커버
        if isinstance(obs, tuple):
            obs, _ = obs

        # MuJoCo 뷰어 업데이트
        # (VecEnv일 경우 첫 번째 env의 render가 호출됨)
        env.render()

        # 에피소드 종료 시 로그 출력 및 reset
        if d0:
            print(f"[Episode {episode_idx}] steps = {episode_steps}, return = {episode_return:.2f}")
            episode_idx += 1
            episode_return = 0.0
            episode_steps = 0

            obs = env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs

        # 너무 빨리 돌지 않도록 (약 60 FPS)
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
