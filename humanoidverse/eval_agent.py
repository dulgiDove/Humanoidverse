import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    logger.add(sys.stdout, level="INFO", colorize=True)

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    # ── 체크포인트 경로 및 학습 config 로드 ──
    if override_config.checkpoint is None:
        logger.error("checkpoint 경로를 지정해주세요: +checkpoint=logs/.../model_XXX.pt")
        return

    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
    if not config_path.exists():
        logger.error(f"학습 config를 찾을 수 없습니다: {config_path}")
        return

    logger.info(f"학습 config 로드: {config_path}")
    with open(config_path) as f:
        train_config = OmegaConf.load(f)

    if train_config.get("eval_overrides") is not None:
        train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

    config = OmegaConf.merge(train_config, override_config)

    # ── WSL에서 뷰어 창 없이 headless로 실행 (MP4 파일로만 저장) ──
    config.headless = True
    config.num_envs = 1

    simulator_type = config.simulator['_target_'].split('.')[-1]
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401

    import torch
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo
    from humanoidverse.utils.helpers import pre_process_config

    pre_process_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")

    ckpt_num = checkpoint.stem.split('_')[-1]  # "model_100.pt" -> "100"
    video_path = str(checkpoint.parent / f"progress_ckpt_{ckpt_num}.mp4")
    config.env.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")

    env = instantiate(config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(str(checkpoint))

    # ── eval 준비 ──
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    eval_policy = algo._get_inference_policy()

    obs_dict = env.reset_all()
    all_envs = torch.arange(env.num_envs, device=device)
    env._resample_target(all_envs)
    env._resample_obstacles(all_envs)

    # ── 녹화 시작 ──
    logger.info(f"녹화 시작 -> {video_path}")
    env.simulator.start_recording(filename=video_path)

    MAX_STEPS = 1500  # 약 30초 (50Hz 기준)
    init_actions = torch.zeros(env.num_envs, algo.num_act, device=device)
    actor_state = {
        "obs": obs_dict,
        "actions": init_actions,
        "done_indices": [],
        "stop": False,
    }

    for step in range(MAX_STEPS):
        actor_state["step"] = step
        with torch.no_grad():
            actions = eval_policy(actor_state["obs"]["actor_obs"])
        actor_state["actions"] = actions
        actor_state = algo.env_step(actor_state)

    env.simulator.stop_recording()
    logger.info(f"녹화 완료! 저장 위치: {video_path}")


if __name__ == "__main__":
    main()
