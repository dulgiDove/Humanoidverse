import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.agents.modules.amp_discriminator import AMPDiscriminator
from loguru import logger


class AMPPPO(PPO):
    """
    PPO + AMP (Adversarial Motion Priors) 스타일 학습.

    기존 style_imitation (frame-by-frame exp 보상) 대신,
    discriminator가 policy 전이와 IK npz 전이를 구분하면서
    style 보상 신호를 생성한다.

    추가 요소:
      - AMPDiscriminator: (dof_pos_t, dof_pos_{t+1}) -> real/fake logit
      - motion_pairs: lafan1_walk_h1_ik.npz에서 빌드한 (s, s') 실제 전이
      - 롤아웃 중 AMP 보상을 기존 보상에 더해서 저장
      - 매 iteration discriminator를 1회 업데이트

    [2026.08.23] fake(정책) 전이용 replay buffer 추가. AMP 논문/IsaacGymEnvs 구현 모두
    "과거 여러 iteration에서 쌓인 정책 전이를 replay buffer에 모아두고 거기서 샘플링"하는
    방식을 씀 — 매번 "이번 iteration 롤아웃"만 fake로 쓰면 discriminator가 그 좁은
    분포에만 맞춰 너무 쉽게/빨리 과적합(=우리가 겪은 조기 붕괴)하기 때문. 여기서도 동일하게
    적용.
    """

    def setup(self):
        super().setup()
        self._setup_discriminator()
        self._setup_motion_dataset()

    def _setup_storage(self):
        super()._setup_storage()
        dof_dim = self.env.config.robot.actions_dim
        self.storage.register_key('dof_pos', shape=(dof_dim,), dtype=torch.float)
        if self.config.amp.get("use_velocity", False):
            self.storage.register_key('dof_vel', shape=(dof_dim,), dtype=torch.float)

    def _setup_discriminator(self):
        dof_dim = self.env.config.robot.actions_dim
        hidden_dims = tuple(self.config.amp.disc_hidden_dims)
        # [2026.08.28] AMP가 관절 위치(dof_pos)만 보고 속도는 안 보다 보니, 정책이 참조
        # 데이터보다 훨씬 빠른 속도(실측 1.06 m/s vs 참조 0.42~0.76 m/s)로 같은 걷기 자세
        # 시퀀스를 재생해도 discriminator가 못 잡아냄 -> 뛰는 것처럼 보이는 원인 중 하나로
        # 판단. AMP 논문 원래 설계(관측에 속도 포함)대로 dof_vel도 state에 포함시킴.
        self.use_velocity = self.config.amp.get("use_velocity", False)
        self.state_dim = dof_dim * 2 if self.use_velocity else dof_dim
        self.discriminator = AMPDiscriminator(
            obs_dim=self.state_dim * 2,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.amp.disc_learning_rate,
        )
        self.amp_reward_weight = self.config.amp.amp_reward_weight
        self.disc_grad_penalty_coef = self.config.amp.get("grad_penalty_coef", 10.0)
        self.disc_logit_reg_coef = self.config.amp.get("logit_reg_coef", 0.05)
        self.disc_weight_decay_coef = self.config.amp.get("weight_decay_coef", 0.0001)
        self.disc_grad_norm = self.config.amp.get("disc_grad_norm", 0.5)
        self.disc_gp_type = self.config.amp.get("gp_type", "r1")

        # fake(정책) 전이 replay buffer: 최근 여러 iteration에 걸친 정책 전이를 모아둠
        # [2026.08.25] gp_type/logit_reg/weight_decay/replay_buffer를 각각 켜고 끌 수 있게
        # 만들어서, R1 vs WGAN-GP + 정규화 유무를 조합별로 동시에 비교 실험할 수 있게 함
        # (걸음 리듬 문제의 원인이 정확히 뭔지 좁히기 위함).
        self.use_replay_buffer = self.config.amp.get("use_replay_buffer", True)
        replay_size = self.config.amp.get("replay_buffer_size", 100_000)
        replay_keep_prob = self.config.amp.get("replay_keep_prob", 0.1)
        self.replay_keep_prob = replay_keep_prob
        self.fake_replay_buffer = torch.zeros(replay_size, self.state_dim * 2, device=self.device)
        self.replay_buffer_size = replay_size
        self.replay_buffer_ptr = 0
        self.replay_buffer_filled = 0

        logger.info(
            f"[AMP] Discriminator: input={self.state_dim*2}, hidden={hidden_dims}, "
            f"use_velocity={self.use_velocity}, "
            f"disc_lr={self.config.amp.disc_learning_rate}, "
            f"amp_reward_weight={self.amp_reward_weight}, "
            f"gp_type={self.disc_gp_type}, "
            f"grad_penalty_coef={self.disc_grad_penalty_coef}, "
            f"logit_reg_coef={self.disc_logit_reg_coef}, "
            f"weight_decay_coef={self.disc_weight_decay_coef}, "
            f"use_replay_buffer={self.use_replay_buffer}, "
            f"replay_buffer_size={replay_size}, replay_keep_prob={replay_keep_prob}"
        )

    def _push_to_replay_buffer(self, fake_pairs: torch.Tensor):
        """새 정책 전이를 확률적으로 replay buffer에 기록 (FIFO, IsaacGymEnvs 방식)."""
        keep_mask = torch.bernoulli(
            torch.full((fake_pairs.shape[0],), self.replay_keep_prob, device=self.device)
        ).bool()
        kept = fake_pairs[keep_mask]
        n = kept.shape[0]
        if n == 0:
            return
        if n >= self.replay_buffer_size:
            self.fake_replay_buffer[:] = kept[-self.replay_buffer_size:]
            self.replay_buffer_ptr = 0
            self.replay_buffer_filled = self.replay_buffer_size
            return
        end = self.replay_buffer_ptr + n
        if end <= self.replay_buffer_size:
            self.fake_replay_buffer[self.replay_buffer_ptr:end] = kept
        else:
            first = self.replay_buffer_size - self.replay_buffer_ptr
            self.fake_replay_buffer[self.replay_buffer_ptr:] = kept[:first]
            self.fake_replay_buffer[:n - first] = kept[first:]
        self.replay_buffer_ptr = end % self.replay_buffer_size
        self.replay_buffer_filled = min(self.replay_buffer_size, self.replay_buffer_filled + n)

    def _sample_replay_buffer(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.replay_buffer_filled, (n,), device=self.device)
        return self.fake_replay_buffer[idx]

    def _setup_motion_dataset(self):
        motion_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'data', 'motions', 'lafan1_walk_h1_ik.npz'
        )
        motion_path = os.path.normpath(motion_path)
        npz = np.load(motion_path)
        clips = torch.tensor(npz['clips'], dtype=torch.float32)   # (1443, 200, 19)
        lengths = torch.tensor(npz['lengths'], dtype=torch.long)  # (1443,)
        # clips는 50Hz로 리샘플된 것 (convert_gmr_to_npz.py TARGET_FPS=50) -> dt=1/50
        clip_dt = 1.0 / 50.0

        pairs_list = []
        for i in range(clips.shape[0]):
            L = lengths[i].item()
            if L < 2:
                continue
            pos = clips[i, :L, :]              # (L, 19)
            s = pos[:-1]                        # (L-1, 19)
            s_next = pos[1:]                    # (L-1, 19)
            if self.use_velocity:
                # finite difference로 근사 속도 계산 (dof_vel 직접 저장 안 되어 있으므로)
                vel = (pos[1:] - pos[:-1]) / clip_dt  # (L-1, 19), s_next 시점 속도로 사용
                # s 시점 속도는 그 이전 프레임과의 차분 (첫 프레임은 s_next 속도로 대체)
                vel_prev = torch.cat([vel[:1], vel[:-1]], dim=0)
                s_full = torch.cat([s, vel_prev], dim=-1)          # (L-1, 38)
                s_next_full = torch.cat([s_next, vel], dim=-1)     # (L-1, 38)
                pairs_list.append(torch.cat([s_full, s_next_full], dim=-1))  # (L-1, 76)
            else:
                pairs_list.append(torch.cat([s, s_next], dim=-1))  # (L-1, 38)

        self.motion_pairs = torch.cat(pairs_list, dim=0).to(self.device)
        logger.info(f"[AMP] Motion dataset: {self.motion_pairs.shape[0]} transition pairs loaded (use_velocity={self.use_velocity})")

    def _sample_motion_pairs(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.motion_pairs.shape[0], (n,), device=self.device)
        return self.motion_pairs[idx]  # (n, state_dim*2)

    def _rollout_step(self, obs_dict):
        dof_dim = self.env.config.robot.actions_dim

        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # t 시점 dof_pos(+dof_vel) 저장 (env.step 호출 전)
                dof_pos_t = self.env.simulator.dof_pos[:, :dof_dim].clone()
                if self.use_velocity:
                    dof_vel_t = self.env.simulator.dof_vel[:, :dof_dim].clone()

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])
                for key in policy_state_dict.keys():
                    self.storage.update_key(key, policy_state_dict[key])

                # dof_pos(+dof_vel) 저장 (discriminator 학습용)
                self.storage.update_key('dof_pos', dof_pos_t)
                if self.use_velocity:
                    self.storage.update_key('dof_vel', dof_vel_t)

                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # t+1 시점 dof_pos(+dof_vel)로 AMP 보상 계산
                dof_pos_t1 = self.env.simulator.dof_pos[:, :dof_dim]
                if self.use_velocity:
                    dof_vel_t1 = self.env.simulator.dof_vel[:, :dof_dim]
                    state_t = torch.cat([dof_pos_t, dof_vel_t], dim=-1)
                    state_t1 = torch.cat([dof_pos_t1, dof_vel_t1], dim=-1)
                else:
                    state_t, state_t1 = dof_pos_t, dof_pos_t1
                amp_reward = self.discriminator.compute_reward(state_t, state_t1)
                rewards = rewards + self.amp_reward_weight * self.env.dt * amp_reward

                self.episode_env_tensors.add(infos["to_log"])
                rewards_stored = rewards.clone().unsqueeze(1)
                if 'time_outs' in infos:
                    rewards_stored += (
                        self.gamma
                        * policy_state_dict['values']
                        * infos['time_outs'].unsqueeze(1).to(self.device)
                    )

                self.storage.update_key('rewards', rewards_stored)
                self.storage.update_key('dones', dones.unsqueeze(1))
                self.storage.increment_step()
                self._process_env_step(rewards, dones, infos)

                if self.log_dir is not None:
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    self.rewbuffer.extend(
                        self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.lenbuffer.extend(
                        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    self.cur_reward_sum[new_ids] = 0
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time

            returns, advantages = self._compute_returns(
                last_obs_dict=obs_dict,
                policy_state_dict=dict(
                    values=self.storage.query_key('values'),
                    dones=self.storage.query_key('dones'),
                    rewards=self.storage.query_key('rewards'),
                ),
            )
            self.storage.batch_update_data('returns', returns)
            self.storage.batch_update_data('advantages', advantages)

        return obs_dict

    def _training_step(self):
        loss_dict = self._init_loss_dict_at_training_step()

        # PPO 업데이트
        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for policy_state_dict in generator:
            for k in policy_state_dict:
                policy_state_dict[k] = policy_state_dict[k].to(self.device)
            loss_dict = self._update_algo_step(policy_state_dict, loss_dict)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for key in loss_dict:
            if key != 'Discriminator':
                loss_dict[key] /= num_updates

        # Discriminator 업데이트 (PPO와 별도)
        disc_loss = self._update_discriminator()
        loss_dict['Discriminator'] = disc_loss

        self.storage.clear()
        return loss_dict

    def _update_discriminator(self) -> float:
        state_dim = self.state_dim

        # 이번 iteration 롤아웃에서 fake (s, s') 추출 (use_velocity면 pos+vel 이어붙임)
        all_dof_pos = self.storage.query_key('dof_pos')  # (T, N, dof_dim)
        if self.use_velocity:
            all_dof_vel = self.storage.query_key('dof_vel')  # (T, N, dof_dim)
            all_state = torch.cat([all_dof_pos, all_dof_vel], dim=-1)  # (T, N, state_dim)
        else:
            all_state = all_dof_pos
        T, N, D = all_state.shape

        fresh_fake_s = all_state[:-1].reshape(-1, D).to(self.device)
        fresh_fake_s_next = all_state[1:].reshape(-1, D).to(self.device)
        n_fresh = fresh_fake_s.shape[0]

        # replay buffer에 새 전이 기록 (다음 iteration 이후부터 샘플링에 사용됨)
        if self.use_replay_buffer:
            self._push_to_replay_buffer(torch.cat([fresh_fake_s, fresh_fake_s_next], dim=-1))

        # fake 배치 구성: buffer에 충분히 쌓였으면 절반은 이번 롤아웃, 절반은 buffer에서
        # 샘플링 (여러 과거 iteration의 정책 분포를 섞어서 discriminator가 "이번 배치"에만
        # 과적합하지 못하게 함)
        if self.use_replay_buffer and self.replay_buffer_filled >= n_fresh:
            n_replay = n_fresh // 2
            n_fresh_used = n_fresh - n_replay
            replay_pairs = self._sample_replay_buffer(n_replay)
            fake_s = torch.cat([fresh_fake_s[:n_fresh_used], replay_pairs[:, :state_dim]], dim=0)
            fake_s_next = torch.cat([fresh_fake_s_next[:n_fresh_used], replay_pairs[:, state_dim:]], dim=0)
        else:
            fake_s, fake_s_next = fresh_fake_s, fresh_fake_s_next
        n_fake = fake_s.shape[0]

        # motion dataset에서 real (s, s') 샘플링
        real_pairs = self._sample_motion_pairs(n_fake)
        real_s = real_pairs[:, :state_dim]
        real_s_next = real_pairs[:, state_dim:]

        disc_loss = self.discriminator.compute_disc_loss(
            real_s, real_s_next, fake_s, fake_s_next,
            grad_penalty_coef=self.disc_grad_penalty_coef,
            logit_reg_coef=self.disc_logit_reg_coef,
            weight_decay_coef=self.disc_weight_decay_coef,
            gp_type=self.disc_gp_type,
        )

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.disc_grad_norm)
        self.disc_optimizer.step()

        return disc_loss.item()

    def _init_loss_dict_at_training_step(self):
        loss_dict = super()._init_loss_dict_at_training_step()
        loss_dict['Discriminator'] = 0.0
        return loss_dict

    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        super()._logging_to_writer(log_dict, train_log_dict, env_log_dict)
        disc_loss = log_dict['loss_dict'].get('Discriminator', 0.0)
        self.writer.add_scalar('Loss/Discriminator', disc_loss, log_dict['it'])

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save(
            {
                'actor_model_state_dict': self.actor.state_dict(),
                'critic_model_state_dict': self.critic.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
            },
            path,
        )

    def load(self, ckpt_path):
        if ckpt_path is None:
            return
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(loaded['actor_model_state_dict'])
        self.critic.load_state_dict(loaded['critic_model_state_dict'])
        if 'discriminator_state_dict' in loaded:
            # [2026.08.23] logit_layer는 self.net의 마지막 Linear를 가리키는 별칭(같은
            # 파라미터 객체)이라 net.*.weight로 로드되면 자동으로 같이 채워짐. 하지만
            # 오늘 이 별칭을 추가하기 *전에* 저장된 옛날 체크포인트는 logit_layer.* 키가
            # 아예 없어서 strict=True면 로딩이 실패함 -> strict=False로 허용 (값 손실 없음).
            self.discriminator.load_state_dict(loaded['discriminator_state_dict'], strict=False)
        if self.load_optimizer:
            self.actor_optimizer.load_state_dict(loaded['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(loaded['critic_optimizer_state_dict'])
            if 'disc_optimizer_state_dict' in loaded:
                self.disc_optimizer.load_state_dict(loaded['disc_optimizer_state_dict'])
            self.actor_learning_rate = loaded['actor_optimizer_state_dict']['param_groups'][0]['lr']
            self.critic_learning_rate = loaded['critic_optimizer_state_dict']['param_groups'][0]['lr']
            self.set_learning_rate(self.actor_learning_rate, self.critic_learning_rate)
        self.current_learning_iteration = loaded['iter']
        return loaded.get('infos')
