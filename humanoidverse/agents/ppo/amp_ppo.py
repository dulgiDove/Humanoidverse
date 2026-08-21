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
    """

    def setup(self):
        super().setup()
        self._setup_discriminator()
        self._setup_motion_dataset()

    def _setup_storage(self):
        super()._setup_storage()
        dof_dim = self.env.config.robot.actions_dim
        self.storage.register_key('dof_pos', shape=(dof_dim,), dtype=torch.float)

    def _setup_discriminator(self):
        dof_dim = self.env.config.robot.actions_dim
        hidden_dims = tuple(self.config.amp.disc_hidden_dims)
        self.discriminator = AMPDiscriminator(
            obs_dim=dof_dim * 2,
            hidden_dims=hidden_dims,
        ).to(self.device)
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.amp.disc_learning_rate,
        )
        self.amp_reward_weight = self.config.amp.amp_reward_weight
        logger.info(
            f"[AMP] Discriminator: input={dof_dim*2}, hidden={hidden_dims}, "
            f"disc_lr={self.config.amp.disc_learning_rate}, "
            f"amp_reward_weight={self.amp_reward_weight}"
        )

    def _setup_motion_dataset(self):
        motion_path = os.path.join(
            os.path.dirname(__file__), '..', '..',
            'data', 'motions', 'lafan1_walk_h1_ik.npz'
        )
        motion_path = os.path.normpath(motion_path)
        npz = np.load(motion_path)
        clips = torch.tensor(npz['clips'], dtype=torch.float32)   # (1443, 200, 19)
        lengths = torch.tensor(npz['lengths'], dtype=torch.long)  # (1443,)

        pairs_list = []
        for i in range(clips.shape[0]):
            L = lengths[i].item()
            if L < 2:
                continue
            s = clips[i, :L - 1, :]      # (L-1, 19)
            s_next = clips[i, 1:L, :]    # (L-1, 19)
            pairs_list.append(torch.cat([s, s_next], dim=-1))  # (L-1, 38)

        self.motion_pairs = torch.cat(pairs_list, dim=0).to(self.device)  # (N_total, 38)
        logger.info(f"[AMP] Motion dataset: {self.motion_pairs.shape[0]} transition pairs loaded")

    def _sample_motion_pairs(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.motion_pairs.shape[0], (n,), device=self.device)
        return self.motion_pairs[idx]  # (n, 38)

    def _rollout_step(self, obs_dict):
        dof_dim = self.env.config.robot.actions_dim

        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                # t 시점 dof_pos 저장 (env.step 호출 전)
                dof_pos_t = self.env.simulator.dof_pos[:, :dof_dim].clone()

                policy_state_dict = {}
                policy_state_dict = self._actor_rollout_step(obs_dict, policy_state_dict)
                values = self._critic_eval_step(obs_dict).detach()
                policy_state_dict["values"] = values

                for obs_key in obs_dict.keys():
                    self.storage.update_key(obs_key, obs_dict[obs_key])
                for key in policy_state_dict.keys():
                    self.storage.update_key(key, policy_state_dict[key])

                # dof_pos 저장 (discriminator 학습용)
                self.storage.update_key('dof_pos', dof_pos_t)

                actions = policy_state_dict["actions"]
                actor_state = {"actions": actions}
                obs_dict, rewards, dones, infos = self.env.step(actor_state)

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)

                # t+1 시점 dof_pos로 AMP 보상 계산
                dof_pos_t1 = self.env.simulator.dof_pos[:, :dof_dim]
                amp_reward = self.discriminator.compute_reward(dof_pos_t, dof_pos_t1)
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
        dof_dim = self.env.config.robot.actions_dim

        # 롤아웃에서 fake (s, s') 추출
        all_dof_pos = self.storage.query_key('dof_pos')  # (T, N, dof_dim)
        T, N, D = all_dof_pos.shape

        fake_s = all_dof_pos[:-1].reshape(-1, D).to(self.device)
        fake_s_next = all_dof_pos[1:].reshape(-1, D).to(self.device)
        n_fake = fake_s.shape[0]

        # motion dataset에서 real (s, s') 샘플링
        real_pairs = self._sample_motion_pairs(n_fake)
        real_s = real_pairs[:, :dof_dim]
        real_s_next = real_pairs[:, dof_dim:]

        disc_loss = self.discriminator.compute_disc_loss(
            real_s, real_s_next, fake_s, fake_s_next,
            grad_penalty_coef=self.config.amp.grad_penalty_coef,
        )

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
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
            self.discriminator.load_state_dict(loaded['discriminator_state_dict'])
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
