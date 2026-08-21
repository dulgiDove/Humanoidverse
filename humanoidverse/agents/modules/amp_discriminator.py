import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPDiscriminator(nn.Module):
    """
    AMP (Adversarial Motion Priors) Discriminator.

    입력: (dof_pos_t, dof_pos_{t+1}) 연결 -> MLP -> logit
    학습: real=IK npz 전이, fake=policy 전이

    보상 공식 (AMP 논문):
        r = max(0, 1 - 0.25 * (1 - D(s, s'))^2)
        D=1(real로 판별) -> r=1.0, D=0(fake로 판별) -> r=0.75
    """

    def __init__(self, obs_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, s_next], dim=-1)
        return self.net(x)  # (N, 1) logit

    def compute_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """롤아웃 중 AMP 보상 계산 (gradient 불필요)."""
        logit = self(s, s_next)
        d = torch.sigmoid(logit)
        r = torch.clamp(1.0 - 0.25 * (1.0 - d).pow(2), min=0.0).squeeze(-1)
        return r  # (N,)

    def compute_disc_loss(
        self,
        real_s: torch.Tensor,
        real_s_next: torch.Tensor,
        fake_s: torch.Tensor,
        fake_s_next: torch.Tensor,
        grad_penalty_coef: float = 5.0,
    ) -> torch.Tensor:
        """
        Discriminator 손실 = BCE(real, 1) + BCE(fake, 0) + gradient penalty.
        Gradient penalty는 real sample에만 적용.
        """
        real_input = torch.cat([real_s, real_s_next], dim=-1)
        fake_input = torch.cat([fake_s, fake_s_next], dim=-1)

        real_logit = self.net(real_input)
        fake_logit = self.net(fake_input)

        real_loss = F.binary_cross_entropy_with_logits(
            real_logit, torch.ones_like(real_logit)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_logit, torch.zeros_like(fake_logit)
        )

        gp = self._gradient_penalty(real_input)

        return real_loss + fake_loss + grad_penalty_coef * gp

    def _gradient_penalty(self, real_input: torch.Tensor) -> torch.Tensor:
        x = real_input.detach().requires_grad_(True)
        logit = self.net(x)
        grad = torch.autograd.grad(
            logit.sum(), x, create_graph=True, retain_graph=True
        )[0]
        return (grad.norm(2, dim=1) - 1).pow(2).mean()
