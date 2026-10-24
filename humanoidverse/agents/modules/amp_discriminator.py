import torch
import torch.nn as nn


class AMPDiscriminator(nn.Module):
    """
    AMP (Adversarial Motion Priors) Discriminator.

    입력: (dof_pos_t, dof_pos_{t+1}) 연결 -> MLP -> raw score D (sigmoid 없음)
    학습: real=IK npz 전이(목표=+1), fake=policy 전이(목표=-1) — LSGAN 방식

    [2026.08.22 버그 수정] 기존엔 discriminator를 BCE(0/1 sigmoid)로 학습시키면서
    보상 공식은 D∈[-1,1]을 가정하는 논문 공식(1-0.25*(D-1)^2)에 sigmoid(D)∈[0,1]을
    그대로 넣고 있었음. D=0(가장 가짜로 판별)을 넣어도 1-0.25*(0-1)^2=0.75가 나와서
    "아무리 들켜도 보상이 0.75 밑으로 안 내려가는" 구조였고, 그 결과 여러 번의 학습에서
    discriminator가 항상 압도적으로 이겨버리는데도(disc_loss가 0에 수렴) 정책은 그걸
    만회할 유인이 거의 없었음. LSGAN 방식(target=+-1, MSE 학습, sigmoid 없이 raw score
    그대로 공식에 대입)으로 바꿔서 D=-1(완전히 가짜로 판별)일 때 보상이 진짜 0까지
    내려가도록 수정함.

    보상 공식 (AMP 논문, LSGAN 버전):
        r = max(0, 1 - 0.25 * (D(s, s') - 1)^2)
        D=+1(real로 판별) -> r=1.0, D=-1(완전히 fake로 판별) -> r=0.0

    [2026.08.23 discriminator 붕괴(loss->0.0005) 대응] 24시간 실험 결과 disc_loss가
    학습 후반 다시 0에 수렴 -> discriminator가 정책을 압도적으로 이겨서 AMP 보상이
    사실상 상수 0에 가까워짐. AMP 원 논문(Peng et al. 2021)과 IsaacGymEnvs 구현을
    대조해 다음 두 가지 차이를 발견하고 반영함:
      1) gradient penalty가 WGAN-GP식 (||grad||-1)^2 (그래디언트 norm이 1이 되도록
         유도) 였는데, 논문은 R1식 ||grad||^2 (real 샘플 근방에서 그래디언트를 아예
         0으로 유도, 결정 경계가 가팔라지는 것 자체를 억제)를 씀. 후자가 discriminator가
         갑자기 확신에 차는(=우리가 겪은 붕괴) 것을 더 직접적으로 막음.
      2) logit regularization과 weight decay가 아예 없었음 (논문/IsaacGymEnvs 둘 다
         사용, 기본값 각각 0.05/0.0001) -> discriminator 마지막 레이어 가중치와 전체
         가중치 크기에 L2 페널티를 줘서 출력이 쉽게 포화(saturate)되지 않도록 함.
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
        # 마지막 Linear(logit) 레이어를 별도 attribute로 새로 등록하면 self.parameters()의
        # 순서가 바뀌어서(logit_layer가 net보다 먼저 attribute로 등록됨) 옛날 체크포인트의
        # optimizer state_dict(파라미터를 이름이 아니라 "몇 번째냐"는 위치로 매칭)가 깨짐
        # (2026.08.23 확인: "tensor a (38) must match tensor b (128)" 에러). 그래서 별도
        # attribute를 만들지 않고 self.net[-1]로 참조만 함 -> 기존 체크포인트와 완전 호환.

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, s_next], dim=-1)
        return self.net(x)  # (N, 1) raw score (sigmoid 없음)

    def compute_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """롤아웃 중 AMP 보상 계산 (gradient 불필요). raw score를 그대로 공식에 대입."""
        d = self(s, s_next)
        r = torch.clamp(1.0 - 0.25 * (d - 1.0).pow(2), min=0.0).squeeze(-1)
        return r  # (N,)

    def compute_disc_loss(
        self,
        real_s: torch.Tensor,
        real_s_next: torch.Tensor,
        fake_s: torch.Tensor,
        fake_s_next: torch.Tensor,
        grad_penalty_coef: float = 10.0,
        logit_reg_coef: float = 0.05,
        weight_decay_coef: float = 0.0001,
        gp_type: str = "r1",
    ) -> torch.Tensor:
        """
        Discriminator 손실 = MSE(real, +1) + MSE(fake, -1)
                            + gradient penalty (real 샘플에만 적용, gp_type으로 방식 선택)
                            + logit regularization + weight decay

        [2026.08.25] R1(gp_type="r1") 방식이 discriminator 과신은 잘 막았지만, 실제
        23500(WGAN-GP 방식) 체크포인트와 걸음 리듬을 직접 비교해보니 300스텝 중 좌우 반전
        13회(23500) vs 8회(R1 버전, 훨씬 느리고 불규칙) -> R1이 "real 데이터 근처에서
        gradient를 아예 0으로 미는" 특성 때문에 미세한 리듬 차이를 구분하는 민감도까지
        같이 죽인 것으로 판단. gp_type="wgan"으로 WGAN-GP식 (||grad||-1)^2 (예전 방식,
        gradient norm을 1로 유도 -> real 근처에서도 완전히 평평해지진 않음)을 선택할 수
        있게 함. logit_reg/weight_decay는 gp_type과 무관하게 별도로 켜고 끌 수 있음
        (이 둘은 "가중치 크기 자체"를 억제하는 거라 R1의 "근처 gradient를 0으로" 만드는
        것과 메커니즘이 달라서, 붕괴 방지 효과를 유지하면서도 리듬 민감도를 살릴 수 있는지
        비교 실험 중).
        """
        real_input = torch.cat([real_s, real_s_next], dim=-1)
        fake_input = torch.cat([fake_s, fake_s_next], dim=-1)

        real_score = self.net(real_input)
        fake_score = self.net(fake_input)

        real_loss = (real_score - 1.0).pow(2).mean()
        fake_loss = (fake_score + 1.0).pow(2).mean()

        gp = self._gradient_penalty(real_input, gp_type=gp_type)

        loss = real_loss + fake_loss + grad_penalty_coef * gp

        if logit_reg_coef > 0:
            logit_reg = self.net[-1].weight.pow(2).sum()
            loss = loss + logit_reg_coef * logit_reg
        if weight_decay_coef > 0:
            weight_decay = sum(p.pow(2).sum() for p in self.net.parameters())
            loss = loss + weight_decay_coef * weight_decay

        return loss

    def _gradient_penalty(self, real_input: torch.Tensor, gp_type: str = "r1") -> torch.Tensor:
        """
        gp_type="r1": E[||grad_x D(x)||^2] (AMP 논문 방식, real 근처 gradient를 0으로 유도)
        gp_type="wgan": E[(||grad_x D(x)||-1)^2] (WGAN-GP 방식, gradient norm을 1로 유도,
                        예전/23500 체크포인트가 이 방식이었음)
        """
        x = real_input.detach().requires_grad_(True)
        score = self.net(x)
        grad = torch.autograd.grad(
            score.sum(), x, create_graph=True, retain_graph=True
        )[0]
        if gp_type == "wgan":
            return (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad.pow(2).sum(dim=1).mean()
