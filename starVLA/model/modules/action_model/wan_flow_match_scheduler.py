from __future__ import annotations

"""Flow-match training schedule (noise + target) aligned with diffusers Wan schedulers.

Used by WM4A joint Wan frameworks alongside ``WanVideoActionMoE``; colocated with
``action_model`` because it parameterises **action / video denoising** training steps.
"""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class FlowMatchScheduler:
    """官方 FlowMatch scheduler 的训练包装层。

    - schedule / add-noise 语义对齐 diffusers `FlowMatchEulerDiscreteScheduler`
    - timestep empirical weighting 保留 DiffSynth 的经验公式
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        sampling_distribution: str = "uniform_index",
        train_sigma_min: float | None = None,
        train_sigma_max: float | None = None,
        hybrid_uniform_ratio: float = 0.3,
        hybrid_uniform_lower: float | None = None,
        hybrid_uniform_upper: float | None = None,
        lognormal_mean: float = 1.39,
        lognormal_std: float = 1.2,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sampling_distribution = str(sampling_distribution or "uniform_index").strip().lower()
        self.train_sigma_min = float(self.sigma_min if train_sigma_min is None else train_sigma_min)
        self.train_sigma_max = float(self.sigma_max if train_sigma_max is None else train_sigma_max)
        self.hybrid_uniform_ratio = float(hybrid_uniform_ratio)
        self.hybrid_uniform_lower = float(self.train_sigma_min if hybrid_uniform_lower is None else hybrid_uniform_lower)
        self.hybrid_uniform_upper = float(self.train_sigma_max if hybrid_uniform_upper is None else hybrid_uniform_upper)
        self.lognormal_mean = float(lognormal_mean)
        self.lognormal_std = float(lognormal_std)
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=self.shift,
        )
        self._build_schedule()

    def _use_absolute_sigmas(self) -> bool:
        return bool(self.sigma_max > 1.0 + 1e-6 or self.sigma_min < -1e-6)

    def _build_sigmas(self, num_steps: int) -> torch.Tensor:
        if self._use_absolute_sigmas():
            return torch.linspace(self.sigma_max, self.sigma_min, int(num_steps), dtype=torch.float32)
        if abs(self.sigma_min) <= 1e-12 and abs(self.sigma_max - 1.0) <= 1e-12:
            if int(num_steps) == int(self.num_train_timesteps):
                return self.scheduler.sigmas.detach().clone().to(dtype=torch.float32, device="cpu")
            infer_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
            infer_scheduler.set_timesteps(int(num_steps), device="cpu")
            return infer_scheduler.sigmas[:-1].detach().clone().to(dtype=torch.float32, device="cpu")
        base_sigmas = torch.linspace(self.sigma_max, self.sigma_min, int(num_steps) + 1, dtype=torch.float32)[:-1]
        return self.shift * base_sigmas / (1 + (self.shift - 1) * base_sigmas)

    def _build_schedule(self) -> None:
        sigmas = self._build_sigmas(self.num_train_timesteps).to(device="cpu")
        self.sigmas = sigmas.to(dtype=torch.float32, device="cpu")
        self.timesteps = (self.sigmas * self.num_train_timesteps).to(dtype=torch.float32, device="cpu")
        self.scheduler.sigmas = self.sigmas.clone()
        self.scheduler.timesteps = self.timesteps.clone()
        self.scheduler.sigma_min = float(self.sigmas[-1].item())
        self.scheduler.sigma_max = float(self.sigmas[0].item())

        x = self.timesteps
        y = torch.exp(-2 * ((x - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2)
        y = y - y.min()
        self.training_weights = y * (self.num_train_timesteps / y.sum().clamp_min(1e-12))

    @staticmethod
    def _ensure_1d(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 0:
            value = value.unsqueeze(0)
        return value

    def _is_sigma_value(self, value: torch.Tensor) -> bool:
        value = self._ensure_1d(value.detach()).to(dtype=torch.float32)
        return bool(value.max().item() <= 1.0 + 1e-6)

    def _nearest_sigma_index(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = self._ensure_1d(sigma).to(device=self.sigmas.device, dtype=self.sigmas.dtype)
        return torch.argmin((self.sigmas[None, :] - sigma[:, None]).abs(), dim=1)

    def _nearest_timestep_index(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self._ensure_1d(timestep).to(device=self.timesteps.device, dtype=self.timesteps.dtype)
        return torch.argmin((self.timesteps[None, :] - timestep[:, None]).abs(), dim=1)

    def _value_to_indices(self, value: torch.Tensor) -> torch.Tensor:
        if self._is_sigma_value(value):
            return self._nearest_sigma_index(value)
        return self._nearest_timestep_index(value)

    def sigma_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        if self._use_absolute_sigmas():
            timestep = self._ensure_1d(timestep).to(dtype=torch.float32)
            return timestep.to(device=timestep.device, dtype=torch.float32) / float(self.num_train_timesteps)
        indices = self._nearest_timestep_index(timestep)
        return self.sigmas.to(device=timestep.device, dtype=torch.float32)[indices]

    def indices_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = self._ensure_1d(sigma).to(dtype=torch.float32)
        return self._nearest_sigma_index(sigma.detach().cpu())

    def indices_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self._ensure_1d(timestep).to(dtype=torch.float32)
        return self._nearest_timestep_index(timestep.detach().cpu())

    def sigmas_from_indices(self, indices: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        values = self.sigmas.index_select(0, indices.to(device=self.sigmas.device, dtype=torch.long))
        if device is not None:
            values = values.to(device=device)
        return values.to(dtype=torch.float32)

    def timesteps_from_indices(self, indices: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        values = self.timesteps.index_select(0, indices.to(device=self.timesteps.device, dtype=torch.long))
        if device is not None:
            values = values.to(device=device)
        return values.to(dtype=torch.float32)

    def weights_from_indices(self, indices: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        values = self.training_weights.index_select(0, indices.to(device=self.training_weights.device, dtype=torch.long))
        if device is not None:
            values = values.to(device=device)
        return values.to(dtype=torch.float32)

    def timestep_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = self._ensure_1d(sigma).to(dtype=torch.float32)
        return sigma * float(self.num_train_timesteps)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        min_boundary: float = 0.0,
        max_boundary: float = 1.0,
        return_timesteps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sampling_distribution == "hybrid_log_uniform":
            sigma = self._sample_hybrid_sigmas(int(batch_size), device=device)
            values = self.timestep_from_sigma(sigma).to(device=device, dtype=torch.float32) if return_timesteps else sigma
            weights = torch.ones((int(batch_size),), device=device, dtype=torch.float32)
            return values, weights

        total = len(self.timesteps)
        min_idx = max(0, min(int(float(min_boundary) * total), total - 1))
        max_idx = max(min_idx + 1, min(int(float(max_boundary) * total), total))
        timestep_ids = torch.randint(min_idx, max_idx, (int(batch_size),), device=device)
        if return_timesteps:
            values = self.timesteps.to(device=device, dtype=torch.float32)[timestep_ids]
        else:
            values = self.sigmas.to(device=device, dtype=torch.float32)[timestep_ids]
        weights = self.training_weights.to(device=device, dtype=torch.float32)[timestep_ids]
        return values, weights

    def _sample_hybrid_sigmas(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        sigma_min = min(self.train_sigma_min, self.train_sigma_max)
        sigma_max = max(self.train_sigma_min, self.train_sigma_max)
        uniform_lower = min(max(float(self.hybrid_uniform_lower), sigma_min), sigma_max)
        uniform_upper = max(min(float(self.hybrid_uniform_upper), sigma_max), uniform_lower)
        mix = float(min(max(self.hybrid_uniform_ratio, 0.0), 1.0))
        choose_uniform = torch.rand((int(batch_size),), device=device) < mix
        sigma = torch.empty((int(batch_size),), device=device, dtype=torch.float32)
        if bool((~choose_uniform).any()):
            lognormal = torch.distributions.LogNormal(self.lognormal_mean, self.lognormal_std)
            sigma[~choose_uniform] = lognormal.sample((int((~choose_uniform).sum().item()),)).to(device=device, dtype=torch.float32)
        if bool(choose_uniform.any()):
            sigma[choose_uniform] = torch.empty((int(choose_uniform.sum().item()),), device=device, dtype=torch.float32).uniform_(
                float(uniform_lower),
                float(uniform_upper),
            )
        return sigma.clamp_(min=float(sigma_min), max=float(sigma_max))

    def build_inference_scheduler(self, *, device: torch.device, num_steps: int) -> FlowMatchEulerDiscreteScheduler:
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        sigmas = self._build_sigmas(int(num_steps)).to(device=device, dtype=torch.float32)
        timesteps = (sigmas * self.num_train_timesteps).to(device=device, dtype=torch.float32)
        scheduler.num_inference_steps = int(num_steps)
        scheduler.timesteps = timesteps
        scheduler.sigmas = torch.cat([sigmas, torch.zeros(1, device=device, dtype=torch.float32)], dim=0)
        scheduler.sigma_min = float(sigmas[-1].item())
        scheduler.sigma_max = float(sigmas[0].item())
        scheduler._step_index = None
        scheduler._begin_index = None
        return scheduler

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self._use_absolute_sigmas():
            sigma = self.sigma_from_timestep(timestep).to(device=original.device, dtype=original.dtype)
            while sigma.ndim < original.ndim:
                sigma = sigma.unsqueeze(-1)
            return sigma * noise + (1.0 - sigma) * original
        indices = self._value_to_indices(timestep)
        schedule_timestep = self.timesteps.to(device=original.device, dtype=torch.float32)[indices]
        return self.scheduler.scale_noise(
            sample=original,
            timestep=schedule_timestep,
            noise=noise,
        )

    def add_noise_with_sigma_scale(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        *,
        sigma_scale: float = 1.0,
    ) -> torch.Tensor:
        if abs(float(sigma_scale) - 1.0) <= 1e-8:
            return self.add_noise(original, noise, timestep)

        sigma = self.sigma_from_timestep(timestep).to(device=original.device, dtype=original.dtype)
        sigma = sigma * float(sigma_scale)
        while sigma.ndim < original.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * original

    @staticmethod
    def training_target(original: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor | None = None) -> torch.Tensor:
        del timestep
        return noise - original

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        indices = self._value_to_indices(timestep)
        return self.training_weights.to(device=timestep.device, dtype=torch.float32)[indices]
