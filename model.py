"""Custom SB3 policy for spatial EBSD action selection."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import CategoricalDistribution


class EBSDFeaturesExtractor(BaseFeaturesExtractor):
    """Lightweight fully-convolutional encoder shared by actor and critic.

    Input:  (batch, C, H, W)   C = N+3
    Output: flat feature vector of shape (batch, features_dim) used by critic.
    Also exposes forward_spatial() for the actor head.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # H/2, W/2
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # H, W
            nn.Conv2d(64, 32, kernel_size=1),  # (batch, 32, H, W)
        )
        self.pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (batch, 32, 1, 1)
            nn.Flatten(),             # (batch, 32)
            nn.Linear(32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns flat critic feature vector (batch, features_dim)."""
        return self.pool_proj(self.encoder(observations))

    def forward_spatial(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns spatial feature map (batch, 32, H, W) for the actor."""
        return self.encoder(observations)


class EBSDPolicy(ActorCriticPolicy):
    """Actor-critic policy with spatial logits and built-in action masking.

    Actor:  Conv2d(32→1) over spatial features → H*W logits, masked at sampled pixels.
    Critic: AdaptiveAvgPool + Linear(32→1) on spatial features.

    Action masking is read directly from channel 0 of the observation
    (binary mask: 1 = already sampled), so no sb3-contrib dependency is needed.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        **kwargs,
    ):
        kwargs.setdefault("features_extractor_class", EBSDFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
        # Disable the default MLP extractor and action/value heads built by _build()
        kwargs["net_arch"] = []
        kwargs["share_features_extractor"] = True
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Note: after super().__init__(), self._build() has already been called,
        # so actor_conv and value_net were set up there.

    def _build(self, lr_schedule: Schedule) -> None:
        """Override _build to install our custom heads and skip wasteful layers."""
        # Build the features extractor (CNN backbone)
        self.features_extractor = self.make_features_extractor()

        # Actor: 1×1 conv on 32-channel spatial map → H*W logits
        self.actor_conv = nn.Conv2d(32, 1, kernel_size=1)

        # Critic: simple linear on pooled features
        self.value_net = nn.Linear(self.features_dim, 1)

        # SB3 expects these attributes; set to Identity to avoid breakage
        self.mlp_extractor = _IdentityMlpExtractor(self.features_dim)

        # Initialise weights
        nn.init.orthogonal_(self.actor_conv.weight, gain=0.01)
        nn.init.constant_(self.actor_conv.bias, 0.0)
        nn.init.orthogonal_(self.value_net.weight, gain=1.0)
        nn.init.constant_(self.value_net.bias, 0.0)

        # Distribution object (Categorical)
        self.action_dist = CategoricalDistribution(int(self.action_space.n))

        # Optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    # ------------------------------------------------------------------
    # Core forward methods (override SB3 defaults)
    # ------------------------------------------------------------------

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.features_extractor.forward_spatial(obs)  # (B,32,H,W)
        logits = self._masked_logits(obs, spatial)              # (B, H*W)
        pooled = self.features_extractor.pool_proj(spatial)     # (B, features_dim)
        values = self.value_net(pooled)                         # (B, 1)

        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        spatial = self.features_extractor.forward_spatial(obs)
        logits = self._masked_logits(obs, spatial)
        pooled = self.features_extractor.pool_proj(spatial)
        values = self.value_net(pooled)

        dist = self.action_dist.proba_distribution(action_logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        spatial = self.features_extractor.forward_spatial(observation)
        logits = self._masked_logits(observation, spatial)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        return dist.get_actions(deterministic=deterministic)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _masked_logits(
        self, obs: torch.Tensor, spatial: torch.Tensor
    ) -> torch.Tensor:
        logits = self.actor_conv(spatial).flatten(1)             # (B, H*W)
        sampled = obs[:, 0, :, :].flatten(1).bool()             # (B, H*W)
        return logits.masked_fill(sampled, -1e9)


class _IdentityMlpExtractor(nn.Module):
    """Stub MlpExtractor that SB3 expects to exist; does nothing."""

    def __init__(self, features_dim: int):
        super().__init__()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features: torch.Tensor):
        return features, features

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


def extract_mask_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """Extract the binary sampling mask from channel 0 of the observation."""
    return obs[:, 0, :, :].flatten(1).bool()  # (B, H*W)
