import os
import pickle
import lzma

import gym 

from gym import spaces

import numpy as np

import carla

import random

import pygame

import cv2 

import time

import math

import torch

import serial

from rich.console import Console

from stable_baselines3 import SAC

# Mixed Precision Training imports (updated for PyTorch 2.x)
from torch.amp import GradScaler, autocast

console = Console()





class CustomSAC(SAC):
    """
    SAC implementation following the original paper exactly:
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor"
    (Haarnoja et al., 2018)
    
    Key principles from the paper (Section 4.2 & Appendix D):
    1. Target entropy H̄ = -dim(A) is FIXED (not decayed over time)
    2. α (entropy coefficient) is LEARNED to satisfy: E[-log π(a|s)] ≥ H̄
    3. α automatically adjusts as policy becomes more/less confident
    4. NO BOUNDS ON α - the paper does not clamp alpha!
    
    From paper Table 1 (Appendix D):
    - Learning rate: 3×10⁻⁴
    - Target smoothing (τ): 0.005
    - Network: 2 hidden layers, 256 units each
    - No alpha clamping mentioned
    
    Additional feature (not in paper):
    - Mixed Precision Training for efficiency
    """

    def __init__(self, *args, use_mixed_precision=True, **kwargs):

        kwargs.pop('logger', None)
        kwargs.pop('ent_coef', None)  # Remove any ent_coef passed
        
        # Use automatic entropy tuning (paper's approach)
        # "auto" means: learn α to match target_entropy = -dim(A)
        super().__init__(*args, ent_coef="auto", **kwargs)
        
        # Target entropy is FIXED at -dim(A) as per the paper
        # For 3D action space (steering, throttle, brake): H̄ = -3
        # This is set automatically by SB3's SAC when ent_coef="auto"
        if isinstance(self.target_entropy, (int, float)):
            self.fixed_target_entropy = float(self.target_entropy)
        else:
            self.fixed_target_entropy = -3.0  # Default for 3D action space
        
        # NO ALPHA BOUNDS - Paper does not clamp α (Table 1, Appendix D)
        # α is learned freely via gradient descent on the entropy loss
        # This allows the algorithm to find the optimal exploration-exploitation balance
        
        # Mixed Precision Training (FP16) - reduces VRAM usage
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            console.log("[green]Mixed Precision Training (FP16) enabled[/green]")
        else:
            self.scaler = None
            console.log("[yellow]Mixed Precision disabled[/yellow]")
        
        console.log(f"[cyan]SAC (Paper Implementation):[/cyan]")
        console.log(f"[cyan]  - Target entropy H̄ = {self.fixed_target_entropy:.3f} (FIXED, = -dim(A))[/cyan]")
        console.log(f"[cyan]  - α is learned automatically to match H̄[/cyan]")
        console.log(f"[cyan]  - α bounds: None (free, as per paper)[/cyan]")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Train the SAC model with Mixed Precision (FP16) support.
        This reduces VRAM usage by ~50% and can speed up training on modern GPUs.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        # Update learning rate and entropy coefficient
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Get current observations
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            dones = replay_data.dones
            rewards = replay_data.rewards
            
            # Get current entropy coefficient (use learned value from log_ent_coef)
            if self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            if self.use_mixed_precision:
                # CRITICAL: Actor action sampling MUST be in FP32 to avoid NaN in Normal distribution
                # The Normal distribution's validate_args check fails with FP16 due to precision issues
                # Only use autocast for critic forward passes (Q-value computation)
                
                with torch.no_grad():
                    # Select action according to policy (FP32 - no autocast!)
                    next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                    
                    # Mixed precision for Q-value computation only
                    with autocast(device_type='cuda'):
                        # Compute the next Q values
                        next_q_values = torch.cat(self.critic_target(next_obs, next_actions), dim=1)
                        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                        # Entropy regularization (cast to FP32 for stability)
                        next_q_values = next_q_values.float() - ent_coef * next_log_prob.reshape(-1, 1)
                        # td error + entropy term
                        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

                # Mixed Precision for critic loss computation
                with autocast(device_type='cuda'):
                    # Get current Q estimates for each critic
                    current_q_values = self.critic(obs, actions)
                    # Compute critic loss
                    critic_loss = 0.5 * sum(
                        torch.nn.functional.mse_loss(current_q, target_q_values.float()) 
                        for current_q in current_q_values
                    )

                # Optimize the critic with gradient scaling
                self.critic.optimizer.zero_grad()
                self.scaler.scale(critic_loss).backward()
                # Gradient clipping to prevent exploding gradients
                self.scaler.unscale_(self.critic.optimizer)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.scaler.step(self.critic.optimizer)

                # Actor loss computation (FP32 for action sampling, autocast for Q-values)
                actions_pi, log_prob = self.actor.action_log_prob(obs)  # FP32 - no autocast!
                log_prob = log_prob.reshape(-1, 1)  # Match SB3's shape for proper broadcasting
                
                with autocast(device_type='cuda'):
                    q_values_pi = torch.cat(self.critic(obs, actions_pi), dim=1)
                    min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                    # Cast to FP32 for actor loss to maintain precision
                    actor_loss = (ent_coef * log_prob - min_qf_pi.float()).mean()

                # Optimize the actor with gradient scaling
                self.actor.optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                # Gradient clipping to prevent exploding gradients
                self.scaler.unscale_(self.actor.optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.scaler.step(self.actor.optimizer)
                
                # Update entropy coefficient (if using automatic tuning)
                if self.ent_coef_optimizer is not None:
                    # Entropy loss in FP32 for stability
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    self.scaler.scale(ent_coef_loss).backward()
                    self.scaler.step(self.ent_coef_optimizer)

                # Update the scaler
                self.scaler.update()
            else:
                # Standard training without mixed precision (fallback for CPU)
                with torch.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                    next_q_values = torch.cat(self.critic_target(next_obs, next_actions), dim=1)
                    next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

                current_q_values = self.critic(obs, actions)
                critic_loss = 0.5 * sum(
                    torch.nn.functional.mse_loss(current_q, target_q_values) 
                    for current_q in current_q_values
                )

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic.optimizer.step()

                actions_pi, log_prob = self.actor.action_log_prob(obs)
                log_prob = log_prob.reshape(-1, 1)  # Match SB3's shape for proper broadcasting
                q_values_pi = torch.cat(self.critic(obs, actions_pi), dim=1)
                min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor.optimizer.step()
                
                # Update entropy coefficient (if using automatic tuning)
                if self.ent_coef_optimizer is not None:
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

            # Update target networks
            self._update_target_networks()

            self._n_updates += 1
            if self.logger is not None:
                self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
                self.logger.record("train/actor_loss", actor_loss.item())
                self.logger.record("train/critic_loss", critic_loss.item())
                if self.ent_coef_optimizer is not None:
                    self.logger.record("train/ent_coef_loss", ent_coef_loss.item())

    def _update_target_networks(self):
        """Soft update of the target networks"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, progress_bar=False):
        """
        Start learning. Follows original SAC paper - no special setup needed.
        """
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)



    def _update_learning_rate(self, optimizers):
        """Update learning rate. No alpha clamping - follows paper exactly."""
        super()._update_learning_rate(optimizers)
        
        # NO ALPHA CLAMPING - Paper does not clamp α (Table 1, Appendix D)
        # α is learned freely via gradient descent on the entropy loss
        
        # Get current α for logging
        if hasattr(self, 'log_ent_coef') and self.log_ent_coef is not None:
            current_alpha = torch.exp(self.log_ent_coef).item()
        else:
            current_alpha = self.ent_coef_tensor.item()
        
        # Logging
        if self.logger is not None:
            self.logger.record("train/entropy_coefficient", current_alpha)
            self.logger.record("train/target_entropy", self.target_entropy)
        
        # Debug output every 1000 steps
        if self.num_timesteps % 1000 == 0:
            print(f"[SAC] step: {self.num_timesteps}, α: {current_alpha:.4f}, "
                  f"target_H: {self.fixed_target_entropy:.3f} (fixed, no clamp)")

    def _excluded_save_params(self):
        """
        Return parameters that should be excluded from being saved.
        
        GradScaler contains _thread.RLock which cannot be pickled.
        The env contains CARLA client, pygame display, etc. which also can't be pickled.
        We exclude them and recreate/reconnect on load.
        """
        excluded = set(super()._excluded_save_params())
        # Add our custom non-picklable attributes
        # GradScaler has thread locks that can't be pickled
        excluded.add("scaler")
        excluded.add("use_mixed_precision")  # Simple bool, recreated on load
        # Environment contains unpicklable CARLA/pygame objects
        excluded.add("env")
        excluded.add("_vec_normalize_env")
        excluded.add("_last_obs")  # May contain env references
        excluded.add("_last_episode_starts")
        return excluded

    def save(self, path, exclude=None, include=None):
        """
        Save the model to a zip file, handling unpicklable CARLA environment.
        
        Uses a WHITELIST approach - only save explicitly safe data.
        This guarantees no pickle errors from env/pygame/carla objects.
        """
        import pathlib
        from stable_baselines3.common.save_util import recursive_getattr, save_to_zip_file
        
        # Build path
        path = pathlib.Path(path)
        if path.suffix == "":
            path = path.with_suffix(".zip")
        
        # Convert train_freq to simple tuple (it's a TrainFreq namedtuple)
        if hasattr(self.train_freq, '_asdict'):
            train_freq_data = (self.train_freq.frequency, self.train_freq.unit.value)
        else:
            train_freq_data = self.train_freq
        
        # WHITELIST: Only save these specific, known-safe parameters
        # These are the essential hyperparameters needed to reconstruct the model
        # IMPORTANT: Only primitive types (int, float, str, tuple, list, dict of primitives)
        safe_data = {
            # Core algorithm params (all primitives)
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "learning_rate": float(self.learning_rate) if isinstance(self.learning_rate, (int, float)) else 3e-4,
            "buffer_size": int(self.buffer_size),
            "batch_size": int(self.batch_size),
            "learning_starts": int(self.learning_starts),
            "train_freq": train_freq_data,
            "gradient_steps": int(self.gradient_steps),
            "ent_coef": "auto",  # We use auto entropy
            "target_update_interval": int(self.target_update_interval),
            "target_entropy": float(self.target_entropy) if isinstance(self.target_entropy, (int, float)) else "auto",
            
            # Training state (all primitives)
            "n_envs": int(self.n_envs),
            "num_timesteps": int(self.num_timesteps),
            "_total_timesteps": int(self._total_timesteps),
            "_num_timesteps_at_start": int(getattr(self, '_num_timesteps_at_start', 0)),
            "_n_updates": int(self._n_updates),
            "_episode_num": int(self._episode_num),
            
            # Device (as string)
            "device": str(self.device),
            
            # SB3 required params (must be present for load() to work)
            "verbose": int(getattr(self, 'verbose', 1)),
            "policy_class": self.policy_class,
            "policy_kwargs": self.policy_kwargs if self.policy_kwargs is not None else {},
            
            # Custom SAC params (primitives)
            "fixed_target_entropy": float(getattr(self, 'fixed_target_entropy', -3.0)),
            "use_mixed_precision": bool(getattr(self, 'use_mixed_precision', True)),
            
            # Environment spaces (REQUIRED for load() to verify environment compatibility)
            # These are gym.spaces objects which are picklable
            "observation_space": self.observation_space,
            "action_space": self.action_space,
        }
        
        # Get neural network params to save via PyTorch
        state_dicts, pytorch_variables = self._get_torch_save_params()
        params_to_save = {}
        for name in state_dicts:
            attr = recursive_getattr(self, name)
            params_to_save[name] = attr.state_dict()
        for name in pytorch_variables:
            attr = recursive_getattr(self, name)
            params_to_save[name] = attr
        
        # Save to zip file - only safe_data, no risky objects
        save_to_zip_file(path, data=safe_data, params=params_to_save, pytorch_variables=None)

    def _get_torch_save_params(self):
        """a
        Get the name of torch variables that will be saved with PyTorch's ``th.save``.
        
        Override to ensure scaler is not included in saved state.
        """
        state_dicts, saved_pytorch_variables = super()._get_torch_save_params()
        # scaler is handled separately (recreated on load)
        return state_dicts, saved_pytorch_variables

    

    def save_replay_buffer(self, path, use_compression=True):
        """
        Save the replay buffer to a file.
        
        Args:
            path: Path to save the replay buffer (will add .pkl.xz or .pkl extension)
            use_compression: If True, use LZMA compression (~60-70% size reduction)
        
        The replay buffer contains all transitions (obs, action, reward, next_obs, done)
        which are essential for continuing training without losing experience.
        
        File size estimate for 50k buffer with our observation space:
        - Uncompressed: ~3-4 GB
        - Compressed (LZMA): ~1-1.5 GB
        """
        import pathlib
        
        path = pathlib.Path(path)
        
        # Calculate actual buffer size first
        actual_size = self.replay_buffer.pos if not self.replay_buffer.full else self.replay_buffer.buffer_size
        
        # Safety check: don't save empty buffer
        if actual_size == 0:
            console.log(f"[yellow]Replay buffer is empty, skipping save[/yellow]")
            return None
        
        # Get buffer data to save
        # We save the internal arrays directly for efficiency
        # Use .copy() to avoid saving views that might cause issues
        buffer_data = {
            'pos': self.replay_buffer.pos,
            'full': self.replay_buffer.full,
            'buffer_size': self.replay_buffer.buffer_size,
            'observations': {},
            'next_observations': {},
            'actions': self.replay_buffer.actions[:actual_size].copy(),
            'rewards': self.replay_buffer.rewards[:actual_size].copy(),
            'dones': self.replay_buffer.dones[:actual_size].copy(),
        }
        
        # Save observations (Dict observation space)
        for key in self.replay_buffer.observations.keys():
            buffer_data['observations'][key] = self.replay_buffer.observations[key][:actual_size].copy()
            buffer_data['next_observations'][key] = self.replay_buffer.next_observations[key][:actual_size].copy()
        
        # Calculate size for logging
        total_transitions = actual_size
        
        if use_compression:
            # Use string concatenation for .pkl.xz since pathlib doesn't handle double extensions well
            save_path = pathlib.Path(str(path) + '.pkl.xz')
            console.log(f"[cyan]Saving replay buffer ({total_transitions:,} transitions) with LZMA compression...[/cyan]")
            with lzma.open(save_path, 'wb', preset=3) as f:  # preset=3 is good balance of speed/compression
                pickle.dump(buffer_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            save_path = path.with_suffix('.pkl')
            console.log(f"[cyan]Saving replay buffer ({total_transitions:,} transitions) uncompressed...[/cyan]")
            with open(save_path, 'wb') as f:
                pickle.dump(buffer_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Get file size
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        console.log(f"[green]Replay buffer saved: {save_path} ({file_size_mb:.1f} MB)[/green]")
        
        return save_path
    
    def load_replay_buffer(self, path):
        """
        Load replay buffer from a file.
        
        Args:
            path: Path to the replay buffer file (.pkl or .pkl.xz)
        
        Returns:
            True if loaded successfully, False otherwise
        """
        import pathlib
        
        path = pathlib.Path(path)
        
        # Try both compressed and uncompressed paths
        # NOTE: Use string concatenation for double extensions (.pkl.xz)
        # because pathlib's with_suffix() only handles single extensions
        if path.suffix == '.xz':
            compressed_path = path
            # Remove .xz to get .pkl path (if it was .pkl.xz -> .pkl)
            uncompressed_path = pathlib.Path(str(path)[:-3])  # Remove '.xz'
        elif path.suffix == '.pkl':
            # Path is .pkl, compressed would be .pkl.xz
            compressed_path = pathlib.Path(str(path) + '.xz')
            uncompressed_path = path
        else:
            # Assume it's a base path (no extension), try both
            # Use string concatenation to handle double extension correctly
            compressed_path = pathlib.Path(str(path) + '.pkl.xz')
            uncompressed_path = pathlib.Path(str(path) + '.pkl')
        
        # Try compressed first (preferred), then uncompressed
        load_path = None
        use_compression = False
        
        if compressed_path.exists():
            load_path = compressed_path
            use_compression = True
        elif uncompressed_path.exists():
            load_path = uncompressed_path
            use_compression = False
        else:
            console.log(f"[yellow]No replay buffer found at {path}[/yellow]")
            return False
        
        try:
            console.log(f"[cyan]Loading replay buffer from {load_path}...[/cyan]")
            
            if use_compression:
                with lzma.open(load_path, 'rb') as f:
                    buffer_data = pickle.load(f)
            else:
                with open(load_path, 'rb') as f:
                    buffer_data = pickle.load(f)
            
            # Restore buffer state
            # actual_size = number of valid transitions in the saved buffer
            saved_pos = buffer_data['pos']
            saved_full = buffer_data['full']
            saved_buffer_size = buffer_data['buffer_size']
            
            # Calculate how many transitions were in the saved buffer
            if saved_full:
                actual_size = saved_buffer_size  # Buffer was full, all slots used
            else:
                actual_size = saved_pos  # Buffer not full, pos = number of transitions
            
            # Check if buffer sizes match
            if saved_buffer_size != self.replay_buffer.buffer_size:
                console.log(f"[yellow]Buffer size mismatch: saved={saved_buffer_size}, current={self.replay_buffer.buffer_size}[/yellow]")
                # Can only load up to the smaller of the two
                load_size = min(actual_size, self.replay_buffer.buffer_size)
                console.log(f"[yellow]Loading {load_size:,} of {actual_size:,} transitions[/yellow]")
                
                # If we're loading fewer than we have, just load linearly
                actual_size = load_size
                self.replay_buffer.pos = actual_size % self.replay_buffer.buffer_size
                self.replay_buffer.full = (actual_size >= self.replay_buffer.buffer_size)
            else:
                # Buffer sizes match - restore exact state (pos and full flag)
                self.replay_buffer.pos = saved_pos
                self.replay_buffer.full = saved_full
            
            # Restore arrays (only up to actual_size)
            self.replay_buffer.actions[:actual_size] = buffer_data['actions'][:actual_size]
            self.replay_buffer.rewards[:actual_size] = buffer_data['rewards'][:actual_size]
            self.replay_buffer.dones[:actual_size] = buffer_data['dones'][:actual_size]
            
            # Restore observations (Dict observation space)
            for key in buffer_data['observations'].keys():
                if key in self.replay_buffer.observations:
                    self.replay_buffer.observations[key][:actual_size] = buffer_data['observations'][key][:actual_size]
                    self.replay_buffer.next_observations[key][:actual_size] = buffer_data['next_observations'][key][:actual_size]
            
            file_size_mb = load_path.stat().st_size / (1024 * 1024)
            console.log(f"[green]Replay buffer loaded: {actual_size:,} transitions ({file_size_mb:.1f} MB)[/green]")
            return True
            
        except Exception as e:
            console.log(f"[red]Error loading replay buffer: {e}[/red]")
            import traceback
            console.log(f"[red]{traceback.format_exc()}[/red]")
            return False

    @classmethod
    def load(cls, path, env=None, device="auto", custom_objects=None, force_reset=True, 
             use_mixed_precision=True, load_replay_buffer=True, **kwargs):
        """
        Load the model from a zip-file.
        
        Follows original SAC paper exactly:
        - Fixed target entropy H̄ = -dim(A) = -3.0
        - NO alpha clamping (α learned freely via gradient descent)
        
        Args:
            path: Path to the model file
            env: Environment to use (REQUIRED - provides observation/action spaces)
            device: Device to load the model on
            custom_objects: Custom objects to load
            force_reset: Whether to force reset
            use_mixed_precision: Whether to use mixed precision training
            load_replay_buffer: Whether to auto-load replay buffer if found (default: True)
        """
        import pathlib
        import tempfile
        from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
        
        # First, try to load the data to check if observation/action spaces exist
        # If they don't exist (old checkpoint), we need to provide them from env
        data, params, pytorch_variables = load_from_zip_file(path, device=device)
        
        # Check if spaces are missing from saved data
        spaces_were_missing = False
        if "observation_space" not in data or "action_space" not in data:
            spaces_were_missing = True
            console.log("[yellow]Checkpoint missing observation/action spaces - extracting from environment...[/yellow]")
            if env is None:
                raise ValueError(
                    "Environment (env) must be provided when loading checkpoints that don't contain "
                    "observation_space and action_space. Please pass env=your_env to load()."
                )
            
            # IMPORTANT: SB3 applies VecTransposeImage wrapper which changes observation space
            # from (H, W, C) to (C, H, W). We need to apply the same transformation.
            # The base env has (H, W, C), but SB3 will compare against (C, H, W) after wrapping.
            from stable_baselines3.common.vec_env import VecTransposeImage
            
            # Get spaces from the environment
            # First try the wrapped env's spaces directly (if already wrapped)
            if hasattr(env, 'observation_space'):
                base_obs_space = env.observation_space
            elif hasattr(env, 'envs') and len(env.envs) > 0:
                base_obs_space = env.envs[0].observation_space
            else:
                raise ValueError("Could not extract observation_space from environment")
            
            # Check if VecTransposeImage will be applied - SB3 does this for image observations
            # We need to match what SB3 will create after wrapping
            # VecTransposeImage transposes Dict spaces with image observations from (H, W, C) to (C, H, W)
            from gym import spaces as gym_spaces
            
            # Create the transposed observation space to match what SB3 expects after VecTransposeImage
            if isinstance(base_obs_space, gym_spaces.Dict):
                transposed_spaces = {}
                for key, space in base_obs_space.spaces.items():
                    if isinstance(space, gym_spaces.Box) and len(space.shape) == 3:
                        # This is an image space - transpose from (H, W, C) to (C, H, W)
                        h, w, c = space.shape
                        transposed_spaces[key] = gym_spaces.Box(
                            low=0,
                            high=255,
                            shape=(c, h, w),
                            dtype=space.dtype
                        )
                    else:
                        # Non-image space, keep as-is
                        transposed_spaces[key] = space
                data["observation_space"] = gym_spaces.Dict(transposed_spaces)
            else:
                # Not a Dict space, use as-is
                data["observation_space"] = base_obs_space
            
            if hasattr(env, 'action_space'):
                data["action_space"] = env.action_space
            elif hasattr(env, 'envs') and len(env.envs) > 0:
                data["action_space"] = env.envs[0].action_space
            else:
                raise ValueError("Could not extract action_space from environment")
            
            console.log(f"[green]Spaces extracted (transposed for VecTransposeImage): obs_space={data['observation_space']}, act_space={data['action_space']}[/green]")
        
        # Also ensure other required SB3 fields exist
        fields_were_missing = False
        if "verbose" not in data:
            data["verbose"] = 1
            fields_were_missing = True
        if "policy_class" not in data:
            # Import the policy class
            from stable_baselines3.sac.policies import MultiInputPolicy
            data["policy_class"] = MultiInputPolicy
            fields_were_missing = True
        if "policy_kwargs" not in data:
            data["policy_kwargs"] = {}
            fields_were_missing = True
        
        # If spaces or other required fields were missing, we need to create a patched checkpoint file
        # because SB3's load() reads directly from the zip file
        if spaces_were_missing or fields_were_missing:
            # Create a temporary file with the complete data
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Save the patched data to the temp file
            save_to_zip_file(tmp_path, data=data, params=params, pytorch_variables=pytorch_variables)
            console.log(f"[cyan]Created patched checkpoint with spaces at {tmp_path}[/cyan]")
            
            # Load from the patched checkpoint
            model = super(CustomSAC, cls).load(tmp_path, env=env, device=device, custom_objects=custom_objects, **kwargs)
            
            # Clean up temp file
            try:
                import os
                os.unlink(tmp_path)
            except:
                pass
        else:
            # Spaces exist, load normally
            model = super(CustomSAC, cls).load(path, env=env, device=device, custom_objects=custom_objects, **kwargs)
        
        # Fixed target entropy from paper: H̄ = -dim(A)
        model.fixed_target_entropy = -3.0
        
        # NO ALPHA BOUNDS - Paper does not clamp α (Table 1, Appendix D)
        # α is learned freely via gradient descent on the entropy loss
        
        # Setup Mixed Precision Training for resumed model
        model.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        if model.use_mixed_precision:
            model.scaler = GradScaler()
            console.log("[green]Mixed Precision Training (FP16) enabled for resumed model[/green]")
        else:
            model.scaler = None
        
        # Get current alpha for logging
        if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:
            current_alpha = torch.exp(model.log_ent_coef).item()
        else:
            current_alpha = model.ent_coef_tensor.item()
        
        print(f"[LOAD] Resumed model - timesteps: {model.num_timesteps}, "
              f"α: {current_alpha:.4f}, target_H: {model.fixed_target_entropy:.3f} (fixed, no clamp)")
        
        # Auto-load replay buffer if found
        if load_replay_buffer:
            # Construct replay buffer path from model path
            model_path = pathlib.Path(path)
            # Remove .zip extension if present
            base_path = model_path.with_suffix('') if model_path.suffix == '.zip' else model_path
            replay_buffer_path = pathlib.Path(str(base_path) + '_replay_buffer')
            
            # Try to load replay buffer
            if model.load_replay_buffer(replay_buffer_path):
                console.log(f"[green]Successfully loaded replay buffer alongside model[/green]")
            else:
                console.log(f"[yellow]No replay buffer found - starting with empty buffer[/yellow]")
                console.log(f"[yellow]Training will need {model.learning_starts} steps before learning starts[/yellow]")

        return model    





class CarlaEnv(gym.Env):

    def __init__(self, num_npcs=5, frame_skip=8, visualize=True,

                 fixed_delta_seconds=0.05, camera_width=84, camera_height=84, model=None, 
                 arduino_port='/dev/ttyACM0', rotate_maps=True, no_rendering_mode=False):

        # Store no_rendering_mode setting (applied after CARLA connection)
        self._no_rendering_mode = no_rendering_mode
        
        # Add Arduino serial communication setup
        try:
            self.arduino = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            console.log("[green]Arduino connected successfully[/green]")
        except Exception as e:
            console.log(f"[red]Failed to connect to Arduino: {e}[/red]")
            self.arduino = None
        
        # Store rotate_maps setting (will be used after connecting to CARLA)
        self._rotate_maps_setting = rotate_maps
            
        super(CarlaEnv, self).__init__()
        self.visualize = visualize

        self.frame_skip = frame_skip



        # Handle SDL video driver configuration.

        if self.visualize:

            if os.environ.get("SDL_VIDEODRIVER") == "dummy":

                del os.environ["SDL_VIDEODRIVER"]

            import pygame

            self.pygame = pygame

        else:

            if not os.environ.get("SDL_VIDEODRIVER"):

                os.environ["SDL_VIDEODRIVER"] = "dummy"



        # Set display dimensions.

        if self.visualize:

            self.display_width = 600

            self.display_height = 400

            pygame.init()

            self.display = pygame.display.set_mode((self.display_width, self.display_height))

            pygame.display.set_caption("Driver's View")

            self.clock = pygame.time.Clock()

        else:

            self.display_width = camera_width

            self.display_height = camera_height

            self.display = None

            self.clock = None



        # Connect to CARLA server.

        self.client = carla.Client('localhost', 2000)

        self.client.set_timeout(10.0)

        self.world = self.client.get_world()

        
        # Map rotation settings
        self.rotate_maps = self._rotate_maps_setting  # Enable/disable map rotation
        self.available_maps = self._get_available_maps()
        self.current_map_index = 0
        self.fixed_delta_seconds = fixed_delta_seconds  # Store for reuse after map change
        
        # Explicitly log map rotation status for debugging
        console.log(f"[bold cyan]╔═══════════════════════════════════════════════════════╗[/bold cyan]")
        console.log(f"[bold cyan]║           MAP ROTATION CONFIGURATION                  ║[/bold cyan]")
        console.log(f"[bold cyan]╠═══════════════════════════════════════════════════════╣[/bold cyan]")
        console.log(f"[cyan]║ rotate_maps setting: {self.rotate_maps}[/cyan]")
        console.log(f"[cyan]║ Number of maps found: {len(self.available_maps)}[/cyan]")
        console.log(f"[cyan]║ Available maps: {self.available_maps}[/cyan]")
        console.log(f"[cyan]║ Will rotate: {self.rotate_maps and len(self.available_maps) > 1}[/cyan]")
        console.log(f"[bold cyan]╚═══════════════════════════════════════════════════════╝[/bold cyan]")



        # Enable synchronous mode.

        settings = self.world.get_settings()

        if not settings.synchronous_mode:

            settings.synchronous_mode = True

        settings.fixed_delta_seconds = fixed_delta_seconds
        
        # Enable no_rendering_mode for faster training (disables CARLA viewport rendering)
        # Sensors (camera, LIDAR) still work - only the spectator/viewport is disabled
        if self._no_rendering_mode:
            settings.no_rendering_mode = True
            console.log("[green]No-rendering mode enabled - 2-3x faster training![/green]")
        
        self.world.apply_settings(settings)

        console.log(f"[green]Synchronous mode enabled (fixed_delta_seconds = {fixed_delta_seconds})[/green]")


        
        # BEV (Bird's Eye View) Occupancy Grid for LIDAR - MUST be defined before observation_space!
        # FRONT-FACING configuration: 180° FOV covers right-front-left
        # Vehicle at bottom center of grid, looking forward (up in grid)
        self.bev_grid_size = 64  # Grid resolution (64x64)
        self.bev_forward_range = 25.0  # Forward range in meters (matches LIDAR range)
        self.bev_lateral_range = 25.0  # Lateral range ±25m (total 50m width)
        self.bev_resolution = self.bev_forward_range / self.bev_grid_size  # ~0.39m per pixel

        # Expand observation space.
        # GRAYSCALE IMAGE: 84x84x1 - standard RL vision size for efficiency
        # Memory: 84*84*1 = 7,056 bytes (very efficient!)
        # Grayscale preserves edge information for lane detection
        # 
        # STATE VECTOR (4D): Only non-visual info that CNN can't see!
        # - x, y: Position (for path planning context)
        # - speed: Current velocity (crucial for control)
        # - yaw: Heading direction (for steering decisions)
        # 
        # NOTE: lidar_min_distance REMOVED - redundant with LIDAR BEV CNN!
        # The CNN extracts spatial obstacle info from the 64x64 BEV grid.

        self.observation_space = spaces.Dict({

            "image": spaces.Box(low=0, high=255, shape=(camera_height, camera_width, 1), dtype=np.uint8),

            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            
            # BEV LIDAR occupancy grid: 64x64 single-channel image
            # Values: 0 = free space, 255 = occupied (uint8 for memory efficiency)
            # Using uint8 saves 75% memory vs float32 (60MB vs 240MB in replay buffer)
            "lidar_bev": spaces.Box(low=0, high=255, shape=(self.bev_grid_size, self.bev_grid_size, 1), dtype=np.uint8)

        })



        # Define action space.

        self.action_space = spaces.Box(

            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),

            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),

            dtype=np.float32

        )



        # Initialize vehicle, sensors, and reward variables.

        self.vehicle = None

        self.camera = None

        self.camera_image = None

        self.camera_image_obs = None

        self.num_npcs = num_npcs

        self.npc_vehicles = []



        self.previous_location = None

        self.previous_steering = 0.0

        self.previous_speed = 0.0

        self.previous_acceleration = 0.0

        self.collision_history = False

        self.collision_sensor = None



        # Existing sensors.

        self.lane_invasion_sensor = None

        self.lane_invasion_history = False

        self.imu_sensor = None

        self.imu_data = None



        # LIDAR sensor.

        self.lidar_sensor = None

        self.lidar_min_distance = float('inf')
        
        # Forward-facing camera for CNN observations (separate from visualization camera)
        self.forward_camera = None
        
        # BEV occupancy grid (bev_grid_size already defined above before observation_space)
        self.lidar_bev = np.zeros((self.bev_grid_size, self.bev_grid_size), dtype=np.uint8)



        # Stuck counter.

        self.stuck_counter = 0

        self.idle_penalty = 0.0  

        self.camera_width = camera_width

        self.camera_height = camera_height



        self.previous_throttle = 0.0  # Initialize previous_throttle
        self.previous_brake = 0.0  # Initialize previous_brake
        self.last_reward = 0.0  # Initialize last_reward
        self.model = model  # Initialize model

        self.episode_count = 0  # Initialize episode_count

        self.font = pygame.font.Font(None, 36)  # Initialize font for rendering text

        self.reward_history = [0] * 50  # Initialize reward history

        self.success_rate = 0.0  # Initialize success rate

        # State normalization parameters (for normalizing state to ~[-1, 1] range)
        # These are approximate ranges based on typical CARLA values
        # NOTE: lidar_min_distance REMOVED - now handled by LIDAR BEV CNN
        self.state_mean = np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float32)  # [x, y, speed, yaw]
        self.state_std = np.array([200.0, 200.0, 15.0, 180.0], dtype=np.float32)  # Std dev for normalization
        
        # Image augmentation flag (enabled during training for robustness)
        self.use_augmentation = True

        self.spawn_npcs()

    def _get_available_maps(self):
        """
        Get list of available maps from CARLA server.
        Filters to include only valid, loadable Town maps.
        """
        try:
            all_maps = self.client.get_available_maps()
            console.log(f"[cyan][MAP DETECTION] All available maps from CARLA: {all_maps}[/cyan]")
            
            # Only include Town maps (they are the reliable, well-tested maps)
            # Mine_01 and other custom maps often have issues
            # Accept both regular Town and Town_Opt variants
            usable_maps = [
                m for m in all_maps 
                if 'Town' in m and 'Template' not in m
            ]
            # Sort for consistent ordering
            usable_maps.sort()
            
            if not usable_maps:
                console.log("[yellow]No Town maps found, map rotation disabled[/yellow]")
                current_map = self.world.get_map().name
                return [current_map]
            
            console.log(f"[green][MAP DETECTION] Usable Town maps: {usable_maps}[/green]")
            return usable_maps
        except Exception as e:
            console.log(f"[red]Error getting available maps: {e}[/red]")
            import traceback
            console.log(f"[red]{traceback.format_exc()}[/red]")
            return [self.world.get_map().name]
    
    def _change_map(self, map_name):
        """
        Change to a different map. This reloads the entire world.
        
        Args:
            map_name: Name of the map to load (e.g., 'Town01', '/Game/Carla/Maps/Town01')
            
        CRITICAL: Uses reset_settings=False to preserve synchronous mode.
        Without this, CARLA resets to async mode which breaks training.
        """
        try:
            console.log(f"[magenta][MAP CHANGE] Loading map: {map_name}[/magenta]")
            
            # Destroy all existing actors first
            self._cleanup_actors()
            
            # CRITICAL: Use reset_settings=False to preserve synchronous mode!
            # Default is True which resets sync mode to False and breaks training.
            # See: https://carla.readthedocs.io/en/latest/python_api/#carla.Client.load_world
            self.world = self.client.load_world(map_name, reset_settings=False)
            
            # Wait for the world to be fully ready (increased from 2.0s for stability)
            time.sleep(3.0)
            
            # Double-check and enforce synchronous mode on the new world
            # Even with reset_settings=False, we re-apply to be safe
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                console.log(f"[yellow][MAP CHANGE] Sync mode was reset! Re-enabling...[/yellow]")
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.fixed_delta_seconds
            # Preserve no_rendering_mode setting
            if self._no_rendering_mode:
                settings.no_rendering_mode = True
            self.world.apply_settings(settings)
            
            # Tick more times to fully stabilize the new world (increased from 5)
            for _ in range(10):
                self.world.tick()
            
            # Re-spawn NPCs on the new map
            self.npc_vehicles = []  # Clear old NPC list
            self.spawn_npcs()
            
            # Verify the map actually changed
            new_map_name = self.world.get_map().name
            console.log(f"[green][MAP CHANGE] Successfully loaded: {new_map_name}[/green]")
            return True
            
        except Exception as e:
            console.log(f"[red][MAP CHANGE] Failed to load map {map_name}: {e}[/red]")
            import traceback
            console.log(f"[red]{traceback.format_exc()}[/red]")
            return False
    
    def _cleanup_actors(self):
        """Clean up all spawned actors before map change."""
        try:
            # Destroy sensors
            for sensor in [self.camera, self.forward_camera, self.collision_sensor, self.lane_invasion_sensor,
                          self.imu_sensor, self.lidar_sensor]:
                if sensor is not None:
                    try:
                        sensor.destroy()
                    except:
                        pass
            
            # Destroy vehicle
            if self.vehicle is not None:
                try:
                    self.vehicle.destroy()
                except:
                    pass
            
            # Destroy NPCs
            for npc in self.npc_vehicles:
                try:
                    npc.destroy()
                except:
                    pass
            
            # Reset references
            self.vehicle = None
            self.camera = None
            self.forward_camera = None
            self.collision_sensor = None
            self.lane_invasion_sensor = None
            self.imu_sensor = None
            self.lidar_sensor = None
            self.npc_vehicles = []
            
        except Exception as e:
            console.log(f"[yellow]Warning during actor cleanup: {e}[/yellow]")



    def _get_small_vehicle_blueprints(self):

        blueprint_library = self.world.get_blueprint_library()

        candidate_filters = ["vehicle.mercedes.coupe", "vehicle.audi.tt", "vehicle.mini.cooper"]

        for candidate in candidate_filters:

            blueprints = blueprint_library.filter(candidate)

            if blueprints:

                console.log(f"[green]Using small vehicle blueprint filter: {candidate}[/green]")

                return blueprints

            else:

                console.log(f"[yellow]No blueprints found for '{candidate}'.[/yellow]")

        blueprints = blueprint_library.filter("vehicle.*")

        if blueprints:

            console.log("[yellow]No candidate small vehicle blueprints found; using any available vehicle.[/yellow]")

            return blueprints

        return []



    def spawn_npcs(self):

        blueprint_library = self.world.get_blueprint_library()

        spawn_points = self.world.get_map().get_spawn_points()

        if len(spawn_points) == 0:

            console.log("[red]Warning: No spawn points available for NPCs![/red]")

            return



        npc_blueprints = self._get_small_vehicle_blueprints()

        if not npc_blueprints:

            console.log("[red]No valid small vehicle blueprints found for NPC spawn.[/red]")

            return



        random.shuffle(spawn_points)

        for i in range(min(self.num_npcs, len(spawn_points))):

            npc_bp = random.choice(npc_blueprints)

            npc_vehicle = self.world.try_spawn_actor(npc_bp, spawn_points[i])

            if npc_vehicle is not None:

                npc_vehicle.set_autopilot(True)

                self.npc_vehicles.append(npc_vehicle)

                console.log(f"[cyan]Spawned NPC small vehicle at spawn point {i}.[/cyan]")



    def reset(self):

        self.episode_count += 1
        
        # Map rotation: change map on EVERY episode when enabled
        # This provides maximum environment diversity for better generalization
        # Note: Map loading takes ~2-5 seconds, so training will be slower
        should_rotate = (
            self.rotate_maps and 
            len(self.available_maps) > 1
        )
        
        if should_rotate:
            # Pick a random map (different from current one)
            # Extract just the map name (e.g., "Town01") from full path for comparison
            current_map_full = self.world.get_map().name
            current_map_name = current_map_full.split('/')[-1] if '/' in current_map_full else current_map_full
            
            console.log(f"[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
            console.log(f"[bold magenta][MAP ROTATION] Episode {self.episode_count} - Rotating map![/bold magenta]")
            console.log(f"[cyan][MAP ROTATION] Current map: {current_map_name}[/cyan]")
            console.log(f"[cyan][MAP ROTATION] Available maps: {self.available_maps}[/cyan]")
            
            # Filter out current map by checking if map name appears in either string
            available_choices = []
            for m in self.available_maps:
                m_name = m.split('/')[-1] if '/' in m else m
                if m_name != current_map_name:
                    available_choices.append(m)
            
            if available_choices:
                next_map = random.choice(available_choices)
                console.log(f"[bold yellow][MAP ROTATION] CHANGING TO: {next_map}[/bold yellow]")
                console.log(f"[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
                success = self._change_map(next_map)
                if not success:
                    console.log(f"[red][MAP ROTATION] Map change failed! Continuing with current map.[/red]")
            else:
                console.log(f"[yellow][MAP ROTATION] No other maps available, staying on {current_map_name}[/yellow]")
                console.log(f"[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
        else:
            # Log why we're NOT rotating (for debugging)
            if not self.rotate_maps:
                console.log(f"[dim][MAP ROTATION] Disabled by setting[/dim]")
            elif len(self.available_maps) <= 1:
                console.log(f"[dim][MAP ROTATION] Only 1 map available: {self.available_maps}[/dim]")
        
        # Update model reference if it was passed through DummyVecEnv
        if hasattr(self, 'venv') and hasattr(self.venv, 'envs'):
            self.model = self.venv.envs[0].model
        
        self.previous_speed = 0.0
        self.previous_steering = 0.0
        self.previous_throttle = 0.0
        self.previous_brake = 0.0
        self.previous_acceleration = 0.0  # Reset to prevent jerk calculation errors
        self.last_reward = 0.0
        self.lidar_min_distance = float('inf')  # Reset LIDAR to prevent stale data
        self.lidar_bev = np.zeros((self.bev_grid_size, self.bev_grid_size), dtype=np.uint8)  # Reset BEV grid
        self.collision_history = False  # Reset collision flag for new episode
        self.camera_image_obs = None  # Reset camera image to prevent stale data
        self.camera_image = None  # Reset display image as well

        resources_to_cleanup = []
        

        try:

            for sensor in [self.camera, self.forward_camera, self.collision_sensor, self.lane_invasion_sensor,

                        self.imu_sensor, self.lidar_sensor]:

                if sensor is not None:

                    try:

                        sensor.destroy()

                    except Exception as e:

                        console.log(f"[red]Error destroying sensor: {e}[/red]")



            if self.vehicle is not None:

                try:

                    self.vehicle.destroy()

                except Exception as e:

                    console.log(f"[red]Error destroying vehicle: {e}[/red]")



            self.stuck_counter = 0

            self.idle_penalty = 0.0  # Reset idle penalty



            blueprint_library = self.world.get_blueprint_library()

            blueprints = self._get_small_vehicle_blueprints()

            if not blueprints:

                raise RuntimeError("No valid small vehicle blueprints found for agent!")

            vehicle_bp = random.choice(blueprints)



            spawn_points = self.world.get_map().get_spawn_points()

            if not spawn_points:

                raise RuntimeError("No spawn points available!")

            

            self.vehicle = None

            random.shuffle(spawn_points)

            for spawn_point in spawn_points:

                try:

                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

                    if self.vehicle is not None:

                        resources_to_cleanup.append(self.vehicle)

                        break

                except Exception as e:

                    console.log(f"[yellow]Failed to spawn vehicle at {spawn_point}: {e}[/yellow]")

                    continue

                    

            if self.vehicle is None:

                raise RuntimeError("Failed to spawn agent vehicle after trying all spawn points!")



            # --- DOMAIN RANDOMIZATION: Randomize weather & lighting ---

            try:

                weather = carla.WeatherParameters(

                    cloudiness=random.uniform(0, 80),

                    precipitation=random.uniform(0, 30),

                    sun_altitude_angle=random.uniform(30, 90),

                    sun_azimuth_angle=random.uniform(0, 360),

                    fog_density=random.uniform(0, 30),

                    fog_distance=random.uniform(10, 100),

                    wetness=random.uniform(0, 100)

                )

                self.world.set_weather(weather)

            except Exception as e:

                console.log(f"[yellow]Failed to set weather: {e}[/yellow]")



            # --- Attach camera sensor ---

            try:

                camera_bp = blueprint_library.find('sensor.camera.rgb')

                camera_bp.set_attribute('enable_postprocess_effects', 'True')

                camera_bp.set_attribute('exposure_mode', 'auto')

                camera_bp.set_attribute('image_size_x', str(self.display_width))

                camera_bp.set_attribute('image_size_y', str(self.display_height))

                camera_bp.set_attribute('fov', '100')

                camera_bp.set_attribute('sensor_tick', '0.1')

                camera_transform = carla.Transform(

                    carla.Location(x=-6.0, y=0.0, z=4.0),

                    carla.Rotation(pitch=-15, yaw=0, roll=0)

                )

                self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

                resources_to_cleanup.append(self.camera)

                self.camera.listen(lambda image: self._process_image(image))

            except Exception as e:

                console.log(f"[red]Failed to create camera: {e}[/red]")

                self.camera = None



            # --- Attach FORWARD-FACING camera for CNN observations ---
            # This camera is positioned on the hood looking forward (driver's POV)
            # Used for the 84x84 grayscale CNN input - separate from visualization camera
            try:
                forward_camera_bp = blueprint_library.find('sensor.camera.rgb')
                forward_camera_bp.set_attribute('enable_postprocess_effects', 'True')
                forward_camera_bp.set_attribute('exposure_mode', 'auto')
                # Use observation dimensions directly (84x84) for efficiency
                forward_camera_bp.set_attribute('image_size_x', str(self.camera_width))
                forward_camera_bp.set_attribute('image_size_y', str(self.camera_height))
                forward_camera_bp.set_attribute('fov', '90')  # Narrower FOV for better forward detail
                forward_camera_bp.set_attribute('sensor_tick', '0.1')
                # Hood-level forward-facing position (like a dashcam)
                forward_camera_transform = carla.Transform(
                    carla.Location(x=2.0, y=0.0, z=1.4),  # Front of hood, ~1.4m height
                    carla.Rotation(pitch=-8, yaw=0, roll=0)  # Slight downward angle to see road
                )
                self.forward_camera = self.world.spawn_actor(forward_camera_bp, forward_camera_transform, attach_to=self.vehicle)
                resources_to_cleanup.append(self.forward_camera)
                self.forward_camera.listen(lambda image: self._process_forward_image(image))
                console.log(f"[green]Forward CNN camera attached (84x84, hood-level, FOV=90°)[/green]")
            except Exception as e:
                console.log(f"[red]Failed to create forward camera: {e}[/red]")
                self.forward_camera = None



            # --- Attach collision sensor ---

            try:

                collision_bp = blueprint_library.find('sensor.other.collision')

                collision_transform = carla.Transform()

                self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)

                resources_to_cleanup.append(self.collision_sensor)

                self.collision_sensor.listen(lambda event: self._on_collision(event))

                self.collision_history = False

            except Exception as e:

                console.log(f"[red]Failed to create collision sensor: {e}[/red]")

                self.collision_sensor = None



            # --- Attach lane invasion sensor ---

            try:

                lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')

                lane_invasion_transform = carla.Transform()

                self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, lane_invasion_transform, attach_to=self.vehicle)

                resources_to_cleanup.append(self.lane_invasion_sensor)

                self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

                self.lane_invasion_history = False

            except Exception as e:

                console.log(f"[red]Failed to create lane invasion sensor: {e}[/red]")

                self.lane_invasion_sensor = None



            # --- Attach IMU sensor ---

            try:

                imu_bp = blueprint_library.find('sensor.other.imu')

                imu_transform = carla.Transform()

                self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)

                resources_to_cleanup.append(self.imu_sensor)

                self.imu_sensor.listen(lambda imu: self._on_imu_update(imu))

                self.imu_data = None

            except Exception as e:

                console.log(f"[red]Failed to create IMU sensor: {e}[/red]")

                self.imu_sensor = None



            # --- Attach LIDAR sensor ---

            try:

                lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

                # LIDAR settings optimized for FRONT-FACING BEV occupancy grid
                # 180° FOV covers right-front-left which is most important for driving
                # This gives 2x point density in the relevant area vs 360°
                lidar_bp.set_attribute('range', '25')  # 25m range for BEV
                lidar_bp.set_attribute('rotation_frequency', '20')  # Match simulation rate
                lidar_bp.set_attribute('channels', '32')  # Vertical resolution
                lidar_bp.set_attribute('points_per_second', '60000')  # Dense point cloud
                    
                # Vertical FOV settings:
                # At z=1.8m height, lower_fov=-15° hits ground at ~6.7m
                # This is acceptable - we filter ground in _on_lidar_update()
                lidar_bp.set_attribute('upper_fov', '10.0')   # Look up for overhead obstacles
                lidar_bp.set_attribute('lower_fov', '-15.0')  # Down angle for near obstacles
                
                # HORIZONTAL FOV: 180° front-facing (right 90° + front + left 90°)
                # Covers the driving-relevant area with higher point density
                # Rear is not critical - we're not driving backwards
                lidar_bp.set_attribute('horizontal_fov', '180.0')  # Front semicircle
                
                # Position: roof level (z=1.8m), slightly forward for better front coverage
                lidar_transform = carla.Transform(carla.Location(x=0.5, z=1.8), carla.Rotation(pitch=0))

                self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)

                resources_to_cleanup.append(self.lidar_sensor)

                self.lidar_sensor.listen(lambda lidar: self._on_lidar_update(lidar))

            except Exception as e:

                console.log(f"[red]Failed to create LIDAR sensor: {e}[/red]")

                self.lidar_sensor = None



            for _ in range(10):

                try:

                    self.world.tick()

                except Exception as e:

                    console.log(f"[yellow]Error during world tick: {e}[/yellow]")



            # Get initial state

            try:

                transform = self.vehicle.get_transform()

                velocity = self.vehicle.get_velocity()

                speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

                self.previous_location = transform.location

                self.previous_steering = 0.0

                self.previous_speed = speed

                self.previous_acceleration = 0.0

            except Exception as e:

                console.log(f"[red]Error getting initial vehicle state: {e}[/red]")

                transform = carla.Transform()

                speed = 0.0

                self.previous_location = transform.location

                self.previous_steering = 0.0

                self.previous_speed = 0.0

                self.previous_acceleration = 0.0



            # Build state vector (4D): position, speed, heading
            # NOTE: lidar_min_distance REMOVED - redundant with LIDAR BEV CNN!
            # The CNN extracts spatial obstacle info from the 64x64 BEV grid.

            state = np.array([

                transform.location.x,

                transform.location.y,

                speed,

                transform.rotation.yaw

            ], dtype=np.float32)

            

            # Normalize the state for better neural network training

            state = self._normalize_state(state)

            # Wait for camera to produce first image (max 20 additional ticks)
            # This prevents returning stale/None images after reset
            wait_ticks = 0
            while self.camera_image_obs is None and wait_ticks < 20:
                try:
                    self.world.tick()
                    wait_ticks += 1
                except Exception as e:
                    console.log(f"[yellow]Error waiting for camera: {e}[/yellow]")
                    break
            
            if self.camera_image_obs is None:
                console.log("[yellow][RESET] Camera image not ready, using blank image[/yellow]")

            console.log(f"[green][RESET] Agent spawned at {transform.location} using a small vehicle[/green]")
            # Clear the resources_to_cleanup list since we're successful
            resources_to_cleanup = []
            
            # Get BEV LIDAR grid (add channel dimension)
            lidar_bev_obs = self.lidar_bev[:, :, np.newaxis]  # Shape: (64, 64, 1)

            return {"image": self.camera_image_obs if self.camera_image_obs is not None else np.zeros((self.camera_height, self.camera_width, 1), dtype=np.uint8),

                    "state": state,
                    "lidar_bev": lidar_bev_obs}

                    

        except Exception as e:

            console.log(f"[red][ERROR] Exception during reset: {e}[/red]")

            # Clean up any resources that were created before the exception

            for resource in resources_to_cleanup:

                try:

                    resource.destroy()

                except Exception as cleanup_error:

                    console.log(f"[red]Error cleaning up resource: {cleanup_error}[/red]")

                    

            # Reset all sensors to None

            self.vehicle = None

            self.camera = None

            self.forward_camera = None

            self.collision_sensor = None

            self.lane_invasion_sensor = None

            self.imu_sensor = None

            self.lidar_sensor = None

            

            # Return a blank observation (normalized zero state - 4D: x, y, speed, yaw)

            default_state = self._normalize_state(np.zeros(4, dtype=np.float32))

            return {"image": np.zeros((self.camera_height, self.camera_width, 1), dtype=np.uint8),

                    "state": default_state,
                    "lidar_bev": np.zeros((self.bev_grid_size, self.bev_grid_size, 1), dtype=np.uint8)}



    def _process_image(self, image):
        """
        Process VISUALIZATION camera image (third-person chase camera).
        This is only used for the pygame display window, NOT for CNN observations.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA format

        # Use cv2.cvtColor to convert from BGRA to RGB for display
        rgb_image = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)

        # Use the full-resolution RGB image for rendering to maintain visual quality
        if self.visualize:
            self.camera_image = rgb_image
        else:
            # For non-visual mode, store grayscale (but render() won't use it)
            gray_image = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
            self.camera_image = gray_image

    def _process_forward_image(self, image):
        """
        Process FORWARD-FACING camera image for CNN observations.
        This camera is hood-level, looking forward (dashcam style).
        Used for the 84x84 grayscale observation sent to the policy network.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA format
        
        # Convert to GRAYSCALE for observation (better lane detection, less memory)
        # Grayscale preserves edge information while reducing 3 channels to 1
        gray_image = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)

        # Camera is already 84x84 (set in blueprint attributes), but resize just in case
        if gray_image.shape[0] != self.camera_height or gray_image.shape[1] != self.camera_width:
            resized_gray = cv2.resize(gray_image, (self.camera_width, self.camera_height))
        else:
            resized_gray = gray_image
        
        # Apply augmentation during training for robustness (grayscale-compatible)
        if self.use_augmentation and random.random() < 0.3:  # 30% chance of augmentation
            resized_gray = self._augment_grayscale_image(resized_gray)
        
        # Add channel dimension: (H, W) -> (H, W, 1) for observation space compatibility
        self.camera_image_obs = resized_gray[:, :, np.newaxis].astype(np.uint8)

    def _augment_image(self, image):
        """
        Apply random augmentations to RGB image for training robustness.
        DEPRECATED: Use _augment_grayscale_image for grayscale observations.
        """
        augmented = image.copy()
        
        # Random brightness adjustment (simulates different times of day)
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            augmented = np.clip(augmented * brightness_factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if random.random() < 0.3:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(augmented, axis=(0, 1), keepdims=True)
            augmented = np.clip((augmented - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Random color jitter (slight hue shift)
        if random.random() < 0.2:
            # Convert to HSV, shift hue slightly, convert back
            hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Random Gaussian noise (simulates sensor noise)
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, augmented.shape).astype(np.float32)
            augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return augmented

    def _augment_grayscale_image(self, image):
        """
        Apply random augmentations to GRAYSCALE image for training robustness.
        Simulates different lighting conditions and sensor noise.
        
        Args:
            image: 2D grayscale image (H, W) with values 0-255
            
        Returns:
            Augmented grayscale image (H, W) with values 0-255
        """
        augmented = image.copy().astype(np.float32)
        
        # Random brightness adjustment (simulates different times of day)
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            augmented = augmented * brightness_factor
        
        # Random contrast adjustment
        if random.random() < 0.3:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(augmented)
            augmented = (augmented - mean) * contrast_factor + mean
        
        # Random Gaussian noise (simulates sensor noise)
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, augmented.shape)
            augmented = augmented + noise
        
        # Clip and convert back to uint8
        return np.clip(augmented, 0, 255).astype(np.uint8)

    def _normalize_state(self, state):
        """
        Normalize state values to approximately [-1, 1] range for better neural network training.
        Uses z-score normalization: (x - mean) / std
        
        State vector (4D): [x, y, speed, yaw]
        NOTE: lidar_min_distance removed - now handled by LIDAR BEV CNN.
        """
        return (state - self.state_mean) / (self.state_std + 1e-8)



    def _on_collision(self, event):

        self.collision_history = True

        console.log("[red][COLLISION] Collision detected![/red]")



    def _on_lane_invasion(self, event):

        self.lane_invasion_history = True

        console.log("[red][LANE INVASION] Lane invasion detected![/red]")



    def _on_imu_update(self, imu):

        self.imu_data = imu



    def _on_lidar_update(self, lidar_measurement):
        """
        Process LIDAR data to:
        1. Find minimum distance to obstacles (for state vector)
        2. Generate BEV (Bird's Eye View) occupancy grid for spatial awareness
        
        LIDAR Configuration:
        - Position: roof level (z=1.8m), slightly forward (x=0.5)
        - horizontal_fov = 180° (front semicircle: right-front-left)
        - lower_fov = -15° (hits ground at ~6.7m)
        - range = 25m
        - 32 channels, 60k points/sec for BEV
        
        Ground Filtering Geometry:
        - Sensor at 1.8m height, lower beam at -15°
        - Ground plane: z_sensor = -1.8m (relative to LIDAR)
        - At distance d, -15° beam z = -d * tan(15°) = -0.268 * d
        - Solution: Distance-dependent ground filter
        
        BEV Grid (FRONT-FACING SEMICIRCLE):
        - 64x64 pixels covering 25m forward x 50m wide area
        - Vehicle is at BOTTOM CENTER of grid (row 63, col 32)
        - Row 0 = 25m ahead (front), Row 63 = vehicle position
        - Col 0 = 25m right, Col 63 = 25m left
        - This gives 2x resolution vs 360° grid for the area that matters!
        - Cell value: 0 = free, 255 = occupied
        """
        points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
        points = np.reshape(points, (-1, 4))  # x, y, z, intensity
        
        if points.shape[0] == 0:
            self.lidar_min_distance = float('inf')
            self.lidar_bev = np.zeros((self.bev_grid_size, self.bev_grid_size), dtype=np.uint8)
            return
        
        # Extract coordinates (LIDAR frame: x=forward, y=left, z=up)
        x = points[:, 0]  # Forward (positive = front of car)
        y = points[:, 1]  # Left/right (positive = left)
        z = points[:, 2]  # Up/down relative to sensor (at 1.8m height)
        
        horizontal_dist = np.sqrt(x**2 + y**2)
        
        # ========== GEOMETRIC GROUND FILTERING ==========
        # Problem: Simple z > -1.5m filter fails for close ground points
        # 
        # Ground plane is at z_ground = -1.8m (sensor at 1.8m above ground)
        # Obstacle must be ABOVE ground to be valid
        # 
        # Method: Point must be at least 0.15m above ground plane
        #         z_point > -1.8m + 0.15m = -1.65m (absolute ground threshold)
        #         BUT we also add distance-dependent margin for sensor noise
        #         
        # Final: z > -1.65m AND z > (ground_ray_z + margin)
        #        where ground_ray_z = -0.268 * dist (what -15° beam would hit at that dist)
        #        margin = 0.3m (above the theoretical ground ray)
        #
        # This filters:
        # - Actual ground reflections (z ≈ -1.8m at all distances)
        # - Near-ground noise along the lower beam angle
        
        ground_threshold = -1.65  # Must be at least 0.15m above ground
        ground_ray_z = -0.268 * horizontal_dist  # Theoretical -15° beam position
        ground_margin = 0.3  # Must be 0.3m above where ground ray would be
        
        # Point is valid if ABOVE ground AND ABOVE the "ground ray + margin"
        not_ground = (z > ground_threshold) & (z > ground_ray_z + ground_margin)
        
        # ========== SELF-DETECTION FILTER ==========
        # Filter points within 0.5m (car body reflections)
        not_self = horizontal_dist > 0.5
        
        # ========== VALID POINTS FOR MIN DISTANCE ==========
        # For min distance: only consider points in driving corridor (|y| < 5m)
        valid_mask = not_self & not_ground & (np.abs(y) < 5.0)
        valid_distances = horizontal_dist[valid_mask]
        
        # Calculate minimum distance
        if valid_distances.size > 0:
            self.lidar_min_distance = np.min(valid_distances)
        else:
            self.lidar_min_distance = float('inf')
        
        # ========== GENERATE BEV OCCUPANCY GRID (FRONT-FACING) ==========
        # Reset grid (uint8 for memory efficiency: 60MB vs 240MB in replay buffer)
        bev_grid = np.zeros((self.bev_grid_size, self.bev_grid_size), dtype=np.uint8)
        
        # Filter for BEV:
        # - Must pass ground filter and self filter
        # - Upper bound: ignore points > 3m above ground (overhangs, bridges)
        # - Only keep FORWARD points (x > 0) since we have 180° front FOV
        upper_threshold = 1.2  # Relative to sensor: 1.8 + 1.2 = 3.0m above ground
        bev_mask = not_self & not_ground & (z < upper_threshold) & (x > 0)  # Front only
        
        bev_x = x[bev_mask]
        bev_y = y[bev_mask]
        
        if len(bev_x) > 0:
            # Convert to grid coordinates for FRONT-FACING BEV:
            # - Vehicle is at BOTTOM CENTER (row 63, col 32)
            # - Row 0 = max range ahead, Row 63 = vehicle position
            # - Col 0 = max range right, Col 63 = max range left
            # - Grid covers bev_forward_range forward (x) and ±bev_lateral_range sideways (y)
            
            # Use instance variables for consistency (set in __init__)
            max_forward = self.bev_forward_range  # 25m forward range
            max_lateral = self.bev_lateral_range  # ±25m lateral (50m total width)
            
            # Map x (forward distance) to row: 0m -> row 63, max_forward -> row 0
            grid_row = ((max_forward - bev_x) / max_forward * (self.bev_grid_size - 1)).astype(np.int32)
            
            # Map y (lateral) to column: -max_lateral (right) -> col 0, +max_lateral (left) -> col 63
            grid_col = ((bev_y + max_lateral) / (2 * max_lateral) * (self.bev_grid_size - 1)).astype(np.int32)
            
            # Clip to valid grid indices
            grid_row = np.clip(grid_row, 0, self.bev_grid_size - 1)
            grid_col = np.clip(grid_col, 0, self.bev_grid_size - 1)
            
            # Mark occupied cells with value 255 (uint8 max)
            bev_grid[grid_row, grid_col] = 255
        
        self.lidar_bev = bev_grid



    def _send_arduino_data(self, speed, reward, steering, throttle, brake):
        """Send data to Arduino for LCD and LEDs"""
        if self.arduino is None:
            return  # Silently skip if Arduino not connected

        try:
            # Format: "S{speed}R{reward}C{controls}"
            # Controls: 4 bits for LEDs (Left, Right, Throttle, Brake)
            controls = 0
            if steering < -0.2:  # Left turn
                controls |= 0b1000
            elif steering > 0.2:  # Right turn
                controls |= 0b0100
            
            if throttle > 0.1:   # Acceleration
                controls |= 0b0010
            if brake > 0.1:      # Brake
                controls |= 0b0001

            # Format data string
            data = f"S{min(int(speed * 3.6), 999):03d}"  # Speed in km/h (3 digits)
            data += f"R{min(int(reward * 10), 999):03d}"  # Reward * 10 (3 digits)
            data += f"C{controls:01d}\n"  # Controls as single digit

            console.log(f"[cyan]Sending to Arduino: {data}[/cyan]")  # Debug print
            self.arduino.write(data.encode())
            self.arduino.flush()  # Make sure data is sent

        except Exception as e:
            console.log(f"[red]Error sending data to Arduino: {e}[/red]")



    def step(self, action):

        steering, throttle, brake = action
        
        # Safety check: if vehicle is None (e.g., after a failed reset), return early
        if self.vehicle is None:
            console.log("[red][STEP] Vehicle is None! Returning dummy observation.[/red]")
            default_state = self._normalize_state(np.zeros(4, dtype=np.float32))
            return {"image": np.zeros((self.camera_height, self.camera_width, 1), dtype=np.uint8),
                    "state": default_state,
                    "lidar_bev": np.zeros((self.bev_grid_size, self.bev_grid_size, 1), dtype=np.uint8)}, -100.0, True, {"error": "vehicle_none"}
        
        # NOTE: previous_steering is updated AFTER reward calculation to compute steering change
        control = carla.VehicleControl(

            steer=float(steering),

            throttle=float(throttle),

            brake=float(brake)

        )

        for _ in range(self.frame_skip):

            self.vehicle.apply_control(control)

            self.world.tick()

            if self.visualize:

                self.render()



        transform = self.vehicle.get_transform()

        current_location = transform.location

        velocity = self.vehicle.get_velocity()

        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        dt = self.frame_skip * self.world.get_settings().fixed_delta_seconds
        dt = max(dt, 1e-6)  # Protect against division by zero



        # --- Stuck Detection ---
        # Stricter stuck detection: speed < 2 m/s for 10 seconds triggers respawn
        # With frame_skip=8 and fixed_delta_seconds=0.05, each step = 0.4s
        # 10 seconds = 25 steps

        dx = current_location.x - self.previous_location.x

        dy = current_location.y - self.previous_location.y

        distance_moved = np.sqrt(dx**2 + dy**2)

        stuck_speed_threshold = 2.0  # m/s - stricter threshold (was 0.7)

        stuck_move_threshold = 0.8   # meters per step

        stuck_max_steps = 25         # ~10 seconds (was 50 steps)

        if speed < stuck_speed_threshold and distance_moved < stuck_move_threshold:

            self.stuck_counter += 1

            stuck_penalty = -1.0

        else:

            self.stuck_counter = 0

            stuck_penalty = 0.0

        if self.stuck_counter >= stuck_max_steps:

            console.log("[red][STUCK] Vehicle is stuck (speed < 2 m/s for 10s). Respawning vehicle.[/red]")

            obs = self.reset()

            return obs, -20, True, {"stuck": True}



        # --- Idle Penalty ---

        idle_threshold = 0.5  

        idle_increment = 10  

        if speed < idle_threshold:

            self.idle_penalty += idle_increment

        else:

            self.idle_penalty = 0.0





        # Build state vector (4D): position, speed, heading
        # NOTE: lidar_min_distance REMOVED - redundant with LIDAR BEV CNN!
        state = np.array([

            current_location.x,

            current_location.y,

            speed,

            transform.rotation.yaw

        ], dtype=np.float32)

        

        # Normalize the state for better neural network training

        state = self._normalize_state(state)



 

        target_speed = 11.5 

        sigma = 3.0

        distance_reward = distance_moved * 5.0

        target_speed_reward = 10.0 * np.exp(-((speed - target_speed) ** 2) / (2 * sigma**2))

        A = 10.0

        log_speed_reward = A * (np.log((speed / 12.0) + 1) - np.log(2))

        smooth_steering_penalty = -abs(steering - self.previous_steering) * 0.005

        acceleration = (speed - self.previous_speed) / dt

        acceleration_penalty = -abs(acceleration) * 0.00005

        jerk = (acceleration - self.previous_acceleration) / dt

        jerk_threshold = 0.5

        if abs(jerk) > jerk_threshold:

            jerk_penalty = -0.005 * (abs(jerk) - jerk_threshold)

        else:

            jerk_penalty = 0.0

        self.previous_acceleration = acceleration

        energy_penalty = - (throttle * 0.02 + brake * 0.02)

        energy_bonus = 0.2 if abs(speed - target_speed) < 1.0 and throttle < 0.7 else 0.0

        collision_penalty = -47.0 if self.collision_history else 0.0

        lane_invasion_penalty = -23 if self.lane_invasion_history else 0.0

        self.lane_invasion_history = False

        if self.imu_data is not None:

            angular_speed = np.linalg.norm([

                self.imu_data.gyroscope.x,

                self.imu_data.gyroscope.y,

                self.imu_data.gyroscope.z

            ])

            imu_penalty = -0.05 * angular_speed

        else:

            imu_penalty = 0.0

        # LIDAR BEV is used as observation input to the CNN (CombinedExtractor) only.
        # We removed occupancy-based reward shaping to let the policy learn end-to-end
        # what BEV patterns matter for safe driving. This avoids:
        # - Coupling sensor noise/artifacts into reward signal
        # - Hand-tuned penalties that may not generalize across maps
        # - Reward hacking where agent exploits sensor quirks
        # The collision penalty (-27) provides the main safety signal.



        reward = (distance_reward + target_speed_reward + log_speed_reward +

                  smooth_steering_penalty*0.3 + acceleration_penalty*0.1 + jerk_penalty*0.1 +

                  energy_penalty*0.001 + energy_bonus + collision_penalty +

                  lane_invasion_penalty + imu_penalty*0.9 - self.idle_penalty + stuck_penalty) * 2.0



        self.previous_location = current_location

        self.previous_speed = speed

        self.previous_steering = steering  # Update AFTER reward calculation for proper steering change detection

        # Store throttle and brake as instance variables
        self.previous_throttle = throttle  # Store for rendering
        self.previous_brake = brake  # Store for rendering
        
        # Get BEV LIDAR grid (add channel dimension)
        lidar_bev_obs = self.lidar_bev[:, :, np.newaxis]  # Shape: (64, 64, 1)

        obs = {"image": self.camera_image_obs if self.camera_image_obs is not None else np.zeros((self.camera_height, self.camera_width, 1), dtype=np.uint8),

               "state": state,
               "lidar_bev": lidar_bev_obs}
        
        # Calculate front occupancy for logging (BEV is observation-only, not used in reward)
        # In front-facing BEV: row 0-7 = far ahead (22-25m), rows 56-63 = very close (0-3m)
        danger_occ = 0.0
        if hasattr(self, 'lidar_bev') and self.lidar_bev is not None:
            grid_center = self.bev_grid_size // 2
            # Danger zone: bottom 8 rows (close to vehicle), center 16 columns
            # Rows 56-63 = closest 3m (25m / 64 * 8 ≈ 3.1m), cols 24-40 = center ±6m
            danger_zone = self.lidar_bev[56:64, grid_center-8:grid_center+8]
            danger_occ = np.mean(danger_zone) / 255.0
        
        # Log with color based on danger zone occupancy (for monitoring only)
        log_color = "red" if danger_occ > 0.05 else "yellow" if danger_occ > 0.01 else "blue"
        console.log(f"[{log_color}][STEP] Loc=({current_location.x:.2f},{current_location.y:.2f}), "
                    f"Speed={speed:.2f}, BEV_DangerZone={danger_occ:.1%}, "
                    f"Reward={reward:.2f}[/{log_color}]")
        if self.visualize:
            self.render()

        self.last_reward = reward  # Update last_reward

        # Define done conditions
        done = False  # Initialize done flag
        
        # Check termination conditions
        if self.collision_history:
            done = True
        
        # Note: stuck_counter >= stuck_max_steps is already handled above with early return

        if done:  # Now done is properly defined before being used
            # Note: episode_count is incremented in reset(), not here (to avoid double-counting)
            # Update success rate based on your criteria
            self.success_rate = self.success_rate * 0.95 + (0 if self.collision_history else 100) * 0.05

        # Send data to Arduino after each step
        self._send_arduino_data(
            speed=self.previous_speed,
            reward=reward,
            steering=self.previous_steering,
            throttle=self.previous_throttle,
            brake=self.previous_brake
        )

        return obs, reward, done, {}



    def render(self, mode='human'):
        if self.visualize:
            # Non-blocking event processing - prevents freeze when window minimized
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return
                    # Handle window minimize/restore without blocking
                    elif event.type == pygame.ACTIVEEVENT:
                        pass  # Just consume the event, don't block
            except Exception as e:
                pass  # Ignore pygame event errors

            if self.camera_image is not None:
                try:
                    # Check if display is still valid (window not minimized)
                    if pygame.display.get_active():
                        surface = pygame.surfarray.make_surface(self.camera_image.swapaxes(0, 1))
                        self.display.blit(surface, (0, 0))
                        pygame.display.flip()
                    self.clock.tick(60)  # Limit frame rate to 60 FPS
                except pygame.error as e:
                    pass  # Window minimized or display issue, continue training
                except Exception as e:
                    console.log(f"[red]Error during rendering: {e}[/red]")

            # Skip dashboard rendering if window is minimized/inactive
            if not pygame.display.get_active():
                return
            
            # Wrap all dashboard rendering in try-except to prevent blocking
            try:
                # Create a dashboard showing current controls
                font = pygame.font.SysFont("Arial", 18)
                speed = getattr(self, 'previous_speed', 0)
                target_speed = getattr(self, 'target_speed', 11.5)

                speed_color = (0, 255, 0) if target_speed - 2 < speed < target_speed + 2 else (255, 255, 0) if speed > 0 else (255, 0, 0)

                # Get steering and action values safely
                steering = getattr(self, 'previous_steering', 0.0)
                throttle = getattr(self, 'previous_throttle', 0.0)
                brake = getattr(self, 'previous_brake', 0.0)

                # Basic information always available
                texts = [
                    (f"Speed: {speed:.1f} km/h", speed_color),
                    (f"Reward: {self.last_reward:.1f}", (0, 255, 0) if self.last_reward > 0 else (255, 0, 0)),
                ]

                # Add episode info
                episode_text = f"Episode: {self.episode_count}"  # Removed success rate
                texts.append((episode_text, (255, 255, 255)))

                # Render text on the display
                for i, (text, color) in enumerate(texts):
                    text_surface = font.render(text, True, color)
                    self.display.blit(text_surface, (10, 10 + i * 25))

                # Reward history graph
                graph_width = 150
                graph_height = 50
                graph_x = self.display_width - graph_width - 20
                graph_y = 50

                # Background for the graph
                pygame.draw.rect(self.display, (30, 30, 30), (graph_x, graph_y, graph_width, graph_height))

                # Store last 50 rewards in a circular buffer if needed
                if not hasattr(self, 'reward_history'):
                    self.reward_history = [0] * 50

                # Update rewards history
                self.reward_history = self.reward_history[1:] + [self.last_reward]

                # Normalize rewards for display
                max_r = max(max(self.reward_history), 1)
                min_r = min(min(self.reward_history), -1)
                range_r = max(max_r - min_r, 1)

                # Plot rewards
                for i, r in enumerate(self.reward_history):
                    normalized_r = (r - min_r) / range_r
                    bar_height = int(normalized_r * graph_height)
                    color = (0, 255, 0) if r > 0 else (255, 0, 0)
                    pygame.draw.line(self.display, color, (graph_x + i * 3, graph_y + graph_height), (graph_x + i * 3, graph_y + graph_height - bar_height), 2)

                # Add label for the graph
                reward_text = font.render("Reward History", True, (255, 255, 255))
                self.display.blit(reward_text, (graph_x, graph_y - 20))

                # Add steering indicator
                center_x = self.display_width - 80
                center_y = self.display_height - 80
                radius = 30
                pygame.draw.circle(self.display, (255, 255, 255), (center_x, center_y), radius, 2)
                indicator_x = center_x + int(self.previous_steering * radius)
                indicator_y = center_y
                pygame.draw.circle(self.display, (255, 255, 255), (indicator_x, indicator_y), 5)

                # Add throttle indicator
                throttle_x = self.display_width - 40
                throttle_height = 60
                throttle_width = 10
                throttle_y = self.display_height - throttle_height - 50
                
                # Draw throttle background
                pygame.draw.rect(self.display, (50, 50, 50), 
                               (throttle_x, throttle_y, throttle_width, throttle_height))
                
                # Draw current throttle level (green bar)
                current_height = int(self.previous_throttle * throttle_height)
                if current_height > 0:
                    pygame.draw.rect(self.display, (0, 255, 0),
                                   (throttle_x, throttle_y + throttle_height - current_height,
                                    throttle_width, current_height))

                # Add throttle label
                throttle_label = font.render(f"T", True, (255, 255, 255))
                self.display.blit(throttle_label, (throttle_x - 5, throttle_y - 20))

                # Add enhanced training progress bar
                progress_width = 500  # Made much longer
                progress_height = 10  # Made thinner
                progress_x = (self.display_width - progress_width) // 2  # Centered horizontally
                progress_y = self.display_height - 35
                border_radius = 7  # Added border radius for curved edges
                
                # Draw progress bar border (white outline) with rounded corners
                pygame.draw.rect(self.display, (255, 255, 255), 
                               (progress_x-2, progress_y-2, progress_width+4, progress_height+4), 
                               2, border_radius=border_radius)
                
                # Create a surface for the background gradient with alpha channel
                gradient_surface = pygame.Surface((progress_width, progress_height), pygame.SRCALPHA)
                
                # Draw gradient progress fill
                for i in range(progress_width):
                    color_value = 30 + (i / progress_width) * 20
                    pygame.draw.line(gradient_surface, (color_value, color_value, color_value, 255),
                                   (i, 0), (i, progress_height))
                
                # Draw the background surface with rounded corners
                pygame.draw.rect(gradient_surface, (0, 0, 0, 0), 
                               (0, 0, progress_width, progress_height), 
                               border_radius=border_radius-2)
                self.display.blit(gradient_surface, (progress_x, progress_y))
                
                # Calculate and draw current progress (if total_timesteps available)
                if self.model is not None and hasattr(self.model, 'num_timesteps'):
                    # Get total timesteps - use _total_timesteps if available, or just show current timesteps
                    total_steps = getattr(self.model, '_total_timesteps', None) or 100000  # Default for display
                    progress = min(1.0, self.model.num_timesteps / total_steps)
                    current_width = int(progress * progress_width)
                    
                    if current_width > 0:
                        # Create a surface for the progress gradient with alpha channel
                        progress_surface = pygame.Surface((current_width, progress_height), pygame.SRCALPHA)
                        
                        # Draw gradient progress fill
                        for i in range(current_width):
                            # Create a gradient from blue to green
                            blue = max(0, 255 * (1 - i / progress_width))
                            green = min(255, 255 * (i / progress_width))
                            pygame.draw.line(progress_surface, (0, green, blue, 255),
                                           (i, 0), (i, progress_height))
                        
                        # Draw the progress surface with rounded corners
                        if current_width >= border_radius * 2:
                            pygame.draw.rect(progress_surface, (0, 0, 0, 0), 
                                           (0, 0, current_width, progress_height), 
                                           border_radius=border_radius-2)
                            self.display.blit(progress_surface, (progress_x, progress_y))
                    
                    # Add progress percentage text with shadow
                    progress_text = f"Training Progress: {progress * 100:.1f}%"
                    font = pygame.font.Font(None, 28)  # Slightly larger font
                    
                    # Center the text above the progress bar
                    text_width = font.size(progress_text)[0]
                    text_x = progress_x + (progress_width - text_width) // 2
                    
                    # Draw text shadow
                    text_shadow = font.render(progress_text, True, (0, 0, 0))
                    self.display.blit(text_shadow, (text_x + 2, progress_y - 25))
                    
                    # Draw main text
                    text_surface = font.render(progress_text, True, (255, 255, 255))
                    self.display.blit(text_surface, (text_x, progress_y - 27))
                    
                    # Add timestep counter below, also centered
                    timestep_text = f"Steps: {self.model.num_timesteps:,}/{total_steps:,}"
                    timestep_width = font.size(timestep_text)[0]
                    timestep_x = progress_x + (progress_width - timestep_width) // 2
                    
                    timestep_shadow = font.render(timestep_text, True, (0, 0, 0))
                    self.display.blit(timestep_shadow, (timestep_x + 2, progress_y + progress_height + 5))
                    
                    timestep_surface = font.render(timestep_text, True, (200, 200, 200))
                    self.display.blit(timestep_surface, (timestep_x, progress_y + progress_height + 3))

                # Update display
                pygame.display.update()
            except pygame.error:
                pass  # Window minimized, skip rendering
            except Exception as e:
                pass  # Don't let rendering errors stop training



    def close(self):

        try:

            for sensor in [self.collision_sensor, self.camera, self.forward_camera, self.imu_sensor,

                        self.lane_invasion_sensor, self.lidar_sensor]:

                if sensor is not None:

                    try:

                        sensor.destroy()

                    except Exception as e:

                        console.log(f"[red]Error destroying sensor during close: {e}[/red]")

            

            if self.vehicle is not None:

                try:

                    self.vehicle.destroy()

                except Exception as e:

                    console.log(f"[red]Error destroying vehicle during close: {e}[/red]")

            

            for npc in self.npc_vehicles:

                try:

                    npc.destroy()

                except Exception as e:

                    console.log(f"[red]Error destroying NPC vehicle during close: {e}[/red]")

            

            # Reset all references to None

            self.vehicle = None

            self.camera = None

            self.forward_camera = None

            self.collision_sensor = None

            self.lane_invasion_sensor = None

            self.imu_sensor = None

            self.lidar_sensor = None

            self.npc_vehicles = []

            

            if self.visualize:

                try:

                    pygame.quit()

                except Exception as e:

                    console.log(f"[red]Error quitting pygame: {e}[/red]")



            # Close Arduino connection
            if hasattr(self, 'arduino') and self.arduino is not None:
                try:
                    self.arduino.close()
                    console.log("[yellow]Arduino connection closed.[/yellow]")
                except Exception as e:
                    console.log(f"[red]Error closing Arduino connection: {e}[/red]")

        except Exception as e:

            console.log(f"[red][ERROR] Exception during close: {e}[/red]")

        finally:

            console.log("[yellow][CLOSE] Environment closed.[/yellow]")
