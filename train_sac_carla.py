import os

import argparse

import gym

import torch

import torch.nn as nn

import numpy as np

from stable_baselines3 import SAC

from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rich.console import Console

import optuna

from stable_baselines3.common.logger import configure

# Note: Mixed Precision Training is handled in carla_env.py's CustomSAC class
    
from carla_env import CarlaEnv, CustomSAC



console = Console()



# NEW: Configure a logger that outputs to stdout, CSV, and tensorboard.

new_logger = configure("./sac_tensorboard/", ["stdout", "csv", "tensorboard"])



from stable_baselines3.common.callbacks import BaseCallback



class EntropyLoggingCallback(BaseCallback):

    def __init__(self, log_interval: int = 1000, verbose: int = 1):

        super(EntropyLoggingCallback, self).__init__(verbose)

        self.log_interval = log_interval



    def _on_step(self) -> bool:

        if self.n_calls % self.log_interval == 0:

            # Access entropy coefficient (check log_ent_coef FIRST for auto entropy mode)

            if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:

                current_ent_coef = torch.exp(self.model.log_ent_coef).item()

            elif hasattr(self.model, 'ent_coef_tensor'):

                current_ent_coef = self.model.ent_coef_tensor.item()

            else:

                current_ent_coef = 0.0

            print(f"[INFO] Step {self.n_calls}: Entropy Coefficient = {current_ent_coef:.4f}")

        return True





# --- Define a Residual Block for the CNN ---

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        return self.relu(out + residual)



# --- Enhanced Feature Extractor with Attention Fusion ---

class CombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1024):

        # State dimension is read dynamically from observation_space

        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        
        # Get image dimensions from observation space
        # NOTE: SB3's VecTransposeImage wrapper transposes the obs space to (C, H, W)
        # So we need to detect if it's channel-first or channel-last
        image_shape = observation_space.spaces["image"].shape
        
        # Detect format: if first dim is small (1-3) and last dim is large, it's channel-first
        if image_shape[0] <= 3 and image_shape[2] > 3:
            # Channel-first format: (C, H, W) - after VecTransposeImage
            image_channels, image_height, image_width = image_shape
        else:
            # Channel-last format: (H, W, C) - original gym format
            image_height, image_width, image_channels = image_shape
        
        console.log(f"[cyan]Image observation shape: {image_shape} -> C={image_channels}, H={image_height}, W={image_width}[/cyan]")
        
        # ========== IMAGE CNN ==========
        # Updated for GRAYSCALE input (1 channel instead of 3)
        # Input: 120x160x1 grayscale for better lane detection
        # Uses Global Average Pooling to fix output size regardless of input resolution
        # This prevents FC layer explosion when using higher resolutions
        self.cnn = nn.Sequential(

            nn.Conv2d(image_channels, 64, kernel_size=8, stride=4),  # Dynamic channel count

            nn.ReLU(),

            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),

            nn.ReLU(),

            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),

            nn.ReLU(),

            ResidualBlock(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=1),

            nn.ReLU(),

            # Global Average Pooling: reduces any spatial size to (256, 4, 4)
            # This fixes the FC layer size regardless of input resolution!
            # 120x160 or 84x84 → always outputs 256*4*4 = 4096 features
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten()

        )

        # Verify CNN output dimension with a dummy tensor
        # This catches any bugs in the architecture or channel format detection
        with torch.no_grad():
            dummy_image = torch.zeros(1, image_channels, image_height, image_width)
            actual_cnn_out = self.cnn(dummy_image)
            cnn_out_dim = actual_cnn_out.shape[1]
        
        expected_cnn_out = 256 * 4 * 4  # = 4096 (from AdaptiveAvgPool2d((4, 4)))
        
        if cnn_out_dim != expected_cnn_out:
            console.log(f"[red][WARNING] CNN output mismatch! Expected {expected_cnn_out}, got {cnn_out_dim}[/red]")
            console.log(f"[red]Input shape: (1, {image_channels}, {image_height}, {image_width})[/red]")
            console.log(f"[red]Output shape: {actual_cnn_out.shape}[/red]")
        
        console.log(f"[green]CNN output dimension: {cnn_out_dim} (verified with dummy tensor)[/green]")
        
        # ========== LIDAR BEV CNN ==========
        # Processes the 64x64 single-channel BEV occupancy grid
        # NOTE: Same format detection as image - VecTransposeImage affects all image-like observations
        lidar_shape = observation_space.spaces["lidar_bev"].shape
        
        # Detect format: if first dim is small (1-3) and last dim is large, it's channel-first
        if lidar_shape[0] <= 3 and lidar_shape[2] > 3:
            # Channel-first format: (C, H, W) - after VecTransposeImage
            lidar_channels, lidar_height, lidar_width = lidar_shape
        else:
            # Channel-last format: (H, W, C) - original gym format
            lidar_height, lidar_width, lidar_channels = lidar_shape
        
        console.log(f"[cyan]LIDAR BEV observation shape: {lidar_shape} -> C={lidar_channels}, H={lidar_height}, W={lidar_width}[/cyan]")
        
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(lidar_channels, 32, kernel_size=5, stride=2),    # 64x64 -> 30x30
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),   # 30x30 -> 14x14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),   # 14x14 -> 6x6
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Determine LIDAR CNN output dimension
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, lidar_channels, lidar_height, lidar_width)
            lidar_cnn_out_dim = self.lidar_cnn(dummy_lidar).shape[1]
        
        # ========== STATE MLP ==========
        state_dim = observation_space.spaces["state"].shape[0]  # dynamically determined (currently 5)
        # MLP for state with LayerNorm for stability

        self.mlp = nn.Sequential(

            nn.Linear(state_dim, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU()

        )

        
        # ========== FUSION LAYERS ==========
        self.attention_layer = nn.Linear(cnn_out_dim, 128)
        
        # LIDAR feature projection to match state features
        self.lidar_projection = nn.Sequential(
            nn.Linear(lidar_cnn_out_dim, 128),
            nn.ReLU()
        )

        
        # Combined: image_features + state_features + lidar_features
        combined_dim = cnn_out_dim + 128 + 128  # image + state + lidar = 4096 + 128 + 128 = 4352

        # Two-stage FC to reduce parameters: 4352 -> 512 -> features_dim
        # This reduces FC params from 4.4M to ~800K (5.5x reduction!)
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

        

        # Initialize weights using Kaiming/He initialization (best practice for ReLU networks)

        self._initialize_weights()
        
        # Log parameter count for debugging VRAM usage
        total_params = sum(p.numel() for p in self.parameters())
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        lidar_params = sum(p.numel() for p in self.lidar_cnn.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        attn_params = sum(p.numel() for p in self.attention_layer.parameters())
        lidar_proj_params = sum(p.numel() for p in self.lidar_projection.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        
        console.log(f"[bold yellow]╔═══════════════════════════════════════════════════════╗[/bold yellow]")
        console.log(f"[bold yellow]║      FEATURE EXTRACTOR PARAMETER COUNT                ║[/bold yellow]")
        console.log(f"[bold yellow]╠═══════════════════════════════════════════════════════╣[/bold yellow]")
        console.log(f"[yellow]║ Image CNN:        {cnn_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ LIDAR CNN:        {lidar_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ State MLP:        {mlp_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ Attention Layer:  {attn_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ LIDAR Projection: {lidar_proj_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ FC Layer:         {fc_params:>10,} params[/yellow]")
        console.log(f"[bold yellow]╠═══════════════════════════════════════════════════════╣[/bold yellow]")
        console.log(f"[bold green]║ TOTAL:            {total_params:>10,} params[/bold green]")
        console.log(f"[bold yellow]╚═══════════════════════════════════════════════════════╝[/bold yellow]")
        
        # VRAM estimation (FP32: 4 bytes per param, FP16: 2 bytes)
        vram_fp32_mb = (total_params * 4) / (1024 * 1024)
        vram_fp16_mb = (total_params * 2) / (1024 * 1024)
        console.log(f"[cyan]Estimated VRAM (weights only): {vram_fp32_mb:.1f} MB (FP32) / {vram_fp16_mb:.1f} MB (FP16)[/cyan]")

    

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                nn.init.constant_(m.bias, 0)



    def forward(self, observations):

        # Process image (no autocast here - causes dtype issues with fc layer)
        # Handle both channel-first (batch, C, H, W) and channel-last (batch, H, W, C) formats
        # For grayscale: C=1, for RGB: C=3
        img = observations["image"]
        if img.ndim == 4:
            # Check if already in channel-first format by comparing dimensions
            # Channel-first: (batch, C, H, W) where C is typically 1 or 3
            # Channel-last: (batch, H, W, C) where C is typically 1 or 3
            # Heuristic: if dim 1 <= 3 and dim 3 > 3, likely channel-first
            if img.shape[1] <= 3 and img.shape[3] > 3:
                # Already channel-first format
                image = img.float() / 255.0
            else:
                # Channel-last format, permute to (batch, C, H, W)
                image = img.permute(0, 3, 1, 2).float() / 255.0
        else:
            # Fallback: assume channel-last and permute
            image = img.permute(0, 3, 1, 2).float() / 255.0
        
        image_features = self.cnn(image)  # shape: [batch, cnn_out_dim]
        
        # Process LIDAR BEV grid (uint8 [0, 255] -> float [0, 1])
        # LIDAR BEV is 64x64x1, same handling as image
        lidar_bev = observations["lidar_bev"]
        if lidar_bev.ndim == 4:
            # Check if already in channel-first format (batch, C, H, W)
            # For 64x64x1 BEV: channel-last is (batch, 64, 64, 1), channel-first is (batch, 1, 64, 64)
            if lidar_bev.shape[1] <= 3 and lidar_bev.shape[3] > 3:
                # Already channel-first format
                lidar = lidar_bev.float() / 255.0
            else:
                # Channel-last format, permute to (batch, C, H, W)
                lidar = lidar_bev.permute(0, 3, 1, 2).float() / 255.0
        else:
            # Fallback: assume channel-last and permute
            lidar = lidar_bev.permute(0, 3, 1, 2).float() / 255.0
        
        lidar_features = self.lidar_cnn(lidar)  # shape: [batch, lidar_cnn_out_dim]
        lidar_features = self.lidar_projection(lidar_features)  # shape: [batch, 128]
        
        # Process state
        state_features = self.mlp(observations["state"].float())  # shape: [batch, 128]
        
        # Attention-weighted fusion (image guides state importance)
        attn_weights = torch.sigmoid(self.attention_layer(image_features))  # shape: [batch, 128]
        fused_state = state_features * attn_weights
        
        # Concatenate all features: image + state (attention-weighted) + lidar
        concatenated = torch.cat([image_features, fused_state, lidar_features], dim=1)

        concatenated = torch.nan_to_num(concatenated, nan=0.0, posinf=1e3, neginf=-1e3)
        return self.fc(concatenated)







def make_env():

    def _init():

        # Try to find the correct port
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        arduino_port = None
        
        for port in ports:
            if "Arduino" in port.description or "USB Serial Device" in port.description:
                arduino_port = port.device
                break
                
        if arduino_port is None:
            console.log("[yellow]No Arduino found. Available ports:[/yellow]")
            for port in ports:
                console.log(f"[yellow]{port.device}: {port.description}[/yellow]")
            arduino_port = 'COM3'  # Default fallback
            
        console.log(f"[green]Using Arduino port: {arduino_port}[/green]")
        
        # ========== PERFORMANCE TIP ==========
        # Set no_rendering_mode=True to disable CARLA's main viewport rendering
        # This gives 2-3x faster training and ELIMINATES the minimized window slowdown!
        # Sensors (camera, LIDAR) still work normally - only spectator view is disabled
        # Set to False if you want to watch the CARLA 3D viewport during training
        # 
        # GRAYSCALE 84x84: Standard RL vision size for efficiency
        # - 84x84x1 = 7,056 bytes (very memory efficient!)
        # - Grayscale preserves edge information (lane lines are high contrast)
        env = CarlaEnv(
            num_npcs=5, 
            frame_skip=8, 
            visualize=True,  # This is pygame window (your dashboard)
            fixed_delta_seconds=0.05,
            camera_width=84,    # Standard RL vision size
            camera_height=84,   # Square 84x84 for efficiency
            model=None,
            arduino_port=arduino_port,
            rotate_maps=False,  # Disabled: only 1 map available, enable when more maps downloaded
            no_rendering_mode=True  # HUGE SPEEDUP: Disable CARLA viewport (fixes minimize slowdown!)
        )
        return env
    return _init





class CustomCheckpointCallback(CheckpointCallback):

    def _on_training_start(self) -> None:

        if hasattr(self.model, "num_timesteps"):

            self.n_calls = self.model.num_timesteps

        super()._on_training_start()





def objective(trial):

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    batch_size = trial.suggest_categorical('batch_size', [256, 512])

    tau = trial.suggest_float('tau', 0.001, 0.01)

    

    policy_kwargs = dict(

        features_extractor_class=CombinedExtractor,

        features_extractor_kwargs=dict(features_dim=1024),

        net_arch=dict(pi=[1024, 1024], qf=[1024, 1024]),
        
        # CRITICAL: Share feature extractor between actor and critics
        # Without this, SB3 creates 3 separate extractors (1 actor + 2 critics)
        # This reduces VRAM by ~2/3 and speeds up training
        share_features_extractor=True

    )

    env = DummyVecEnv([make_env() for _ in range(1)])

    model = CustomSAC(

        "MultiInputPolicy",

        env,

        verbose=1,

        tensorboard_log="./sac_tensorboard/",

        device="cuda" if torch.cuda.is_available() else "cpu",

        learning_rate=lr,

        buffer_size=22000,

        # Note: optimize_memory_usage not supported with Dict observation spaces

        learning_starts=1000,

        batch_size=batch_size,

        tau=tau,

        policy_kwargs=policy_kwargs

    )



    # Train for a short trial.

    model.learn(total_timesteps=10000)

    # Evaluate performance over 1000 steps.

    rewards = []

    obs = env.reset()   

    for _ in range(1000):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(action)

        rewards.append(reward)

        if done:

            obs = env.reset()

    avg_reward = np.mean(rewards)

    return avg_reward



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SAC on CARLA environment with advanced features")

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint/model file to resume training from")

    parser.add_argument("--total_timesteps", type=int, default=150000, help="Total timesteps for training")

    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")

    args = parser.parse_args()



    if args.optimize:

        console.log("[bold yellow]Starting hyperparameter optimization...[/bold yellow]")

        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=10)

        best_params = study.best_trial.params

        console.log(f"[bold green]Best hyperparameters: {best_params}[/bold green]")

        learning_rate = best_params['learning_rate']

        batch_size = best_params['batch_size']

        tau = best_params['tau']

    else:

        learning_rate = 2e-4

        batch_size = 512

        tau = 0.004



    console.rule("[bold green]Starting Training")

    env = DummyVecEnv([make_env() for _ in range(1)])



    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),  # Match paper's network size
        net_arch=dict(pi=[256, 256], qf=[256, 256]),  # Paper: 2 hidden layers, 256 units each
        
        # CRITICAL: Share feature extractor between actor and critics
        # Without this, SB3 creates 3 separate extractors (1 actor + 2 critics)
        # This reduces VRAM by ~2/3 and speeds up training
        share_features_extractor=True
    )

    checkpoint_callback = CustomCheckpointCallback(save_freq=2000, save_path='./checkpoints/', name_prefix='sac_carla')



    class LossLoggingCallback(BaseCallback):

        def __init__(self, log_interval: int = 1000, verbose: int = 1):

            super(LossLoggingCallback, self).__init__(verbose)

            self.log_interval = log_interval



        def _on_step(self) -> bool:

            if self.n_calls % self.log_interval == 0:

                self.logger.record("custom/progress", self.n_calls)

                if self.verbose > 0:

                    print(f"Step: {self.n_calls}")

            return True



    class StuckDetectionCallback(BaseCallback):     

        def __init__(self, verbose: int = 1):

            super(StuckDetectionCallback, self).__init__(verbose)



        def _on_step(self) -> bool: 

            infos = self.locals.get("infos", [])

            for info in infos:

                if isinstance(info, dict) and info.get("stuck", False):

                    print("[yellow][STUCK CALLBACK] Vehicle was respawned due to being stuck.[/yellow]")

            return True

        

    loss_logging_callback = LossLoggingCallback(log_interval=1000, verbose=1)

    stuck_detection_callback = StuckDetectionCallback(verbose=1)

    

    entropy_logging_callback = EntropyLoggingCallback(log_interval=1000, verbose=1)

    

    if args.resume is not None and os.path.exists(args.resume):

        console.log(f"[yellow]Resuming training from checkpoint: {args.resume}[/yellow]")

        

        # Load the model

        model = CustomSAC.load(

            args.resume, 

            env=env,

            device="cuda" if torch.cuda.is_available() else "cpu"

        )   

        

        # Update the logger

        model.set_logger(new_logger)

        

        # Calculate the remaining timesteps

        remaining_timesteps = args.total_timesteps - model.num_timesteps

        

        # Log the current state (check log_ent_coef first for auto entropy mode)

        if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:

            current_alpha = torch.exp(model.log_ent_coef).item()

        else:

            current_alpha = model.ent_coef_tensor.item()

        console.log(f"[cyan]Current entropy coefficient: {current_alpha:.4f}[/cyan]")

        console.log(f"[cyan]Current timesteps: {model.num_timesteps}[/cyan]")

        console.log(f"[cyan]Remaining timesteps: {remaining_timesteps}[/cyan]")

    else:

        console.log("[cyan]Creating a new model.[/cyan]")

        model = CustomSAC(

            "MultiInputPolicy",

            env,

            verbose=1,

            logger=new_logger,

            tensorboard_log="./sac_tensorboard/",

            device="cuda" if torch.cuda.is_available() else "cpu",

            learning_rate=learning_rate,

            buffer_size=15000,  # Reduced from 30000 to save ~600MB RAM

            # Note: optimize_memory_usage not supported with Dict observation spaces

            learning_starts=1000,

            batch_size=batch_size,

            tau=tau,

            policy_kwargs=policy_kwargs

        )



    console.log("[bold green]Starting training...")

    callbacks = [checkpoint_callback, loss_logging_callback, stuck_detection_callback, entropy_logging_callback]

    # IMPORTANT: Set model reference BEFORE training so render() can access it
    env.envs[0].model = model

    # When resuming, use remaining timesteps and don't reset the counter
    if args.resume is not None and os.path.exists(args.resume):
        remaining_timesteps = args.total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            console.log(f"[yellow]Model already trained for {model.num_timesteps} steps (target: {args.total_timesteps}). Nothing to do.[/yellow]")
        else:
            console.log(f"[cyan]Resuming training for {remaining_timesteps} more steps...[/cyan]")
            model.learn(total_timesteps=remaining_timesteps, callback=callbacks, reset_num_timesteps=False)
    else:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    model_path = "sac_carla_model_enhanced"

    model.save(model_path)


    console.log(f"[bold green]Model saved to {model_path}[/bold green]")
