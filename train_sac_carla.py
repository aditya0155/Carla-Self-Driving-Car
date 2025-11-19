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

import serial
import time
    
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

            # Directly access the entropy coefficient from the model.

            current_ent_coef = torch.exp(self.model.log_ent_coef).item()

            print(f"[INFO] Step {self.n_calls}: Entropy Coefficient = {current_ent_coef}")

        return True





# --- Define a Residual Block for the CNN ---

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        residual = x

        out = self.relu(self.conv1(x))

        out = self.conv2(out)

        return self.relu(out + residual)



# --- Enhanced Feature Extractor with Attention Fusion ---

class CombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1024):

        # Expect state to be 8-dim now.

        super(CombinedExtractor, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=8, stride=4),

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

            nn.Flatten()

        )

        # Determine the CNN output dimension.

        with torch.no_grad():

            dummy_image = torch.zeros(1, 3, 84, 84)

            cnn_out_dim = self.cnn(dummy_image).shape[1]

        

        state_dim = observation_space.spaces["state"].shape[0]  # now 8

        # MLP for state.

        self.mlp = nn.Sequential(

            nn.Linear(state_dim, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU()

        )



        self.attention_layer = nn.Linear(cnn_out_dim, 128)

        

        combined_dim = cnn_out_dim + 128

        self.fc = nn.Sequential(

            nn.Linear(combined_dim, features_dim),

            nn.ReLU()

        )

        self._features_dim = features_dim



    def forward(self, observations):

        # Process image.

        if observations["image"].ndim == 4 and observations["image"].shape[1] == 3:

            image = observations["image"].float() / 255.0

        else:

            image = observations["image"].permute(0, 3, 1, 2).float() / 255.0

        

        image_features = self.cnn(image)  # shape: [batch, cnn_out_dim]

        state_features = self.mlp(observations["state"].float())  # shape: [batch, 128]

        



        attn_weights = torch.sigmoid(self.attention_layer(image_features))  # shape: [batch, 128]

        fused_state = state_features * attn_weights

        

        concatenated = torch.cat([image_features, fused_state], dim=1)



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
        
        env = CarlaEnv(
            num_npcs=5, 
            frame_skip=8, 
            visualize=True,
            fixed_delta_seconds=0.05,
            camera_width=84,
            camera_height=84,
            model=None,
            arduino_port=arduino_port
        )
        return env
    return _init





class CustomCheckpointCallback(CheckpointCallback):

    def _on_training_start(self) -> None:

        if hasattr(self.model, "num_timesteps"):

            self.n_calls = self.model.num_timesteps

        super()._on_training_start()





def objective(trial):

    lr = 2e-4

    batch_size = trial.suggest_categorical('batch_size', [512])

    tau = 0.002191

    

    policy_kwargs = dict(

        features_extractor_class=CombinedExtractor, 

        features_extractor_kwargs=dict(features_dim=1024),

        net_arch=dict(pi=[1024, 1024], qf=[1024, 1024])

    )

    env = DummyVecEnv([make_env() for _ in range(1)])

    model = SAC(

        "MultiInputPolicy",

        env,

        verbose=1,

        tensorboard_log="./sac_tensorboard/",

        device="cuda" if torch.cuda.is_available() else "cpu",

        learning_rate=lr,

        buffer_size=30000,

        learning_starts=1000,

        batch_size=batch_size,

        tau=tau,

        ent_coef=0.5,  # Fixed entropy coefficient during optimization

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

        features_extractor_kwargs=dict(features_dim=1024),

        net_arch=dict(pi=[1024, 1024], qf=[1024, 1024])

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

            device="cuda" if torch.cuda.is_available() else "cpu",

            total_timesteps_for_entropy=args.total_timesteps

        )   

        

        # Update the logger

        model.set_logger(new_logger)

        

        # Calculate the remaining timesteps

        remaining_timesteps = args.total_timesteps - model.num_timesteps

        

        # Log the current state

        with torch.no_grad():

            current_alpha = torch.exp(model.log_ent_coef).item()

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

            buffer_size=30000,

            learning_starts=1000,

            batch_size=batch_size,

            tau=tau,

            policy_kwargs=policy_kwargs,

            total_timesteps_for_entropy=args.total_timesteps

        )



    console.log("[bold green]Starting training...")

    callbacks = [checkpoint_callback, loss_logging_callback, stuck_detection_callback, entropy_logging_callback]



    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)



    # After model creation, update the environment with the model reference
    env.envs[0].model = model



    model_path = "sac_carla_model_enhanced"

    model.save(model_path)


    console.log(f"[bold green]Model saved to {model_path}[/bold green]")
