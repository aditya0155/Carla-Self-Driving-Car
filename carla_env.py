import os

import gym 

from gym import spaces

import numpy as np

import carla

import random

import pygame

import cv2 

import time

from rich.console import Console

import math

import torch

from stable_baselines3 import SAC

import serial



console = Console()



import math

import torch

from stable_baselines3 import SAC



import torch

import math

from stable_baselines3 import SAC





class CustomSAC(SAC):

    def __init__(self, *args, total_timesteps_for_entropy=150000, **kwargs):

        kwargs.pop('logger', None)

        if 'ent_coef' in kwargs:

            del kwargs['ent_coef']

        super().__init__(*args, ent_coef="auto", **kwargs)

        

        self.total_timesteps_for_entropy = total_timesteps_for_entropy

        self.num_timesteps_at_start = self.num_timesteps

        self.initial_alpha = 1.0

        self.min_alpha = 0.01  

        self.current_alpha = self.initial_alpha  



    def learn(self, total_timesteps, callback=None, log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, progress_bar=False):

        if self.num_timesteps_at_start == 0 or reset_num_timesteps:

            self.num_timesteps_at_start = self.num_timesteps

        

        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)



    def _update_learning_rate(self, optimizers):

        """Update the entropy coefficient based on training progress"""

        super()._update_learning_rate(optimizers)

        

        # Calculate progress based on total timesteps since start
        total_elapsed = self.num_timesteps
        progress_fraction = min(1.0, total_elapsed / self.total_timesteps_for_entropy)
        


        # Linear interpolation between initial_alpha and min_alpha
        new_alpha = self.initial_alpha * (1.0 - progress_fraction) + self.min_alpha * progress_fraction
        self.current_alpha = new_alpha



        # Update the entropy coefficient
        with torch.no_grad():
            self.log_ent_coef.copy_(torch.log(torch.tensor([new_alpha], device=self.device)))



        if self.logger is not None:
            self.logger.record("train/entropy_coefficient", new_alpha)
            


        if self.num_timesteps % 1000 == 0:
            print(f"[DEBUG] timesteps: {self.num_timesteps}, progress: {progress_fraction:.4f}, entropy_coef: {new_alpha:.4f}")

    

    @classmethod

    def load(cls, path, env=None, device="auto", custom_objects=None, force_reset=True, total_timesteps_for_entropy=150000, **kwargs):

        """

        Load the model from a zip-file.

        """

        model = super(CustomSAC, cls).load(path, env, device, custom_objects, **kwargs)
        

        # Set the total_timesteps_for_entropy
        model.total_timesteps_for_entropy = total_timesteps_for_entropy
        

        with torch.no_grad():
            current_alpha = torch.exp(model.log_ent_coef).item()
        

        # Set the initial alpha to maintain the current trajectory
        model.initial_alpha = current_alpha
        model.current_alpha = current_alpha
        model.min_alpha = 0.01
        
        print(f"[LOAD] Resumed model - timesteps: {model.num_timesteps}/{model.total_timesteps_for_entropy}, "
              f"entropy_coefficient: {current_alpha:.4f}")

        

        return model    





class CarlaEnv(gym.Env):

    def __init__(self, num_npcs=5, frame_skip=8, visualize=True,

                 fixed_delta_seconds=0.05, camera_width=84, camera_height=84, model=None, arduino_port='/dev/ttyACM0'):

        # Add Arduino serial communication setup
        try:
            self.arduino = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            console.log("[green]Arduino connected successfully[/green]")
        except Exception as e:
            console.log(f"[red]Failed to connect to Arduino: {e}[/red]")
            self.arduino = None
            
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



        # Enable synchronous mode.

        settings = self.world.get_settings()

        if not settings.synchronous_mode:

            settings.synchronous_mode = True

        settings.fixed_delta_seconds = fixed_delta_seconds

        self.world.apply_settings(settings)

        console.log(f"[green]Synchronous mode enabled (fixed_delta_seconds = {fixed_delta_seconds})[/green]")



        # Setup PyGame window if visualizing.

        if self.visualize:

            pygame.init()

            self.display = pygame.display.set_mode((self.display_width, self.display_height))

            pygame.display.set_caption("Driver's View")

            self.clock = pygame.time.Clock()



        # Expand observation space.

        self.observation_space = spaces.Dict({

            "image": spaces.Box(low=0, high=255, shape=(camera_height, camera_width, 3), dtype=np.uint8),

            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

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

        self.spawn_npcs()



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
        
        # Update model reference if it was passed through DummyVecEnv
        if hasattr(self, 'venv') and hasattr(self.venv, 'envs'):
            self.model = self.venv.envs[0].model
        
        self.previous_speed = 0.0
        self.previous_steering = 0.0
        self.previous_throttle = 0.0
        self.previous_brake = 0.0
        self.last_reward = 0.0

        resources_to_cleanup = []
        

        try:

            for sensor in [self.camera, self.collision_sensor, self.lane_invasion_sensor,

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

                # Optimized LIDAR settings

                lidar_bp.set_attribute('range', '50')

                lidar_bp.set_attribute('rotation_frequency', '10')

                lidar_bp.set_attribute('channels', '16')  # Reduced from 32

                lidar_bp.set_attribute('points_per_second', '20000')  # Reduced from 56000

                lidar_bp.set_attribute('upper_fov', '10.0')

                lidar_bp.set_attribute('lower_fov', '-30.0')

                lidar_bp.set_attribute('horizontal_fov', '100.0')

                lidar_transform = carla.Transform(carla.Location(x=1.5, z=3), carla.Rotation(pitch=-10))

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



            # Default values in case of failure

            lidar_min = float('inf')

            

            # Safely retrieve sensor data

            if hasattr(self, 'lidar_min_distance'):

                lidar_min = self.lidar_min_distance



            # Build state vector (5-dim, reduced from 8-dim)

            state = np.array([

                transform.location.x,

                transform.location.y,

                speed,

                transform.rotation.yaw,

                lidar_min

            ], dtype=np.float32)



            console.log(f"[green][RESET] Agent spawned at {transform.location} using a small vehicle[/green]")
            # Clear the resources_to_cleanup list since we're successful
            resources_to_cleanup = []
            

            return {"image": self.camera_image_obs if self.camera_image_obs is not None else np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8),

                    "state": state}

                    

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

            self.collision_sensor = None

            self.lane_invasion_sensor = None

            self.imu_sensor = None

            self.lidar_sensor = None

            

            # Return a blank observation

            default_state = np.zeros(5, dtype=np.float32)

            return {"image": np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8),

                    "state": default_state}



    def _process_image(self, image):

        array = np.frombuffer(image.raw_data, dtype=np.uint8)

        array = array.reshape((image.height, image.width, 4))  # BGRA format

        # Use cv2.cvtColor to convert from BGRA to RGB

        rgb_image = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)

        resized = cv2.resize(rgb_image, (self.camera_width, self.camera_height))

        self.camera_image_obs = resized.astype(np.uint8)

        # Use the full-resolution RGB image for rendering to maintain consistency.

        if self.visualize:

            self.camera_image = rgb_image

        else:

            self.camera_image = resized



    def _on_collision(self, event):

        self.collision_history = True

        console.log("[red][COLLISION] Collision detected![/red]")



    def _on_lane_invasion(self, event):

        self.lane_invasion_history = True

        console.log("[red][LANE INVASION] Lane invasion detected![/red]")



    def _on_imu_update(self, imu):

        self.imu_data = imu



    def _on_lidar_update(self, lidar_measurement):

        points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)

        points = np.reshape(points, (-1, 4))

        distances = np.linalg.norm(points[:, :3], axis=1)

        # Filter out points that are too close (likely self-detections)

        filtered = distances[distances > 2.0]  # Adjust threshold as needed

        if filtered.size > 0:

            self.lidar_min_distance = np.min(filtered)

        else:

            self.lidar_min_distance = np.min(distances) if distances.size > 0 else float('inf')



    def _send_arduino_data(self, speed, reward, steering, throttle, brake):
        """Send data to Arduino for LCD and LEDs"""
        if self.arduino is None:
            console.log("[red]Arduino connection not established[/red]")
            return

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
        self.previous_steering = steering  # Store the steering value
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



        # --- Stuck Detection ---

        dx = current_location.x - self.previous_location.x

        dy = current_location.y - self.previous_location.y

        distance_moved = np.sqrt(dx**2 + dy**2)

        stuck_speed_threshold = 0.7

        stuck_move_threshold = 0.6

        stuck_max_steps = 50

        if speed < stuck_speed_threshold and distance_moved < stuck_move_threshold:

            self.stuck_counter += 1

            stuck_penalty = -1.0

        else:

            self.stuck_counter = 0

            stuck_penalty = 0.0

        if self.stuck_counter >= stuck_max_steps:

            console.log("[red][STUCK] Vehicle is stuck. Respawning vehicle.[/red]")

            obs = self.reset()

            return obs, -20, True, {"stuck": True}



        # --- Idle Penalty ---

        idle_threshold = 0.5  

        idle_increment = 10  

        if speed < idle_threshold:

            self.idle_penalty += idle_increment

        else:

            self.idle_penalty = 0.0





        state = np.array([

            current_location.x,

            current_location.y,

            speed,

            transform.rotation.yaw,

            self.lidar_min_distance

        ], dtype=np.float32)



 

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

            jerk_penalty = -0.0005 * (abs(jerk) - jerk_threshold)

        else:

            jerk_penalty = 0.0

        self.previous_acceleration = acceleration

        energy_penalty = - (throttle * 0.02 + brake * 0.02)

        energy_bonus = 0.2 if abs(speed - target_speed) < 1.0 and throttle < 0.7 else 0.0

        collision_penalty = -27.0 if self.collision_history else 0.0

        lane_invasion_penalty = -14 if self.lane_invasion_history else 0.0

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

        if self.lidar_min_distance < 2.0:

            lidar_penalty = - (2.0 - self.lidar_min_distance) * 10.0

        else:

            lidar_penalty = 0.0



        reward = (distance_reward + target_speed_reward + log_speed_reward +

                  smooth_steering_penalty*0.3 + acceleration_penalty*0.1 + jerk_penalty*0.1 +

                  energy_penalty*0.001 + energy_bonus + collision_penalty +

                  lane_invasion_penalty + imu_penalty*0.9 + lidar_penalty*0.9 - self.idle_penalty + stuck_penalty) * 2.0



        self.previous_location = current_location

        self.previous_speed = speed



        # Store throttle and brake as instance variables
        self.previous_throttle = throttle  # Store for rendering
        self.previous_brake = brake  # Store for rendering

        obs = {"image": self.camera_image_obs if self.camera_image_obs is not None else np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8),

               "state": state}
        if self.lidar_min_distance < 2.0:
            console.log(f"[blue][STEP] Loc=({current_location.x:.2f},{current_location.y:.2f}), "
                        f"Speed={speed:.2f}, LIDAR_min={self.lidar_min_distance:.2f}, "
                        f"Reward={reward:.2f}[/blue]")
        else:
            console.log(f"[blue][STEP] Loc=({current_location.x:.2f},{current_location.y:.2f}), "
                        f"Speed={speed:.2f}, LIDAR_min={self.lidar_min_distance:.2f}, "
                        f"Reward={reward:.2f}[/blue]")
        if self.visualize:
            self.render()

        self.last_reward = reward  # Update last_reward

        # Define done conditions
        done = False  # Initialize done flag
        
        # Check termination conditions
        if self.collision_history:
            done = True
        
        if self.stuck_counter >= stuck_max_steps:
            console.log("[red][STUCK] Vehicle is stuck. Respawning vehicle.[/red]")
            obs = self.reset()
            return obs, -20, True, {"stuck": True}

        if done:  # Now done is properly defined before being used
            self.episode_count += 1
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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            if self.camera_image is not None:
                try:
                    surface = pygame.surfarray.make_surface(self.camera_image.swapaxes(0, 1))
                    self.display.blit(surface, (0, 0))
                    pygame.display.flip()
                    self.clock.tick(60)  # Limit frame rate to 60 FPS
                except Exception as e:
                    console.log(f"[red]Error during rendering: {e}[/red]")

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
            
            # Calculate and draw current progress
            if hasattr(self.model, 'num_timesteps') and hasattr(self.model, 'total_timesteps_for_entropy'):
                progress = min(1.0, self.model.num_timesteps / self.model.total_timesteps_for_entropy)
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
                timestep_text = f"Steps: {self.model.num_timesteps:,}/{self.model.total_timesteps_for_entropy:,}"
                timestep_width = font.size(timestep_text)[0]
                timestep_x = progress_x + (progress_width - timestep_width) // 2
                
                timestep_shadow = font.render(timestep_text, True, (0, 0, 0))
                self.display.blit(timestep_shadow, (timestep_x + 2, progress_y + progress_height + 5))
                
                timestep_surface = font.render(timestep_text, True, (200, 200, 200))
                self.display.blit(timestep_surface, (timestep_x, progress_y + progress_height + 3))

            pygame.display.update()



    def close(self):

        try:

            for sensor in [self.collision_sensor, self.camera, self.imu_sensor,

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
