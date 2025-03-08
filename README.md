# Carla-Self-Driving-Car
This project trains a self-driving car in CARLA using SAC reinforcement learning. It features a custom CNN with residual blocks and attention, reward shaping (speed, collision, smoothness), Arduino real-time feedback, hyper parameter tuning, check-pointing, and logging.


This project is about training a simulated self-driving car using reinforcement learning. Here’s a simple breakdown:

  1)Simulated Driving: It uses the CARLA simulator to create a realistic driving environment complete with traffic, weather changes, and various obstacles.

  2)Learning to Drive: The car learns to drive using a method called Soft Actor-Critic (SAC), which is a popular reinforcement learning algorithm.

  3)Custom Features: The system processes camera images and other state information using a custom neural network. This network uses convolutional layers with residual blocks and an attention mechanism to better understand what’s happening     on the road.

  4)Reward System: The car receives rewards or penalties based on factors like distance traveled, maintaining a target speed, avoiding collisions, and smooth driving.

  5)Real-time Feedback: The project also integrates with an Arduino board to send real-time data (like speed and rewards) to external displays.

  6)Training Enhancements: It includes features like hyperparameter tuning (using optuna), checkpointing (to save progress), and detailed logging to track how the training is going.

------------------------------------------------------------------------------------------------------------------------------------------------------

Videos showing the project:

https://www.youtube.com/watch?v=uS3r_8r4riY&t=10s

https://www.youtube.com/watch?v=lIOpiagK0PU

------------------------------------------------------------------------------------------------------------------------------------------------------

For beginners to use the project - 

1) download all the files and remember the folder name where you have the checkpoint 

2) Download Carla-0.10.0 and run it (REMEMBER TO DISABLE RAY TRACING FOR BETTER PERFORMANCE! THROUGH THE CONFIG)

3) In the folder having train_sac_carla.py and carla_env.py. type cmd in the folder's address and paste 
python train_sac_carla.py --resume "YOUR_CHECKPOINT_ADDRESS"\sac_carla_160000_steps.zip

if you want to train it from starting just remove --resume 

#### Run hyperparameter optimization before training:

python train_sac_carla.py --optimize


Make sure you're in the correct directory where the script is located when running these commands.

------------------------------------------------------------------------------------------------------------------------------------------------------

Features for futures release: (for contributors <3)

1)If possible add a LIDAR visualization

2) making the reward system better and better training features

3) More training matrices for visualization


