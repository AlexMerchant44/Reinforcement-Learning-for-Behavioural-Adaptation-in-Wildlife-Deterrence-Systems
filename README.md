# Reinforcement Learning for Behavioural Adaptation in Wildlife Deterrence Systems

This project investigates the use of reinforcement learning (RL) to adaptively control a physical bird deterrent system.  
The goal is to learn the minimal energy action required to deter target species (crows and magpies) while avoiding unnecessary disturbance and adapting with bird habituation.

The system supports two interchangeable RL control modes, selectable via configuration files:

- Discrete control (Q-learning)
- Continuous control (policy gradient / REINFORCE)

Ran by `python3 main --config config/continuous.yaml`

---

## Dataset Collection & Model Training

- A trail camera was mounted facing a bird feeder for two weeks.
- ~1500 images were collected containing various combinations of crows, magpies, and empty scenes.
- Images were manually labelled and passed through a pre-trained Yolo11n model to localise birds.
- YOLO crops were fed into a custom RenNet18 Image Classifier, trained to identify:
  - `Crow`
  - `Magpie`
  - `None`

Training results (epoch 4):

Train loss: 0.0028 Train accuracy: 1.0000
Val loss: 0.0341 Val accuracy: 0.9921


---

## Hardware Setup

- Raspberry Pi 5  
- Pi Camera Module 3 (Entaniya waterproof housing)  
- IRLZ44N MOSFET driver circuit  
- 5V brushed motor mounted beside the feeder  
- Reinforcement learning controller running continuously between specified hours  

---

## System Pipeline

During operation:

1. **Image capture**  
   The Pi camera captures a frame at fixed intervals.

2. **Detection + Classification**  
   - YOLO11n detects whether a bird is present.
   - If a bird is detected, the cropped region is passed into the ResNet18 classifier.

3. **State construction**  
   A state is defined by `(species_detected, mode)` where `species_detected` ∈ {Crow, Magpie, None}
   `mode` is read from `mode.txt` and mapped via `STATE_TABLE`.

4. **Action selection**  
   Action selection depends on the configured RL mode:

   Discrete Mode:

   - ε-greedy selection over a fixed action set 
   - Actions correspond to predefined (PWM duty, duration) pairs

   Continuous Mode:

   - Continuous sampling of:
      - motor duty cycle
      - motor duration
   - Actions are drawn from state-conditioned Beta distributions
   - Parameters are updated via policy gradient (REINFORCE)

5. **Environmental feedback**  
   - The system waits 5 seconds.
   - A second image is captured and classified to determine whether the bird remained or left.

6. **Reward Function Design**  
   Defined in `reward.py`:
   - The total reward is computed as the sum of four components:
      reward = r_power + r_success + r_false + r_presence

   Energy Penalty:
   - power_cost = duty x duration
   - r_power = -energy penalty * power_cost

   Success Bonus:
   - r_success = clear_bonus  (if success)

   False Positive Penalty: (No bird present)
   - r_false = -false_positive_penalty × power_cost

   False Negative Penalty: (Bird Remains)
   - r_presence = −false_negative_penalty × conf_after


7. **Learning update** 
   Discrete (Q-Learning):

   - The relevant entry in `q_table.npy` is updated via:
      
      Q[s][a] ← Q[s][a] + α (reward − Q[s][a])

   Continuous (Policy Gradient):

   - REINFORCE update with per-state baseline
   - Beta distribution parameters updated via:

      θ ← θ + η (reward - baseline) Δθ log πθ(action)


---

## Expected Behaviour & Hypothesis

- At the beginning, the average energy per event should decrease, as the agent learns to deter birds using minimal necessary energy.
- Over time, birds are expected to habituate to the deterrent.  
Thus, average required energy may rise again, showing adaptation at both ends:
- The RL system adapts to the birds  
- The birds adapt to the deterrent  

This dynamic interaction is a key focus of the investigation.

---

## Data Logging

All events are logged locally on the Pi:

- Timestamped before/after images
- Detected species
- State transitions
- Action taken
- Reward
- Learning updates

Logs are saved to `data/` including:
- `history.csv`
- Episode image pairs
- `q_table.npy` or `beta_params.npz`, `baseline.npy`

---

## Results

The system is currently running autonomously.  
Once enough event data has been collected, the following will be added:

- Plots of average energy per event over time  
- Learning evolution visualisations  
- Behavioural adaptation analysis  

Results will be published in a dedicated `results/` folder.

---
