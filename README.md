# Path-following for DRL Control

This repo implements a 6-DOF simulation model for an AUV for reinforcement learning control. The algorithm is constructed based on the stable baselines (OpenAI) interface. The environment contains a 3D path and an ocean current disturbance. The goal for the agents is to steer the AUV on-path while combating disturbances. 

## Getting Started

To install all packages needed in your virtual environment, run:

```
pip install -r requirements.txt
```
 
### Training an agent:

All hyperparameters and setup can be tuned in the file [train.py](https://github.com/simentha/gym-auv/blob/master/train3d.py) and [__init__.py](https://github.com/simentha/gym-auv/blob/master/gym_auv/__init__.py).

For training an agent, run:

```
python train3d.py --exp_id [x]
```

Where x is the experiment id number. 


## Running an agent in the environment

For running an agent in any scenario, use:

```
python run3d.py --exp_id [x] --scenario [scenario] --controller_scenario [controller_scenario] --controller [y]
```

Where x is the experiment id number, scenario is what scenario to run, controller_scenario is which scenario the controller was trained in and y is
which agent number to run. If no y is provided, the agent called "best_model" is chosen. Scenarios can be either of "beginner", "intermediate", 
"test_path", "test_path_current" (Path following with disturbance). 


