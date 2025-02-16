# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import (
    ActorModelCriticPolicy,
    register_policy
)

ModelPolicy = ActorModelCriticPolicy

register_policy("ModelPolicy", ActorModelCriticPolicy) # param name: the policy name； param policy: the policy class
