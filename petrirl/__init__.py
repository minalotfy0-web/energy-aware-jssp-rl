from gymnasium.envs.registration import register


register(
    id="petrirl-moo-v0",
    entry_point="petrirl.envs.moo.gym_env:MooEnv",
    )
