from gym.envs.registration import register

register(
    id='one-stage-v0',
    entry_point='ligo_environment.envs:OneStage',
)
register(
    id='one-stage-v1',
    entry_point='ligo_environment.envs:OneStage1',
)
register(
    id='two-stage-v0',
    entry_point='ligo_environment.envs:TwoStage',
)