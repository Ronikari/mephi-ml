import random

# 8) Агент, первым ходом всегда выбрасывающий камень, затем случайные элементы   
def rock_first_agent_act(observation, configuration):
    if observation.step == 0:
        return 0
    else:
        return random.randrange(0, configuration.signs)
