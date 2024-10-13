import random

# 7) Агент, всегда выбрасывающий случайный элемент
def random_agent_act(observation, configuration):
    return random.randrange(0, configuration.signs)
