import random

# 9) Агент, копирующий предыдущий ход оппонента
def copy_opponent_agent_act(observation, configuration):
    # Если это не первый ход, выбросить значение предыдущего хода оппонента
    if observation.step > 0:
        return observation.lastOpponentAction
    else:
        # Если это первый ход, выбросить случайный элемент
        return random.randrange(0, configuration.signs)
