import random

# 11) Агент, анализирующий статистику ходов оппонента
class StatisticalAgent():
    def __init__(self):
        self.action_histogram = {}

# Инициализируем экземпляр класса агента и метод его действия
agent = StatisticalAgent()
def act(observation, configuration):
    if observation.step == 0:
        agent.action_histogram = {}
        return

    # Агент реагирует на каждое предыдущее действие оппонента и записывает его в словарь action_histogram,
    # ключами которого являются действия оппонента, а значениями - количество тех или иных действий
    action = observation.lastOpponentAction
    if action not in agent.action_histogram:
        agent.action_histogram[action] = 0
    agent.action_histogram[action] += 1

    mode_action = None # действие оппонента, совершённое им наибольшее число раз
    mode_action_count = None # наибольшее количество действий оппонента

    for k, v in agent.action_histogram.items():
        if mode_action_count is None or v > mode_action_count:
            mode_action = k
            mode_action_count = v

    return (mode_action + 1) % configuration.signs
