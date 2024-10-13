import random
import math

# 10) Агент, выбрасывающий элемент, побеждающий предыдущий ход оппонента
class LastReactWinAgent():
    def __init__(self):
        self.last_react_action = None

    # Метод для вычисления награды за ход (для агентов, анализирующих действия оппонента без обучения)
    def get_score(self, agent_action, opponent_action):
        # Вычислить разницу между действиями оппонента и агента для определения очков
        delta = (
            opponent_action - agent_action
            if (agent_action + opponent_action) % 2 == 0
            else agent_action - opponent_action
        )
        return 0 if delta == 0 else math.copysign(1, delta)

# Инициализируем экземпляр класса агента и метод его действия
agent = LastReactWinAgent()
def act(observation, configuration):
    # Первый ход совершается случайным образом
    if observation.step == 0:
        agent.last_react_action = random.randrange(0, configuration.signs)
    # Каждый последующий ход агент выбрасывает значение, побеждающее предыдущее действие оппонента
    elif agent.get_score(agent.last_react_action, observation.lastOpponentAction) != 1:
        agent.last_react_action = (agent.last_react_action + 2) % configuration.signs

    return agent.last_react_action
