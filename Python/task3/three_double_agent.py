# 6) Агент, повторяющий свой ход 3 раза подряд в прямой последовательности (0, 1, 2)
class ThreeDoubleAgent:
    def __init__(self):
        self.action = 0 # начальное состояние

# Инициализируем экземпляр класса агента и метод его действия
agent = ThreeDoubleAgent()
def act(observation, configuration):
    if observation.step > 0 and observation.step % 3 == 0:
        if agent.action == 0:
            agent.action = 1
        elif agent.action == 1:
            agent.action = 2
        elif agent.action == 2:
            agent.action = 0
    return agent.action
