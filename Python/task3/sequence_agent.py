# 4) Агент, выбрасывающий элементы в прямой последовательности (0, 1, 2)
class SequenceAgent:
    def __init__(self):
        self.action = 0 # начальное состояние

# Инициализируем экземпляр класса агента и метод его действия
agent = SequenceAgent()
def act(observation, configuration):
    if agent.action == 0:
        agent.action = 1
    elif agent.action == 1:
        agent.action = 2
    elif agent.action == 2:
        agent.action = 0
    return agent.action
