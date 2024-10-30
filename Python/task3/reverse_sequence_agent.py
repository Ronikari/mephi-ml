# 5) Агент, выбрасывающий элементы в обратной последовательности (2, 1, 0)
class ReverseSequenceAgent:
    def __init__(self):
        self.action = 2 # начальное состояние

# Инициализируем экземпляр класса агента и метод его действия
agent = ReverseSequenceAgent()
def act(observation, configuration):
    if agent.action == 2:
        agent.action = 1
    elif agent.action == 1:
        agent.action = 0
    elif agent.action == 0:
        agent.action = 2
    return agent.action
