from agent import Agent
import numpy as np

# 13) Агент, обучающийся с помощью алгоритма SARSA
class SarsaAgent(Agent):
    def __init__(self):
        super().__init__()

    def _learn(self, reward, state):
        # В соответствии с формулой SARSA обновляем значения в Q-таблице
        self.Q[self.state][self.action] = self.alpha * (reward + self.gamma * self.Q[state][self.action] \
            - self.Q[self.state][self.previous_action])

# Инициализируем экземпляр класса агента и метод его действия
agent = SarsaAgent()
def act(observation, configuration):
    # Обновление состояния агента с последующим обучением инициализируем после первого хода
    if observation.step > 0:
        agent.fit(observation.lastOpponentAction)
    return agent.act()
