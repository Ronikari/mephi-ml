from agent import Agent
import numpy as np

# 12) Агент, обучающийся с помощью алгоритма Q-learning
class QLearningAgent(Agent):
    def __init__(self):
        super().__init__()

    def _learn(self, reward, state):
        # В соответствии с уравнением Беллмана обновляем значения в Q-таблице
        self.Q[self.state][self.action] = (1 - self.alpha) * self.Q[self.state][self.previous_action] \
            + self.alpha * (reward + self.gamma * np.max(self.Q[state]) - self.Q[self.state][self.previous_action])
    
# Инициализируем экземпляр класса агента и метод его действия
agent = QLearningAgent()
def act(observation, configuration):
    # Обновление состояния агента с последующим обучением инициализируем после первого хода
    if observation.step > 0:
        agent.fit(observation.lastOpponentAction)
    return agent.act()
