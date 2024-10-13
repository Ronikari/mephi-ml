import numpy as np
import math

# Базовый класс, от которого наследуются последующие классы-агенты с алгоритмами обучения
class Agent:
    def __init__(self):
       # Инициализация параметров обучения агента
       self.alpha = 0.1 # коэффициент обучения
       self.gamma = 0.9 # коэффициент дисконтирования
       self.epsilon = 0.1 # параметр исследования

       # Инициализация начального состояния и действия агента
       self.state = np.random.randint(0, 3) # 0 - бросок камня, 1 - бросок бумаги, 2 - бросок ножниц
       self.action = np.random.randint(0, 3)
       self.previous_action = None

       # Инициализация Q-таблицы
       # В качестве условия примем, что начальные вероятности равны
       self.Q = np.array([[1/3, 1/3, 1/3], 
                          [1/3, 1/3, 1/3], 
                          [1/3, 1/3, 1/3]])
    
    # Метод для выбора действия агента
    def act(self):
        # Реализация политики агента
        self.previous_action = self.action # сохранить предыдущее действие
        self.action = self._choose_action()
        return int(self.action)
    
    # Метод для обновления текущего состояния агента на основании полученной награды за ход
    def fit(self, state):
        # Вычислить разницу между текущим действием и состоянием среды для определения награды
        delta = (
            state - self.action
            if (self.action + state) % 2 == 0
            else self.action - state
        )
        reward = 0 if delta == 0 else math.copysign(1, delta)

        # Обновить значения на основании полученной награды
        self._learn(reward, state)
        # Обновить текущее состояние агента
        self.state = state
    
    # Метод для обучения агента
    # Переопределяется дочерними классами-агентами с алгоритмами обучения
    def _learn(self, reward, state):
        pass
    
    # Реализация жадного алгоритма, определяющего политику агента
    def _choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 3) # выбираем случайное действие
        else:
            return np.argmax(self.Q[self.state]) # выбираем максимальное значение в Q-таблице, соответствующее текущему состоянию
