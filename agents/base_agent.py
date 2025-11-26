
class BaseBrain:
    def act(self, state):
        raise NotImplementedError

    def learn(self, reward, next_state):
        raise NotImplementedError
