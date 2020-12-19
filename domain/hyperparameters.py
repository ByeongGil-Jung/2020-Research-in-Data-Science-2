from domain.base import Domain


class Hyperparameters(Domain):

    def __init__(self, **kwargs):
        super(Hyperparameters, self).__init__()
        self.__dict__.update(kwargs)

    def __repr__(self):
        return self.__dict__.__repr__()
