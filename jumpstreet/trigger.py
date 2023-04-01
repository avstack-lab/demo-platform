import numpy as np

from jumpstreet.utils import BaseClass


class Trigger(BaseClass):
    NAME = "trigger"

    def __init__(self, identifier) -> None:
        super().__init__(self.NAME, identifier)

    def trigger(self, tracks):
        raise NotImplementedError


class AlwaysTrigger(Trigger):
    def __init__(self, identifier) -> None:
        super().__init__(identifier)

    def trigger(self, *args, **kwargs):
        """Return infinite t_start/end bounds"""
        return -np.inf, np.inf
