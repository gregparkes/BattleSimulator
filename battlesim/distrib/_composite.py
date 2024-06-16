from ._distrib2 import Sampling


class Composite:
    """Holds an army set information."""

    def __init__(
        self,
        name: str,
        n: int,
        pos_dist=None,
        init_ai="nearest",
        rolling_ai="nearest",
        decision_ai="aggressive",
    ):
        self.name = name
        self.n = n
        if pos_dist is not None:
            self.pos = pos_dist
        else:
            self.pos = Sampling("normal")

        self.init_ai = init_ai
        self.rolling_ai = rolling_ai
        self.decision_ai = decision_ai

    def __repr__(self):
        return (
            f"Composite('{self.name}', n={self.n}, pos={self.pos}, "
            + f"init_ai='{self.init_ai}', rolling_ai='{self.rolling_ai}', decision_ai='{self.decision_ai}')"
        )
