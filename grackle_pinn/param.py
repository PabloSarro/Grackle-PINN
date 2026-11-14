"""State parameterisations (A4).

All data live as natural-log states `ls = ln [T, nHI, nHII]` (shape (..., 3)). A parameterisation defines
  - the network features and the one-step targets built from a pair (ls_t, ls_t+1),
  - how the raw network output becomes a physical increment and how the state is advanced,
so that training, rollouts and metrics work for any of them.

`report`       the report's choice: features = ln state, output = 3 standardised log-increments, mass
               conservation only through a soft penalty.
`constrained`  features z = (ln e, logit x, ln nH) with nH = nHI + nHII, x = nHII / nH, e ~ T (1 + x)
               (specific internal energy up to a constant). The network outputs two numbers:
                 d ln e     = -softplus(o1)           (<= 0: energy never increases)
                 d logit x  = L tanh(o2)              (bounded)
               and ln nH is carried unchanged. Then nHI + nHII = nH holds to round-off, x stays in (0, 1)
               and the energy is monotone for every network output.

`lograte`      (A5) the same constrained state and hard guarantees, but the network predicts the increments in a
               space where the flat regime is resolved:
                 u1 = ln(-d ln e)                  (the energy decrement is a positive rate; ln spans 1e-10..2)
                 u2 = asinh(d logit x / s)         (signed log; linear below s, logarithmic above)
               The data loss is a plain MSE on the standardised (u1, u2): a flat-regime increment of 1e-9 and a
               cliff increment of 1 then carry comparable weight. The physical increment is
                 d ln e = -exp(clip(u1)),   d logit x = s sinh(clip(u2)).
               The clips (lo, hi), taken from the training targets with a margin, only matter for networks that
               leave the training range; they keep the update finite and the guarantees valid.
"""
import torch
import torch.nn.functional as F

EPS = 1e-8


class Report:
    name = "report"
    in_dim = 3
    out_dim = 3

    def config(self):
        return {"name": self.name}

    def features(self, ls):
        return ls

    def targets(self, ls, ls_next):
        return ls_next - ls

    def init_model(self, model):
        pass

    def to_standardised(self, out, sc):
        """Network output -> standardised increments (what the data loss compares with the standardised target)."""
        return out

    def to_physical(self, out, sc):
        return out * (sc["tg_std"] + EPS) + sc["tg_mean"]

    def advance(self, ls, increment):
        return ls + increment

    def step(self, model, sc, ls):
        x = (self.features(ls) - sc["in_mean"]) / (sc["in_std"] + EPS)
        return self.advance(ls, self.to_physical(model(x), sc))


class Constrained:
    name = "constrained"
    in_dim = 3
    out_dim = 2

    def __init__(self, lmax=25.0, e_bias=-12.0):
        self.lmax = float(lmax)
        self.e_bias = float(e_bias)

    def config(self):
        return {"name": self.name, "lmax": self.lmax, "e_bias": self.e_bias}

    # --- state conversions
    @staticmethod
    def to_z(ls):
        lT, lI, lII = ls[..., 0], ls[..., 1], ls[..., 2]
        logit = lII - lI
        lnH = torch.logaddexp(lI, lII)
        lne = lT + torch.log1p(torch.sigmoid(logit))
        return torch.stack([lne, logit, lnH], dim=-1)

    @staticmethod
    def from_z(z):
        lne, logit, lnH = z[..., 0], z[..., 1], z[..., 2]
        x = torch.sigmoid(logit)
        lT = lne - torch.log1p(x)
        lII = lnH + F.logsigmoid(logit)
        lI = lnH + F.logsigmoid(-logit)
        return torch.stack([lT, lI, lII], dim=-1)

    # --- interface
    def features(self, ls):
        return self.to_z(ls)

    def targets(self, ls, ls_next):
        d = self.to_z(ls_next) - self.to_z(ls)
        return d[..., :2]

    def init_model(self, model):
        """Start with small energy decrements: softplus(e_bias) ~ 6e-6 instead of 0.69."""
        with torch.no_grad():
            model.net[-1].bias[0] = self.e_bias

    def to_physical(self, out, sc):
        return torch.stack([-F.softplus(out[..., 0]), self.lmax * torch.tanh(out[..., 1])], dim=-1)

    def to_standardised(self, out, sc):
        return (self.to_physical(out, sc) - sc["tg_mean"]) / (sc["tg_std"] + EPS)

    def advance(self, ls, increment):
        z = self.to_z(ls)
        z = torch.stack([z[..., 0] + increment[..., 0], z[..., 1] + increment[..., 1], z[..., 2]], dim=-1)
        return self.from_z(z)

    def step(self, model, sc, ls):
        x = (self.features(ls) - sc["in_mean"]) / (sc["in_std"] + EPS)
        return self.advance(ls, self.to_physical(model(x), sc))


class LogRate(Constrained):
    name = "lograte"
    FLOOR = 1e-14

    def __init__(self, s=1e-10, lo=(-33.0, -30.0), hi=(1.5, 30.0)):
        self.s = float(s)
        self.lo = [float(v) for v in lo]
        self.hi = [float(v) for v in hi]

    def config(self):
        return {"name": self.name, "s": self.s, "lo": self.lo, "hi": self.hi}

    def targets(self, ls, ls_next):
        d = self.to_z(ls_next) - self.to_z(ls)
        return torch.stack([torch.log((-d[..., 0]).clamp(min=self.FLOOR)), torch.asinh(d[..., 1] / self.s)], dim=-1)

    def init_model(self, model):
        pass

    def to_standardised(self, out, sc):
        return out

    def to_physical(self, out, sc):
        u = out * (sc["tg_std"] + EPS) + sc["tg_mean"]
        lo = torch.as_tensor(self.lo, dtype=u.dtype, device=u.device)
        hi = torch.as_tensor(self.hi, dtype=u.dtype, device=u.device)
        u = torch.minimum(torch.maximum(u, lo), hi)
        return torch.stack([-torch.exp(u[..., 0]), self.s * torch.sinh(u[..., 1])], dim=-1)


def build(cfg):
    """cfg: {"name": "report"} or {"name": "constrained", "lmax": ...}."""
    cfg = dict(cfg or {"name": "report"})
    name = cfg.pop("name")
    if name == "report":
        return Report()
    if name == "constrained":
        return Constrained(**cfg)
    if name == "lograte":
        return LogRate(**cfg)
    raise ValueError(f"unknown parameterisation {name}")
