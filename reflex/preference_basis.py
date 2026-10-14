"""Explicit meanings for preference inputs when terminal purpose is available.

An adapter may declare a WHOLE channel to be an instrumental estimate, or an
estimate of the declared purpose. Unlisted channels are intrinsic and unchanged.
This is not automatic psychology inference or weight calibration. A channel
mixing intrinsic enjoyment and instrumental advice must be separated upstream.
"""
from dataclasses import dataclass, asdict
import copy
from .core import NEEDS, VALUES, TRAITS, digest, number, identifier


CHANNELS = {f'{group}.{name}' for group, names in
            (('needs', NEEDS), ('values', VALUES), ('style', TRAITS)) for name in names}


@dataclass(frozen=True)
class PreferenceBasis:
    target: str
    instrumental: tuple = ()
    purpose: tuple = ()

    def __post_init__(self):
        identifier(self.target)
        if not isinstance(self.instrumental, tuple) or not isinstance(self.purpose, tuple):
            raise ValueError('immutable explicit channel declarations required')
        keys = list(self.instrumental)
        for row in self.purpose:
            if not isinstance(row, tuple) or len(row) != 2:
                raise ValueError('purpose channel and fixed unit conversion required')
            key, scale = row
            number(scale, -1., 1.)
            keys.append(key)
        if any(k not in CHANNELS for k in keys) or len(set(keys)) != len(keys):
            raise ValueError('distinct known preference channels required')

    @property
    def identity(self):
        return digest(['preference-basis-v1', asdict(self)])

    def resolve(self, flow, value, *, target, complete):
        """Replace estimates only with full terminal-purpose model coverage.

        complete describes the model horizon, NOT correctness of its law. Cost,
        probability and every unlisted channel retain exactly their old values.
        The caller still transports objective separately, in its existing units.
        """
        if target != self.target or complete is not True:
            raise ValueError('same purpose and complete modeled terminal coverage required')
        number(value, -1., 1.)
        out = copy.deepcopy(flow)
        for key in self.instrumental:
            group, name = key.split('.')
            out[group].pop(name, None)
        for key, scale in self.purpose:
            group, name = key.split('.')
            out[group][name] = scale * value
        return out
