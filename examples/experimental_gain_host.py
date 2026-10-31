"""Unadopted Python candidate on the same six-step connection example.

Keep stable_host.py beside this file. This checks the API, not long-world quality.
The experiment needs the matching development package, not the v0.1.0 release.
"""
import stable_host as host
from reflex.experimental_gain import GainHorizonOwner,transfer_event_reader


def make_controller():
    context=host.observe(host.initial_world())
    reader=transfer_event_reader(context['scope'],('a-usual','b-usual'),host.models,host.outcome,
        legal_for=lambda c:host.RESPONSES,model_id=host.RULES,known_conditionals=True)
    return host.make_controller(reader,GainHorizonOwner)


if __name__=='__main__':host.main(make_controller)
