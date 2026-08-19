"""Minimal stand-in for the organiser's pirate_client.

Only provides what strategy.py needs: a base class with initialize()/act().
The real framework is not public, so this reproduces the interface only.
"""


class Strategy:
    def __init__(self, player_id=0):
        self.player_id = player_id
        self.initialize(player_id)

    def initialize(self, player_id):
        self.player_id = player_id

    def act(self, state):
        raise NotImplementedError
