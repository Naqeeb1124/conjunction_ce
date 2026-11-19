"""conjce: coordinator recommendations for satellite conjunctions.

Modules
-------
pc          2D (short-encounter) collision probability in the encounter plane
maneuver    tangential-burn to miss-distance mapping (Clohessy-Wiltshire)
model       the baseline two-operator conjunction and its cost model
games       Nash equilibria, correlated-equilibrium LPs with an obedience margin, public recommendations
inference   operators who learn the recommendation rule from k past events
"""

__version__ = "0.1.0"
