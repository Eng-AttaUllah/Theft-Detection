"""Shop theft detection package.

The package watches a shop camera/video and raises alerts for behaviour that
is typically associated with shoplifting:

* entering restricted areas (counter, stock room)
* loitering / standing too long in front of a shelf
* items disappearing from a shelf while a person is next to it
* objects left behind by a person
* sudden, fast movement (snatching)
* groups of people gathering (distraction theft)
* camera tampering (covered, frozen, out of focus)
"""

__version__ = "1.0.0"

from .config import Config  # noqa: F401
from .pipeline import Pipeline  # noqa: F401
