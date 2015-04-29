'''Py3k compatibility hacks.'''

__all__ = ['range']

# If we're on Python 2, use range instead of range
if type(range(1)) is list:
  range = xrange
else:
  range = range
