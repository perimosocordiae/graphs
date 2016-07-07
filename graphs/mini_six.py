'''Py3k compatibility hacks.'''

__all__ = ['range', 'zip', 'zip_longest']

# If we're on Python 2, use xrange instead of range, etc
if type(range(1)) is list:
  range = xrange
  from itertools import izip_longest as zip_longest, izip as zip
else:
  range = range
  zip = zip
  from itertools import zip_longest
