#!/bin/sh

test_dirs=$(find graphs/ -type d -name tests | xargs)
nosetests --with-cov --cov-report html --cov=graphs/ $test_dirs \
  && coverage report

