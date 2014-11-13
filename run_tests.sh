#!/bin/sh

nosetests --with-cov --cov-report html --cov=graphs/ \
  graphs/tests/ graphs/base/tests/ graphs/construction/tests/  \
  graphs/generators/tests/ && coverage report

