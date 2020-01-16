"""Run the test suite."""

import pathlib
import sys
import unittest

tests_dir = pathlib.Path(__file__).absolute().parent
tests = [f.stem for f in tests_dir.glob('test_*.py') if f.is_file()]

suite = unittest.TestSuite()

for test in tests:
    suite.addTest(unittest.defaultTestLoader.loadTestsFromName(test))

unittest.TextTestRunner().run(suite).wasSuccessful()
