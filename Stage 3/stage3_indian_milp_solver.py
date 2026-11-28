#!/usr/bin/env python3
"""Compatibility entry point using the canonical checked Stage 3 solver.

This alias prevents the older independent implementation from bypassing
candidate validation and the explicit no-plan failure contract.
"""
import stage3_milp_solver as _canonical
from stage3_milp_solver import *


def __getattr__(name):
    return getattr(_canonical, name)


if __name__ == '__main__':
    raise SystemExit(_canonical.main())
