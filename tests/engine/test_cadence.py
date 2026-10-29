import numpy as np
from lsdub_engine.audits import cadence
from lsdub_engine.core import retime


def test_hold_flagged_and_still_shot_not():
    sp = np.r_[np.full(10, 2.0), [0.4, 0.5, 0.4], np.full(10, 2.0)]
    ev = cadence.find(sp)
    assert [(e['kind'], e['a'], e['b']) for e in ev] == [('HOLD', 10, 13)]
    assert cadence.find(np.full(20, 0.1)) == []                       # a still shot holding still is no hitch


def test_skip_flagged_and_end_terminates():
    sp = np.r_[np.full(8, 1.0), [5.0], np.full(8, 1.0)]
    assert [(e['kind'], e['a']) for e in cadence.find(sp)] == [('SKIP', 8)]
    cadence.find(np.r_[np.full(5, 2.0), [0.1]])                         # hold candidate at the very end: no hang


def test_windows_merge_and_pad():
    ev = [dict(kind='HOLD', a=10, b=12), dict(kind='HOLD', a=14, b=15)]
    assert retime.windows_for(ev, 100, pad=3) == [(7, 18)]
