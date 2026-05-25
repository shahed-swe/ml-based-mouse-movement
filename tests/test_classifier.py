import pytest

from handgester.classifier import Gesture, RuleBasedClassifier
from tests.fixtures.sample_landmarks import (
    FIST_RIGHT,
    OK_RIGHT,
    OPEN_PALM_RIGHT,
    PEACE_RIGHT,
    POINTING_RIGHT,
    THUMBS_UP_RIGHT,
)


@pytest.fixture
def clf():
    return RuleBasedClassifier()


def test_open_palm(clf):
    assert clf.classify(OPEN_PALM_RIGHT) == Gesture.OPEN_PALM


def test_fist(clf):
    assert clf.classify(FIST_RIGHT) == Gesture.FIST


def test_thumbs_up(clf):
    assert clf.classify(THUMBS_UP_RIGHT) == Gesture.THUMBS_UP


def test_peace(clf):
    assert clf.classify(PEACE_RIGHT) == Gesture.PEACE


def test_pointing(clf):
    assert clf.classify(POINTING_RIGHT) == Gesture.POINTING


def test_ok(clf):
    assert clf.classify(OK_RIGHT) == Gesture.OK
