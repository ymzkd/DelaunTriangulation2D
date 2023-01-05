import pytest

from geometric_trait2 import Vector


def test_vector_division():
    v1 = Vector(2.0, 3.0) / 2.0
    v2 = Vector(2.0, 3.0) / 2

    vt1 = Vector(1.0, 1.5)

    assert pytest.approx(v1.x) == vt1.x
    assert pytest.approx(v1.y) == vt1.y

    assert pytest.approx(v2.x) == vt1.x
    assert pytest.approx(v2.y) == vt1.y
