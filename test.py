import unittest

import triangulation as tr

class TestSegment(unittest.TestCase):
    
    def test_side(self):
        v1 = tr.Vertex(0, 0)
        v2 = tr.Vertex(1, 0)
        seg1 = tr.Segment(v1, v2)

        v_a = tr.Vertex(0.5, 0.5)
        self.assertFalse(seg1.ispt_rightside(v_a.point))

        v_b = tr.Vertex(0.5, -0.5)
        self.assertTrue(seg1.ispt_rightside(v_b.point))

    def test_segment_cross(self):
        v1 = tr.Vertex(0, 0)
        v2 = tr.Vertex(1, 0)
        seg1 = tr.Segment(v1, v2)

        v3 = tr.Vertex(0.5, -0.5)
        v4 = tr.Vertex(0.5, 0.5)
        seg2 = tr.Segment(v3, v4)
        self.assertTrue(seg1.is_cross(seg2))

        v5 = tr.Vertex(0.5, 1.5)
        v6 = tr.Vertex(0.5, 5.5)
        seg3 = tr.Segment(v5, v6)
        self.assertFalse(seg1.is_cross(seg3))

if __name__ == "__main__":
    unittest.main()
