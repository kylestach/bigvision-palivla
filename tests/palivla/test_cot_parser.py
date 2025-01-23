import unittest

import numpy as np

from palivla.cot_parser import State, parse_cot_string


class TestCoTParser(unittest.TestCase):
    def test_single_gripper_and_object(self):
        """Test parsing a simple string with one gripper position and one object."""
        test_string = "gripper<loc0264><loc0360>;silver tray<loc0788><loc0400><loc1020><loc0816>;<begin_of_action><act487><act495><eos>"
        result = parse_cot_string(test_string)

        # Check gripper state
        self.assertEqual(len(result.gripper), 1)
        gripper_state = result.gripper[0]
        self.assertIsInstance(gripper_state, State)
        self.assertEqual(gripper_state.x, 360.0)
        self.assertEqual(gripper_state.y, 264.0)
        self.assertIsNone(gripper_state.width)
        self.assertIsNone(gripper_state.height)

        # Check gripper array
        np.testing.assert_array_almost_equal(
            result.get_gripper_array(), np.array([[360.0, 264.0]], dtype=np.float32)
        )

        # Check object state
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(len(result.objects["silver tray"]), 1)
        obj_state = result.objects["silver tray"][0]
        self.assertIsInstance(obj_state, State)
        self.assertEqual(obj_state.x, 400.0)
        self.assertEqual(obj_state.y, 788.0)
        self.assertEqual(obj_state.width, 416.0)  # 816 - 400
        self.assertEqual(obj_state.height, 232.0)  # 1020 - 788

        self.assertEqual(result.action, ["<act487>", "<act495>"])

    def test_trajectory_gripper_and_object(self):
        """Test parsing a trajectory with multiple gripper and object positions."""
        test_string = (
            "gripper<loc0264><loc0360>;silver tray<loc0788><loc0400><loc1020><loc0816>;"
            "gripper<loc0300><loc0428>;silver tray<loc0788><loc0500><loc1020><loc0916>;"
            "<begin_of_action><act487><act495><act477><eos>"
        )
        result = parse_cot_string(test_string)

        # Check gripper trajectory
        self.assertEqual(len(result.gripper), 2)
        np.testing.assert_array_almost_equal(
            result.get_gripper_array(),
            np.array(
                [
                    [360.0, 264.0],
                    [428.0, 300.0],
                ],
                dtype=np.float32,
            ),
        )

        # Check object trajectory
        self.assertEqual(len(result.objects["silver tray"]), 2)
        tray_states = result.objects["silver tray"]

        self.assertEqual(tray_states[0].x, 400.0)
        self.assertEqual(tray_states[0].y, 788.0)
        self.assertEqual(tray_states[0].width, 416.0)  # 816 - 400
        self.assertEqual(tray_states[0].height, 232.0)  # 1020 - 788

        self.assertEqual(tray_states[1].x, 500.0)
        self.assertEqual(tray_states[1].y, 788.0)
        self.assertEqual(tray_states[1].width, 416.0)  # 916 - 500
        self.assertEqual(tray_states[1].height, 232.0)  # 1020 - 788

        # Test get_object_array method
        np.testing.assert_array_almost_equal(
            result.get_object_array("silver tray"),
            np.array(
                [
                    [400.0, 788.0, 416.0, 232.0],  # width = 816-400, height = 1020-788
                    [500.0, 788.0, 416.0, 232.0],  # width = 916-500, height = 1020-788
                ],
                dtype=np.float32,
            ),
        )

        # Check actions
        self.assertEqual(result.action, ["<act487>", "<act495>", "<act477>"])

        # Check num_timesteps property
        self.assertEqual(result.num_timesteps, 2)

    def test_multiple_objects(self):
        """Test parsing a string with multiple different objects."""
        test_string = (
            "gripper<loc0264><loc0360>;silver tray<loc0788><loc0400><loc1020><loc0816>;"
            "plate<loc0100><loc0200><loc0300><loc0400>;"
            "<begin_of_action><act487><eos>"
        )
        result = parse_cot_string(test_string)

        self.assertEqual(len(result.objects), 2)
        self.assertIn("silver tray", result.objects)
        self.assertIn("plate", result.objects)

        plate_state = result.objects["plate"][0]
        self.assertEqual(plate_state.x, 200.0)
        self.assertEqual(plate_state.y, 100.0)
        self.assertEqual(plate_state.width, 200.0)  # 400 - 200
        self.assertEqual(plate_state.height, 200.0)  # 300 - 100

    def test_empty_components(self):
        """Test handling of empty or malformed components."""
        test_string = "gripper<loc0264><loc0360>;;<begin_of_action><act487><eos>"
        result = parse_cot_string(test_string)

        self.assertEqual(len(result.gripper), 1)
        np.testing.assert_array_almost_equal(
            result.get_gripper_array(), np.array([[360.0, 264.0]], dtype=np.float32)
        )
        self.assertEqual(result.objects, {})
        self.assertEqual(result.action, ["<act487>"])

    def test_no_actions(self):
        """Test handling string without action tokens."""
        test_string = (
            "gripper<loc0264><loc0360>;silver tray<loc0788><loc0400><loc1020><loc0816>"
        )
        result = parse_cot_string(test_string)

        self.assertEqual(len(result.gripper), 1)
        np.testing.assert_array_almost_equal(
            result.get_gripper_array(), np.array([[360.0, 264.0]], dtype=np.float32)
        )
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.action, [])

    def test_empty_gripper(self):
        """Test handling string with empty gripper component."""
        test_string = "gripper;<begin_of_action><act0><eos>"
        result = parse_cot_string(test_string)

        self.assertEqual(len(result.gripper), 0)
        self.assertEqual(len(result.objects), 0)
        self.assertEqual(result.action, ["<act0>"])

    def test_no_reasoning(self):
        """Test handling string without reasoning."""
        test_string = "<begin_of_action><act487><eos>"
        result = parse_cot_string(test_string)

        self.assertEqual(len(result.gripper), 0)
        self.assertEqual(len(result.objects), 0)
        self.assertEqual(result.action, ["<act487>"])

if __name__ == "__main__":
    unittest.main()
