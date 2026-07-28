import unittest
import os
import pygame
from genetic_algorithm import generate_devices, GeneticAlgorithm, Device
from app import PygameApp

class TestCustomMapFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        # Initialize in headless/dummy mode for testing if needed
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    def setUp(self):
        self.app = PygameApp()

    def test_custom_map_starts_empty(self):
        """Verify that Custom Map mode initializes cleanly with 0 devices for user placement."""
        self.assertEqual(self.app.mode, "CUSTOM_MAP")
        self.assertEqual(len(self.app.devices), 0)
        self.assertEqual(len(self.app.ga.devices), 0)

    def test_image_scan_and_load(self):
        """Test image scanning in current directory."""
        self.assertGreater(len(self.app.available_image_paths), 0)
        self.assertIsNotNone(self.app.scaled_bg_image)

    def test_device_placement_and_ga_sync(self):
        """Test adding, moving, and removing custom devices."""
        initial_count = len(self.app.devices)
        
        # 1. Add device
        new_dev = Device(id=initial_count, x=50.0, y=50.0)
        self.app.devices.append(new_dev)
        self.app.ga.set_devices(self.app.devices)
        self.assertEqual(len(self.app.devices), initial_count + 1)
        self.assertEqual(len(self.app.ga.devices), initial_count + 1)
        
        # 2. Move device
        self.app.devices[-1].x = 60.0
        self.app.devices[-1].y = 70.0
        self.app.ga.set_devices(self.app.devices)
        self.assertEqual(self.app.ga.devices[-1].x, 60.0)
        self.assertEqual(self.app.ga.devices[-1].y, 70.0)
        
        # 3. Remove device
        self.app.devices.pop()
        self.app.ga.set_devices(self.app.devices)
        self.assertEqual(len(self.app.devices), initial_count)
        self.assertEqual(len(self.app.ga.devices), initial_count)

    def test_coordinate_scaling(self):
        """Test coordinate transformation between screen pixel space and grid space."""
        self.app.grid_size_px = 800
        self.app.grid_offset_x = 100
        self.app.grid_offset_y = 0
        self.app.ga.grid_size = 100
        
        # Screen (500, 400) -> GA (50.0, 50.0)
        gx, gy = self.app.screen_to_ga_coords(500, 400)
        self.assertAlmostEqual(gx, 50.0)
        self.assertAlmostEqual(gy, 50.0)
        
        # GA (25.0, 75.0) -> Screen (300, 600)
        sx, sy = self.app.ga_to_screen_coords(25.0, 75.0)
        self.assertEqual(sx, 300)
        self.assertEqual(sy, 600)

    def test_ap_count_change_preserves_custom_devices(self):
        """Verify that applying new AP count in CUSTOM_MAP mode preserves user placed devices."""
        # Add 3 custom devices
        self.app.devices = [
            Device(id=0, x=10.0, y=10.0),
            Device(id=1, x=20.0, y=20.0),
            Device(id=2, x=30.0, y=30.0)
        ]
        self.app.ga.set_devices(self.app.devices)
        
        # Change target AP count from 3 to 5
        self.app.target_num_aps = 5
        self.app.perform_apply_and_reset(new_seed=False)
        
        # Verify custom devices are unchanged
        self.assertEqual(len(self.app.devices), 3)
        self.assertEqual(self.app.devices[0].x, 10.0)
        self.assertEqual(self.app.devices[1].x, 20.0)
        self.assertEqual(self.app.devices[2].x, 30.0)
        self.assertEqual(self.app.ga.num_aps, 5)

if __name__ == '__main__':
    unittest.main()
