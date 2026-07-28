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
        # Verify img1 is present and prioritized
        image_stems = [os.path.splitext(os.path.basename(p))[0].lower() for p in self.app.available_image_paths]
        self.assertIn('img1', image_stems)
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
        # Set dynamic grid rect size
        self.app.grid_width = 1000
        self.app.grid_height = 800
        self.app.ga.grid_size = 100
        
        # Screen (500, 400) -> GA (50.0, 50.0)
        gx, gy = self.app.screen_to_ga_coords(500, 400)
        self.assertAlmostEqual(gx, 50.0)
        self.assertAlmostEqual(gy, 50.0)
        
        # GA (25.0, 75.0) -> Screen (250, 600)
        sx, sy = self.app.ga_to_screen_coords(25.0, 75.0)
        self.assertEqual(sx, 250)
        self.assertEqual(sy, 600)

if __name__ == '__main__':
    unittest.main()
