#!/usr/bin/env python3
import pygame
import sys
import time
import math
import os
from typing import Dict, Any, List, Optional, Tuple
from genetic_algorithm import generate_devices, GeneticAlgorithm, Individual, Device

# Initial default configuration (configurable via code constants)
DEFAULT_GRID_SIZE = 100
DEFAULT_NUM_DEVICES = 100
DEFAULT_NUM_APS = 3 #ap_count ap count

# Color definitions (Premium Slate/Cyan/Purple Theme)
COLOR_BG = (15, 23, 42)          # Slate-900
COLOR_GRID_BG = (5, 8, 17)        # Very dark blue
COLOR_PANEL_BG = (30, 41, 59)     # Slate-800
COLOR_PANEL_BORDER = (71, 85, 105) # Slate-600
COLOR_TEXT = (243, 244, 246)      # Slate-100
COLOR_TEXT_MUTED = (148, 163, 184) # Slate-400
COLOR_ACCENT = (6, 182, 212)      # Cyan
COLOR_CHART_AVG = (139, 92, 246)  # Purple
COLOR_RED = (239, 68, 68)         # Red
COLOR_GREEN = (16, 185, 129)      # Emerald Green
COLOR_ORANGE = (249, 115, 22)     # Orange
COLOR_YELLOW = (234, 179, 8)      # Yellow/Amber
COLOR_WHITE = (255, 255, 255)

# Up to 15 distinct AP Colors for dynamic configurations
AP_COLORS = [
    (59, 130, 246),   # 1. Blue
    (168, 85, 247),  # 2. Purple
    (16, 185, 129),  # 3. Emerald Green
    (249, 115, 22),   # 4. Orange
    (6, 182, 212),   # 5. Cyan
    (236, 72, 153),  # 6. Pink
    (234, 179, 8),    # 7. Yellow
    (99, 102, 241),  # 8. Indigo
    (20, 184, 166),  # 9. Teal
    (132, 204, 22),  # 10. Lime
    (245, 158, 11),  # 11. Amber
    (244, 63, 94),   # 12. Rose
    (139, 92, 246),  # 13. Violet
    (5, 150, 105),   # 14. Dark Green
    (217, 70, 239)   # 15. Fuchsia
]

class PygameApp:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        
        # Display setup - Store native desktop resolution prior to window creation
        info = pygame.display.Info()
        self.desktop_w = info.current_w if info.current_w > 0 else 1920
        self.desktop_h = info.current_h if info.current_h > 0 else 1080
        
        self.fullscreen = True
        try:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.RESIZABLE)
        except pygame.error:
            self.screen = pygame.display.set_mode((self.desktop_w, self.desktop_h), pygame.FULLSCREEN)
            self.fullscreen = True
            
        pygame.display.set_caption("Wireless AP Genetic Algorithm Optimizer - Custom Map Mode")
        self.clock = pygame.time.Clock()
        
        # Mode state: "CUSTOM_MAP" or "GA"
        self.mode = "CUSTOM_MAP"
        
        # Load premium system fonts safely
        self.font_title = pygame.font.SysFont("Trebuchet MS", 21, bold=True)
        self.font_subtitle = pygame.font.SysFont("Verdana", 11, italic=True)
        self.font_header = pygame.font.SysFont("Trebuchet MS", 14, bold=True)
        self.font_body = pygame.font.SysFont("Verdana", 11)
        self.font_mono = pygame.font.SysFont("Courier New", 11, bold=True)
        
        # Initialize active setup: Custom Map mode starts clean (0 devices) so user places all points
        self.current_seed = 42
        if self.mode == "CUSTOM_MAP":
            self.devices = []
            self.target_num_nodes = 0
            initial_capacity = 20
        else:
            self.devices = generate_devices(num_devices=DEFAULT_NUM_DEVICES, grid_size=DEFAULT_GRID_SIZE, seed=self.current_seed)
            self.target_num_nodes = DEFAULT_NUM_DEVICES
            initial_capacity = int(math.ceil(DEFAULT_NUM_DEVICES / DEFAULT_NUM_APS))
        
        # Initial GA configurations
        self.ga_config = {
            "pop_size": 100,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8,
            "elitism_count": 2,
            "ap_radius": 25.0,
            "ap_capacity": initial_capacity,
            "power_weight": 1.0,
            "overlap_weight": 120.0,
            "capacity_weight": 500.0,
            "power_exponent": 2.0
        }
        self.ga = GeneticAlgorithm(
            devices=self.devices,
            num_aps=DEFAULT_NUM_APS,
            grid_size=DEFAULT_GRID_SIZE,
            **self.ga_config
        )
        
        # Target parameters for runtime configuration
        self.target_grid_size = DEFAULT_GRID_SIZE
        self.target_num_aps = DEFAULT_NUM_APS
        self.show_links = False
        self.throttle_speed = False
        
        # Textbox Input variables
        self.active_input_key = None
        self.input_text = ""
        self.placeholder_text = ""
        
        # Execution control
        self.is_running = False
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0.0
        
        # Interactive Device Placement State
        self.dragging_device: Optional[Device] = None
        self.drag_start_pos: Optional[Tuple[int, int]] = None
        
        # Background Image management
        self.available_image_paths: List[str] = []
        self.current_image_idx = 0
        self.raw_bg_image: Optional[pygame.Surface] = None
        self.scaled_bg_image: Optional[pygame.Surface] = None
        
        # Layout metrics
        self.sidebar_width = 400
        self.update_layout()
        
        # Scan and load background image from current working directory
        self.scan_and_load_images()

        # Parameter Adjusters setup
        self.adjusters = [
            {"name": "Access Point Radius", "key": "ap_radius", "fmt": lambda x: f"{x:.1f}", "step": 1.0, "type": "slider"},
            {"name": "Mutation Rate", "key": "mutation_rate", "fmt": lambda x: f"{x:.3f}", "step": 0.01, "type": "slider"},
            {"name": "Crossover Rate", "key": "crossover_rate", "fmt": lambda x: f"{x:.3f}", "step": 0.05, "type": "slider"},
            {"name": "Power Weight", "key": "power_weight", "fmt": lambda x: f"{x:.1f}", "step": 0.1, "type": "slider"},
            {"name": "Overlap Weight", "key": "overlap_weight", "fmt": lambda x: f"{x:.0f}", "step": 10.0, "type": "slider"},
            {"name": "Capacity Weight", "key": "capacity_weight", "fmt": lambda x: f"{x:.0f}", "step": 50.0, "type": "slider"},
            {"name": "Grid Size", "key": "grid_size", "fmt": lambda x: f"{x:d}", "step": 50, "type": "slider"},
            {"name": "Devices", "key": "nodes", "fmt": lambda x: f"{x:d}", "step": 10, "type": "slider"},
            {"name": "Access Point Count", "key": "aps", "fmt": lambda x: f"{x:d}", "step": 1, "type": "slider"},
            {"name": "Show Links", "key": "show_links", "fmt": lambda x: "ON" if x else "OFF", "step": 0, "type": "toggle"},
            {"name": "Throttle Speed", "key": "throttle_speed", "fmt": lambda x: "ON" if x else "OFF", "step": 0, "type": "toggle"}
        ]
        self.reposition_controls()

    def update_layout(self):
        """Calculates dynamic layout rects, expanding sidebar and scaling fonts when extra space is available."""
        self.screen_width, self.screen_height = self.screen.get_size()
        container_h = self.screen_height
        
        # Grid is a 1:1 aspect ratio square bounded by height and screen width minus minimum sidebar (380px)
        max_grid_w = max(100, self.screen_width - 380)
        self.grid_size_px = min(max_grid_w, container_h)
        self.grid_offset_x = 0
        self.grid_offset_y = (container_h - self.grid_size_px) // 2
        
        self.rect_grid_container = pygame.Rect(0, 0, self.grid_size_px, container_h)
        self.rect_grid = pygame.Rect(0, self.grid_offset_y, self.grid_size_px, self.grid_size_px)
        
        # Sidebar expands to take up ALL remaining horizontal space on the right (min 380px)
        self.sidebar_width = max(380, self.screen_width - self.grid_size_px)
        self.rect_sidebar = pygame.Rect(self.grid_size_px, 0, self.sidebar_width, self.screen_height)
        
        # Dynamic font scaling factor (scales fonts up to 3x when extra space is available)
        width_ratio = self.sidebar_width / 380.0
        height_ratio = self.screen_height / 750.0
        self.font_scale = max(1.0, min(3, min(width_ratio, height_ratio)))
        
        # Reload fonts dynamically with larger scaled sizes
        f_title_sz = int(22 * self.font_scale)
        f_hdr_sz = int(15 * self.font_scale)
        f_body_sz = int(12 * self.font_scale)
        f_mono_sz = int(12 * self.font_scale)
        
        self.font_title = pygame.font.SysFont("Trebuchet MS", f_title_sz, bold=True)
        self.font_subtitle = pygame.font.SysFont("Verdana", f_body_sz, italic=True)
        self.font_header = pygame.font.SysFont("Trebuchet MS", f_hdr_sz, bold=True)
        self.font_body = pygame.font.SysFont("Verdana", f_body_sz)
        self.font_mono = pygame.font.SysFont("Courier New", f_mono_sz, bold=True)
        
        # Scale factor (pixels per grid unit)
        grid_size = float(self.ga.grid_size) if hasattr(self, 'ga') else 100.0
        self.scale = float(self.grid_size_px) / grid_size
        
        # Dynamic Sidebar Action Buttons
        sb_x = self.rect_sidebar.x
        pad_x = 20
        avail_w = self.sidebar_width - 40
        btn_h = int(28 * self.font_scale)
        
        btn_w_half = (avail_w - 10) // 2
        btn_w_third = (avail_w - 20) // 3
        
        y_start = int(90 * self.font_scale)
        spacing = btn_h + 6
        
        # Row 1: Mode & Image Controls
        self.btn_mode = pygame.Rect(sb_x + pad_x, y_start, btn_w_half, btn_h)
        self.btn_next_image = pygame.Rect(sb_x + pad_x + btn_w_half + 10, y_start, btn_w_half, btn_h)
        
        # Row 2: Play & Step Actions
        y2 = y_start + spacing
        self.btn_play = pygame.Rect(sb_x + pad_x, y2, btn_w_third, btn_h)
        self.btn_step = pygame.Rect(sb_x + pad_x + btn_w_third + 10, y2, btn_w_third, btn_h)
        self.btn_step10 = pygame.Rect(sb_x + pad_x + (btn_w_third + 10)*2, y2, btn_w_third, btn_h)
        
        # Row 3: GA Reset, Device Edit Actions
        y3 = y2 + spacing
        self.btn_reset_ga = pygame.Rect(sb_x + pad_x, y3, btn_w_third, btn_h)
        self.btn_clear_devices = pygame.Rect(sb_x + pad_x + btn_w_third + 10, y3, btn_w_third, btn_h)
        self.btn_rotate_nodes = pygame.Rect(sb_x + pad_x + (btn_w_third + 10)*2, y3, btn_w_third, btn_h)
        
        # Apply Settings button
        y4 = y3 + spacing
        self.btn_apply_tgt = pygame.Rect(sb_x + pad_x, y4, avail_w, btn_h)
        
        # Scale background image if loaded
        self.scale_background_image()
        
        if hasattr(self, 'adjusters'):
            self.reposition_controls()

    def get_visible_adjusters(self):
        """Returns parameter adjusters relevant to the active mode (hides Grid Size & Devices in User Defined mode)."""
        if self.mode == "CUSTOM_MAP":
            return [adj for adj in self.adjusters if adj["key"] not in ("grid_size", "nodes")]
        return self.adjusters

    def reposition_controls(self):
        """Positions parameter adjusters inside sidebar with dynamic widths and spacing."""
        sb_x = self.rect_sidebar.x
        pad_x = 20
        avail_w = self.sidebar_width - 40
        reset_needed = self.is_reset_required()
        
        btn_h = int(28 * self.font_scale)
        y4 = int(90 * self.font_scale) + (btn_h + 6) * 3
        start_y = y4 + btn_h + int((26 if not reset_needed else 38) * self.font_scale)
        
        row_h = int(24 * self.font_scale)
        lbl_w = min(int(175 * self.font_scale), avail_w // 2)
        btn_pm_w = int(38 * self.font_scale)
        val_w = max(90, avail_w - lbl_w - btn_pm_w * 2 - 15)
        
        for idx, adj in enumerate(self.get_visible_adjusters()):
            y_pos = start_y + idx * row_h
            adj["rect_val"] = pygame.Rect(sb_x + pad_x + lbl_w, y_pos, val_w, row_h - 4)
            if adj["type"] == "slider":
                adj["rect_minus"] = pygame.Rect(sb_x + pad_x + lbl_w + val_w + 5, y_pos, btn_pm_w, row_h - 4)
                adj["rect_plus"] = pygame.Rect(sb_x + pad_x + lbl_w + val_w + 10 + btn_pm_w, y_pos, btn_pm_w, row_h - 4)
            else: # toggle
                adj["rect_toggle"] = pygame.Rect(sb_x + pad_x + lbl_w + val_w + 5, y_pos, btn_pm_w * 2 + 5, row_h - 4)

    def scan_and_load_images(self):
        """Scans current working directory for image files, prioritizing img1.jpg."""
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        try:
            files = os.listdir('.')
        except Exception:
            files = []
            
        img_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        
        # Prioritize img1.* image files
        img_files.sort(key=lambda x: (0 if os.path.splitext(x.lower())[0] == 'img1' else 1, x.lower()))
        
        self.available_image_paths = [os.path.abspath(f) for f in img_files]
        self.current_image_idx = 0
        self.load_current_image()

    def load_current_image(self):
        """Loads current image surface from available image paths."""
        if not self.available_image_paths:
            self.raw_bg_image = None
            self.scaled_bg_image = None
            return
            
        try:
            img_path = self.available_image_paths[self.current_image_idx]
            self.raw_bg_image = pygame.image.load(img_path).convert()
            self.scale_background_image()
        except Exception as e:
            print(f"Failed to load image: {e}")
            self.raw_bg_image = None
            self.scaled_bg_image = None

    def scale_background_image(self):
        """Scales raw background image surface to fit square grid viewport dimensions."""
        if self.raw_bg_image is not None and hasattr(self, 'grid_size_px') and self.grid_size_px > 0:
            try:
                self.scaled_bg_image = pygame.transform.smoothscale(
                    self.raw_bg_image,
                    (self.grid_size_px, self.grid_size_px)
                )
            except Exception:
                self.scaled_bg_image = None
        else:
            self.scaled_bg_image = None

    def cycle_next_image(self):
        """Switches to the next background image in the working directory."""
        if len(self.available_image_paths) > 1:
            self.current_image_idx = (self.current_image_idx + 1) % len(self.available_image_paths)
            self.load_current_image()

    def screen_to_ga_coords(self, sx: int, sy: int) -> Tuple[float, float]:
        """Converts screen pixel coordinates inside grid viewport to GA grid space [0, grid_size]."""
        grid_size = float(self.ga.grid_size)
        rel_x = sx - self.grid_offset_x
        rel_y = sy - self.grid_offset_y
        gx = max(0.0, min(grid_size, (rel_x / float(self.grid_size_px)) * grid_size))
        gy = max(0.0, min(grid_size, (rel_y / float(self.grid_size_px)) * grid_size))
        return (gx, gy)

    def ga_to_screen_coords(self, gx: float, gy: float) -> Tuple[int, int]:
        """Converts GA grid space coordinates [0, grid_size] to screen pixel coordinates."""
        grid_size = float(self.ga.grid_size)
        sx = self.grid_offset_x + int((gx / grid_size) * self.grid_size_px)
        sy = self.grid_offset_y + int((gy / grid_size) * self.grid_size_px)
        return (sx, sy)

    def find_device_near_screen_pos(self, sx: int, sy: int, threshold_pixels: float = 14.0) -> Optional[Device]:
        """Finds closest device within a pixel distance threshold."""
        closest_dev = None
        min_dist = float('inf')
        for dev in self.devices:
            dsx, dsy = self.ga_to_screen_coords(dev.x, dev.y)
            dist = math.hypot(sx - dsx, sy - dsy)
            if dist < min_dist and dist <= threshold_pixels:
                min_dist = dist
                closest_dev = dev
        return closest_dev

    def toggle_fullscreen(self):
        """Toggles display between fullscreen and windowed mode reliably across Wayland and X11."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            try:
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.RESIZABLE)
            except pygame.error:
                self.screen = pygame.display.set_mode((self.desktop_w, self.desktop_h), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
        self.update_layout()

    def run(self):
        while True:
            self.handle_events()
            self.update_logic()
            self.update_cursor()
            self.draw()
            self.clock.tick(60)

    def update_logic(self):
        if self.is_running:
            if self.throttle_speed:
                current_time = time.time()
                if current_time - self.last_step_time >= self.step_delay:
                    self.ga.step()
                    self.last_step_time = current_time
            else:
                self.ga.step()

    def update_cursor(self):
        """Changes mouse cursor to hand/crosshair based on active interactions."""
        pos = pygame.mouse.get_pos()
        hovering_button = False
        
        # Check action buttons
        for btn in [self.btn_mode, self.btn_next_image, self.btn_play, self.btn_step, 
                    self.btn_step10, self.btn_reset_ga, self.btn_clear_devices, 
                    self.btn_rotate_nodes, self.btn_apply_tgt]:
            if btn.collidepoint(pos):
                hovering_button = True
                break
                
        if not hovering_button:
            for adj in self.get_visible_adjusters():
                if "rect_val" in adj and adj["rect_val"].collidepoint(pos):
                    hovering_button = True
                    break
                if adj["type"] == "slider":
                    if "rect_minus" in adj and (adj["rect_minus"].collidepoint(pos) or adj["rect_plus"].collidepoint(pos)):
                        hovering_button = True
                        break
                elif adj["type"] == "toggle":
                    if "rect_toggle" in adj and adj["rect_toggle"].collidepoint(pos):
                        hovering_button = True
                        break
                        
        if hovering_button:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        elif self.rect_grid.collidepoint(pos):
            near_dev = self.find_device_near_screen_pos(pos[0], pos[1])
            if near_dev or self.dragging_device:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
            else:
                pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    def is_reset_required(self) -> bool:
        """Checks if target setup parameters differ from active GA state."""
        return (
            self.target_grid_size != self.ga.grid_size or
            self.target_num_nodes != len(self.devices) or
            self.target_num_aps != self.ga.num_aps
        )

    def get_max_bound(self, key: str) -> float:
        if key == "ap_radius":
            return float(self.ga.grid_size)
        elif key == "nodes":
            return float(self.target_grid_size * self.target_grid_size)
        elif key == "aps":
            return 15.0
        elif key == "grid_size":
            return 1000.0
        elif key in ["mutation_rate", "crossover_rate"]:
            return 1.0
        elif key in ["power_weight", "overlap_weight", "capacity_weight"]:
            return 10000.0
        return float('inf')

    def get_min_bound(self, key: str) -> float:
        if key in ["grid_size", "nodes", "aps"]:
            return 10.0 if key == "grid_size" else 1.0
        elif key == "ap_radius":
            return 0.5
        return 0.0

    def set_ap_count_live(self, new_num_aps: int):
        """Updates AP count dynamically on-the-fly without restarting or wiping GA progress."""
        new_num_aps = max(1, min(20, int(new_num_aps)))
        old_num_aps = self.ga.num_aps
        self.target_num_aps = new_num_aps
        
        if new_num_aps == old_num_aps:
            return
            
        self.ga.num_aps = new_num_aps
        
        # Calculate dynamic AP capacity
        num_devices = len(self.devices)
        dynamic_capacity = int(math.ceil(num_devices / new_num_aps)) if (num_devices > 0 and new_num_aps > 0) else 1
        self.ga_config["ap_capacity"] = dynamic_capacity
        self.ga.ap_capacity = dynamic_capacity
        
        # Adapt AP coordinates array for all individuals in current population
        for ind in self.ga.population:
            if new_num_aps > len(ind.aps):
                for _ in range(new_num_aps - len(ind.aps)):
                    rx = self.ga.rng.uniform(0, self.ga.grid_size)
                    ry = self.ga.rng.uniform(0, self.ga.grid_size)
                    ind.aps.append((rx, ry))
            elif new_num_aps < len(ind.aps):
                ind.aps = ind.aps[:new_num_aps]
                
        # Re-evaluate fitness for updated AP count
        self.ga.evaluate_population(self.ga.population)
        self.ga.sort_population()
        if self.ga.best_history:
            self.ga.best_history[-1] = self.ga.population[0].total_cost
            self.ga.avg_history[-1] = sum(ind.total_cost for ind in self.ga.population) / len(self.ga.population)

    def apply_input_value(self):
        if not self.active_input_key:
            return
        
        val_text = self.input_text.replace('%', '').strip()
        if not val_text:
            self.active_input_key = None
            return
            
        try:
            val = float(val_text)
            key = self.active_input_key
            
            min_b = self.get_min_bound(key)
            max_b = self.get_max_bound(key)
            val = max(min_b, min(max_b, val))
            
            if key == "grid_size":
                self.target_grid_size = int(val)
                self.target_num_nodes = min(self.target_grid_size * self.target_grid_size, self.target_num_nodes)
            elif key == "nodes":
                self.target_num_nodes = int(val)
            elif key == "aps":
                self.set_ap_count_live(int(val))
            else:
                self.update_param(key, val)
                
        except ValueError:
            pass
            
        self.active_input_key = None

    def perform_apply_and_reset(self, new_seed: bool = False):
        """Re-initializes GA using target values, preserving user-placed devices in CUSTOM_MAP mode."""
        self.is_running = False
        if new_seed:
            self.current_seed += 1
            
        if self.mode == "GA" or (new_seed and self.mode == "CUSTOM_MAP"):
            self.devices = generate_devices(
                num_devices=self.target_num_nodes,
                grid_size=self.target_grid_size,
                seed=self.current_seed
            )
        else:
            # In CUSTOM_MAP mode when applying settings, preserve existing user-placed devices
            self.target_num_nodes = len(self.devices)
            
        num_devices = len(self.devices)
        dynamic_capacity = int(math.ceil(num_devices / self.target_num_aps)) if (num_devices > 0 and self.target_num_aps > 0) else 1
        self.ga_config["ap_capacity"] = dynamic_capacity
        
        self.ga = GeneticAlgorithm(
            devices=self.devices,
            num_aps=self.target_num_aps,
            grid_size=self.target_grid_size,
            pop_size=self.ga_config["pop_size"],
            mutation_rate=self.ga_config["mutation_rate"],
            crossover_rate=self.ga_config["crossover_rate"],
            ap_radius=self.ga_config["ap_radius"],
            ap_capacity=self.ga_config["ap_capacity"],
            power_weight=self.ga_config["power_weight"],
            overlap_weight=self.ga_config["overlap_weight"],
            capacity_weight=self.ga_config["capacity_weight"],
            power_exponent=self.ga_config["power_exponent"]
        )
        self.last_step_time = 0.0

    def toggle_mode(self):
        """Toggles mode between CUSTOM_MAP and GA."""
        if self.mode == "CUSTOM_MAP":
            self.mode = "GA"
            if not self.devices:
                self.target_num_nodes = DEFAULT_NUM_DEVICES
                self.devices = generate_devices(num_devices=self.target_num_nodes, grid_size=self.target_grid_size, seed=self.current_seed)
                self.ga.set_devices(self.devices)
        else:
            self.mode = "CUSTOM_MAP"
            # When switching into Custom Map mode, start clean so user places all points
            self.devices = []
            self.target_num_nodes = 0
            self.ga.set_devices(self.devices)
        self.reposition_controls()

    def clear_all_devices(self):
        """Clears all devices on the map and updates GA."""
        self.devices = []
        self.target_num_nodes = 0
        self.ga.set_devices(self.devices)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.VIDEORESIZE:
                self.update_layout()
                
            elif self.active_input_key is not None:
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                        self.apply_input_value()
                    elif event.key == pygame.K_ESCAPE:
                        self.active_input_key = None
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    else:
                        if event.unicode in "0123456789.-":
                            self.input_text += event.unicode
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        clicked_another = False
                        for adj in self.get_visible_adjusters():
                            if adj["rect_val"].collidepoint(event.pos):
                                self.apply_input_value()
                                self.active_input_key = adj["key"]
                                val = self.target_grid_size if adj["key"] == "grid_size" else (
                                      self.target_num_nodes if adj["key"] == "nodes" else (
                                      self.target_num_aps if adj["key"] == "aps" else (
                                      self.show_links if adj["key"] == "show_links" else (
                                      self.throttle_speed if adj["key"] == "throttle_speed" else self.ga_config[adj["key"]]
                                ))))
                                self.placeholder_text = adj["fmt"](val) if adj["key"] not in ["show_links", "throttle_speed"] else ""
                                self.input_text = ""
                                clicked_another = True
                                break
                        if not clicked_another:
                            self.apply_input_value()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_SPACE:
                    self.is_running = not self.is_running
                elif event.key == pygame.K_s:
                    self.is_running = False
                    self.ga.step()
                elif event.key == pygame.K_r:
                    self.is_running = False
                    self.ga.initialize_population()
                elif event.key == pygame.K_c:
                    self.clear_all_devices()
                elif event.key == pygame.K_m:
                    self.toggle_mode()
                elif event.key == pygame.K_n:
                    self.perform_apply_and_reset(new_seed=True)
                elif event.key == pygame.K_a:
                    self.perform_apply_and_reset(new_seed=False)
                elif event.key == pygame.K_UP:
                    self.step_delay = max(0.005, self.step_delay - 0.01)
                elif event.key == pygame.K_DOWN:
                    self.step_delay = min(0.5, self.step_delay + 0.01)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if event.button == 1: # Left Click
                    if self.rect_grid.collidepoint(pos):
                        # Map device placement/drag interaction
                        near_dev = self.find_device_near_screen_pos(pos[0], pos[1])
                        if near_dev:
                            self.dragging_device = near_dev
                        else:
                            # Add new device
                            gx, gy = self.screen_to_ga_coords(pos[0], pos[1])
                            new_dev = Device(id=len(self.devices), x=gx, y=gy)
                            self.devices.append(new_dev)
                            self.dragging_device = new_dev
                            self.target_num_nodes = len(self.devices)
                            self.ga.set_devices(self.devices, sort=False)
                    else:
                        # Sidebar Action buttons
                        if self.btn_mode.collidepoint(pos):
                            self.toggle_mode()
                        elif self.btn_next_image.collidepoint(pos):
                            self.cycle_next_image()
                        elif self.btn_play.collidepoint(pos):
                            self.is_running = not self.is_running
                        elif self.btn_step.collidepoint(pos):
                            self.is_running = False
                            self.ga.step()
                        elif self.btn_step10.collidepoint(pos):
                            self.is_running = False
                            for _ in range(10):
                                self.ga.step()
                        elif self.btn_reset_ga.collidepoint(pos):
                            self.is_running = False
                            self.ga.initialize_population()
                        elif self.btn_clear_devices.collidepoint(pos):
                            self.clear_all_devices()
                        elif self.btn_rotate_nodes.collidepoint(pos):
                            self.perform_apply_and_reset(new_seed=True)
                        elif self.btn_apply_tgt.collidepoint(pos):
                            self.perform_apply_and_reset(new_seed=False)
                            
                        # Adjusters
                        for adj in self.get_visible_adjusters():
                            key = adj["key"]
                            if adj["rect_val"].collidepoint(pos):
                                self.active_input_key = key
                                val = self.target_grid_size if key == "grid_size" else (
                                      self.target_num_nodes if key == "nodes" else (
                                      self.target_num_aps if key == "aps" else (
                                      self.show_links if key == "show_links" else (
                                      self.throttle_speed if key == "throttle_speed" else self.ga_config[key]
                                ))))
                                self.placeholder_text = adj["fmt"](val) if key not in ["show_links", "throttle_speed"] else ""
                                self.input_text = ""
                                break
                                
                            if adj["type"] == "toggle":
                                if adj["rect_toggle"].collidepoint(pos):
                                    if key == "show_links":
                                        self.show_links = not self.show_links
                                    elif key == "throttle_speed":
                                        self.throttle_speed = not self.throttle_speed
                            else:
                                val = self.target_grid_size if key == "grid_size" else (
                                      self.target_num_nodes if key == "nodes" else (
                                      self.target_num_aps if key == "aps" else self.ga_config[key]
                                ))
                                min_b = self.get_min_bound(key)
                                max_b = self.get_max_bound(key)
                                
                                if adj["rect_minus"].collidepoint(pos):
                                    new_val = max(min_b, val - adj["step"])
                                    if key == "grid_size":
                                        self.target_grid_size = int(new_val)
                                        self.target_num_nodes = min(self.target_grid_size * self.target_grid_size, self.target_num_nodes)
                                    elif key == "nodes":
                                        self.target_num_nodes = int(new_val)
                                    elif key == "aps":
                                        self.set_ap_count_live(int(new_val))
                                    else:
                                        self.update_param(key, new_val)
                                elif adj["rect_plus"].collidepoint(pos):
                                    new_val = min(max_b, val + adj["step"])
                                    if key == "grid_size":
                                        self.target_grid_size = int(new_val)
                                    elif key == "nodes":
                                        self.target_num_nodes = int(new_val)
                                    elif key == "aps":
                                        self.set_ap_count_live(int(new_val))
                                    else:
                                        self.update_param(key, new_val)

                elif event.button == 3: # Right Click - Remove device
                    if self.rect_grid.collidepoint(pos):
                        near_dev = self.find_device_near_screen_pos(pos[0], pos[1])
                        if near_dev:
                            self.devices.remove(near_dev)
                            # Re-index device IDs
                            for idx, d in enumerate(self.devices):
                                d.id = idx
                            self.target_num_nodes = len(self.devices)
                            self.ga.set_devices(self.devices, sort=False)

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_device is not None and event.buttons[0] == 1:
                    pos = event.pos
                    gx, gy = self.screen_to_ga_coords(pos[0], pos[1])
                    self.dragging_device.x = gx
                    self.dragging_device.y = gy
                    self.ga.set_devices(self.devices, sort=False)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_device = None

    def update_param(self, key: str, value: Any):
        if key in ["pop_size"]:
            value = int(value)
        self.ga_config[key] = value
        self.ga.update_parameters({key: value})

    def draw(self):
        self.screen.fill(COLOR_BG)
        self.draw_grid_panel()
        self.draw_sidebar_panel()
        pygame.display.flip()

    def draw_grid_panel(self):
        # 1. Fill grid container area
        pygame.draw.rect(self.screen, COLOR_BG, self.rect_grid_container)
        pygame.draw.rect(self.screen, COLOR_GRID_BG, self.rect_grid)
        
        # 2. Render Scaled Background Image if available
        if self.scaled_bg_image is not None:
            self.screen.blit(self.scaled_bg_image, (self.grid_offset_x, self.grid_offset_y))
            overlay = pygame.Surface((self.grid_size_px, self.grid_size_px), pygame.SRCALPHA)
            overlay_alpha = 40 if self.mode == "CUSTOM_MAP" else 100
            overlay.fill((0, 0, 0, overlay_alpha))
            self.screen.blit(overlay, (self.grid_offset_x, self.grid_offset_y))
            
        best_ind = self.ga.get_best_individual()
        grid_size = float(self.ga.grid_size)
        
        # 3. Draw subtle grid lines
        scale_x = float(self.grid_size_px) / grid_size
        if scale_x >= 5.0:
            step = 1
        elif scale_x >= 2.0:
            step = 5
        elif scale_x * 10 >= 4.0:
            step = 10
        elif scale_x * 50 >= 4.0:
            step = 50
        else:
            step = 100
            
        grid_line_color = (255, 255, 255, 25) if self.scaled_bg_image else (20, 25, 40)
        grid_overlay = pygame.Surface((self.grid_size_px, self.grid_size_px), pygame.SRCALPHA)
        for i in range(step, int(grid_size), step):
            grid_pos = int((i / grid_size) * self.grid_size_px)
            pygame.draw.line(grid_overlay, grid_line_color, (grid_pos, 0), (grid_pos, self.grid_size_px), 1)
            pygame.draw.line(grid_overlay, grid_line_color, (0, grid_pos), (self.grid_size_px, grid_pos), 1)
        self.screen.blit(grid_overlay, (self.grid_offset_x, self.grid_offset_y))
        
        # 4. Draw AP coverage circles (translucent overlay with 1:1 scale)
        ap_overlay = pygame.Surface((self.grid_size_px, self.grid_size_px), pygame.SRCALPHA)
        ap_radius_px = int((self.ga.ap_radius / grid_size) * self.grid_size_px)
        for idx, ap in enumerate(best_ind.aps):
            cx = int((ap[0] / grid_size) * self.grid_size_px)
            cy = int((ap[1] / grid_size) * self.grid_size_px)
            color = AP_COLORS[idx % len(AP_COLORS)]
            pygame.draw.circle(ap_overlay, color + (30,), (cx, cy), ap_radius_px)
            pygame.draw.circle(ap_overlay, color + (70,), (cx, cy), ap_radius_px, 1)
        self.screen.blit(ap_overlay, (self.grid_offset_x, self.grid_offset_y))
        
        # Outer Border around square grid area
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.rect_grid, 2)
        
        # 5. Draw connection paths (only if show_links toggle is ON)
        if self.show_links and self.devices:
            for dev in self.devices:
                if dev.id < len(best_ind.device_assignments):
                    ap_idx = best_ind.device_assignments[dev.id]
                    if 0 <= ap_idx < len(best_ind.aps):
                        ap = best_ind.aps[ap_idx]
                        ap_color = AP_COLORS[ap_idx % len(AP_COLORS)]
                        start_pos = self.ga_to_screen_coords(dev.x, dev.y)
                        end_pos = self.ga_to_screen_coords(ap[0], ap[1])
                        pygame.draw.line(self.screen, ap_color + (80,), start_pos, end_pos, 1)
                    
        # 6. Draw Device Nodes
        mouse_pos = pygame.mouse.get_pos()
        hovered_dev = self.find_device_near_screen_pos(mouse_pos[0], mouse_pos[1]) if self.rect_grid.collidepoint(mouse_pos) else None
        
        for dev in self.devices:
            ap_idx = best_ind.device_assignments[dev.id] if dev.id < len(best_ind.device_assignments) else -1
            color = AP_COLORS[ap_idx % len(AP_COLORS)] if (0 <= ap_idx < len(best_ind.aps)) else COLOR_TEXT_MUTED
            pos = self.ga_to_screen_coords(dev.x, dev.y)
            
            is_active = (dev == self.dragging_device or dev == hovered_dev)
            radius = 6 if is_active else 4
            
            pygame.draw.circle(self.screen, color, pos, radius)
            pygame.draw.circle(self.screen, COLOR_WHITE if is_active else COLOR_GRID_BG, pos, radius + 1, 1)
            
        # 7. Draw Access Point Centers
        for idx, ap in enumerate(best_ind.aps):
            cx, cy = self.ga_to_screen_coords(ap[0], ap[1])
            color = AP_COLORS[idx % len(AP_COLORS)]
            load = best_ind.ap_loads[idx] if idx < len(best_ind.ap_loads) else 0
            is_overcapacity = load > self.ga.ap_capacity
            
            if is_overcapacity:
                pulse_r = 15 + int(3 * math.sin(time.time() * 10))
                pygame.draw.circle(self.screen, COLOR_RED, (cx, cy), pulse_r, 2)
                
            pygame.draw.circle(self.screen, color, (cx, cy), 10)
            pygame.draw.circle(self.screen, COLOR_WHITE, (cx, cy), 10, 2)
            pygame.draw.circle(self.screen, COLOR_WHITE, (cx, cy), 3)
            
            lbl = self.font_mono.render(str(idx + 1), True, COLOR_WHITE)
            self.screen.blit(lbl, (cx - lbl.get_width()//2, cy - 25))

        # 8. Map Placement Guidance Banner
        img_name = os.path.basename(self.available_image_paths[self.current_image_idx]) if self.available_image_paths else "No Image"
        
        t_img = self.font_subtitle.render(f"Map Image: {img_name} | Devices: {len(self.devices)}", True, COLOR_TEXT_MUTED)
        t_hint = self.font_subtitle.render("Left-Click: Add/Drag Device | Right-Click: Remove", True, COLOR_YELLOW)
        
        pad_b = 10
        calc_w = max(t_img.get_width(), t_hint.get_width()) + pad_b * 2
        b_w = max(380, min(calc_w, self.grid_size_px - 20))
        
        lh2 = t_img.get_height()
        lh3 = t_hint.get_height()
        b_h = lh2 + lh3 + pad_b * 2
        
        banner_x = self.grid_offset_x + 10
        banner_y = self.grid_offset_y + 10
        banner_surface = pygame.Surface((b_w, b_h), pygame.SRCALPHA)
        banner_surface.fill((15, 23, 42, 220))
        self.screen.blit(banner_surface, (banner_x, banner_y))
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, (banner_x, banner_y, b_w, b_h), 1, border_radius=4)
        
        y_pos1 = banner_y + pad_b
        y_pos2 = y_pos1 + lh2 + 2
        
        self.screen.blit(t_img, (banner_x + pad_b, y_pos1))
        self.screen.blit(t_hint, (banner_x + pad_b, y_pos2))

    def draw_sidebar_panel(self):
        sb_x = self.rect_sidebar.x
        pad_x = 20
        avail_w = self.sidebar_width - 40
        right_edge = sb_x + self.sidebar_width - 20
        
        pygame.draw.rect(self.screen, COLOR_BG, self.rect_sidebar)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (sb_x, 0), (sb_x, self.screen_height), 2)
        
        best_ind = self.ga.get_best_individual()
        reset_needed = self.is_reset_required()
        mouse_pos = pygame.mouse.get_pos()
        
        # 1. Header Title
        title = self.font_title.render("Wireless GA Optimizer", True, COLOR_ACCENT)
        self.screen.blit(title, (sb_x + pad_x, 15))
        
        sub_text = f"Active: Grid {self.ga.grid_size}x{self.ga.grid_size} | Nodes: {len(self.devices)} | APs: {self.ga.num_aps}"
        sub = self.font_subtitle.render(sub_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(sub, (sb_x + pad_x, int(42 * self.font_scale)))
        
        seed_txt = f"Seed: {self.current_seed}"
        lbl_seed = self.font_subtitle.render(seed_txt, True, COLOR_ACCENT)
        self.screen.blit(lbl_seed, (right_edge - lbl_seed.get_width(), int(42 * self.font_scale)))
        
        line_y1 = int(62 * self.font_scale)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (sb_x + pad_x, line_y1), (right_edge, line_y1), 1)
        
        # 2. Status Details
        txt_gen = self.font_header.render(f"Generation: {self.ga.generation}", True, COLOR_TEXT)
        self.screen.blit(txt_gen, (sb_x + pad_x, line_y1 + 10))
        
        status_text = "RUNNING" if self.is_running else "PAUSED"
        status_color = COLOR_GREEN if self.is_running else COLOR_ORANGE
        txt_status = self.font_header.render(status_text, True, status_color)
        self.screen.blit(txt_status, (right_edge - txt_status.get_width(), line_y1 + 10))
        
        # 3. Action Buttons Rendering
        mode_btn_lbl = "Mode: User defined" if self.mode == "CUSTOM_MAP" else "Mode: Auto Generate"
        mode_btn_color = COLOR_ACCENT if self.mode == "CUSTOM_MAP" else COLOR_PANEL_BG
        pygame.draw.rect(self.screen, mode_btn_color, self.btn_mode, border_radius=6)
        lbl_m = self.font_header.render(mode_btn_lbl, True, COLOR_WHITE if self.mode == "CUSTOM_MAP" else COLOR_TEXT)
        self.screen.blit(lbl_m, (self.btn_mode.centerx - lbl_m.get_width()//2, self.btn_mode.centery - lbl_m.get_height()//2))
        if self.btn_mode.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_mode, 2, border_radius=6)
            
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_next_image, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_next_image, 1, border_radius=6)
        lbl_img = self.font_header.render("Switch Image", True, COLOR_TEXT)
        self.screen.blit(lbl_img, (self.btn_next_image.centerx - lbl_img.get_width()//2, self.btn_next_image.centery - lbl_img.get_height()//2))
        if self.btn_next_image.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_next_image, 2, border_radius=6)
            
        # Optimize / Step / Step 10 Buttons
        play_btn_color = COLOR_ORANGE if self.is_running else COLOR_GREEN
        play_btn_lbl = "Pause" if self.is_running else "Optimize"
        pygame.draw.rect(self.screen, play_btn_color, self.btn_play, border_radius=6)
        lbl_play = self.font_header.render(play_btn_lbl, True, COLOR_WHITE)
        self.screen.blit(lbl_play, (self.btn_play.centerx - lbl_play.get_width()//2, self.btn_play.centery - lbl_play.get_height()//2))
        if self.btn_play.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_play, 2, border_radius=6)
            
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step, 1, border_radius=6)
        lbl_step = self.font_header.render("Step 1", True, COLOR_TEXT)
        self.screen.blit(lbl_step, (self.btn_step.centerx - lbl_step.get_width()//2, self.btn_step.centery - lbl_step.get_height()//2))
        if self.btn_step.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_step, 2, border_radius=6)
            
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step10, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step10, 1, border_radius=6)
        lbl_step10 = self.font_header.render("Step 10", True, COLOR_TEXT)
        self.screen.blit(lbl_step10, (self.btn_step10.centerx - lbl_step10.get_width()//2, self.btn_step10.centery - lbl_step10.get_height()//2))
        if self.btn_step10.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_step10, 2, border_radius=6)

        # Reset APs / Clear Devices / New Devices Buttons
        pygame.draw.rect(self.screen, COLOR_RED, self.btn_reset_ga, border_radius=6)
        lbl_reset_ga = self.font_header.render("Reset APs", True, COLOR_WHITE)
        self.screen.blit(lbl_reset_ga, (self.btn_reset_ga.centerx - lbl_reset_ga.get_width()//2, self.btn_reset_ga.centery - lbl_reset_ga.get_height()//2))
        if self.btn_reset_ga.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_reset_ga, 2, border_radius=6)

        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_clear_devices, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_clear_devices, 1, border_radius=6)
        lbl_clr = self.font_header.render("Clear Devices", True, COLOR_TEXT)
        self.screen.blit(lbl_clr, (self.btn_clear_devices.centerx - lbl_clr.get_width()//2, self.btn_clear_devices.centery - lbl_clr.get_height()//2))
        if self.btn_clear_devices.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_clear_devices, 2, border_radius=6)

        pygame.draw.rect(self.screen, COLOR_CHART_AVG, self.btn_rotate_nodes, border_radius=6)
        lbl_rot = self.font_header.render("New Devices", True, COLOR_WHITE)
        self.screen.blit(lbl_rot, (self.btn_rotate_nodes.centerx - lbl_rot.get_width()//2, self.btn_rotate_nodes.centery - lbl_rot.get_height()//2))
        if self.btn_rotate_nodes.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_rotate_nodes, 2, border_radius=6)

        # Apply Tgt Configurations
        apply_btn_color = COLOR_ORANGE if reset_needed else COLOR_PANEL_BG
        apply_text_color = COLOR_WHITE if reset_needed else COLOR_TEXT
        pygame.draw.rect(self.screen, apply_btn_color, self.btn_apply_tgt, border_radius=6)
        if not reset_needed:
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_apply_tgt, 1, border_radius=6)
        lbl_apply = self.font_header.render("Apply Target Settings", True, apply_text_color)
        self.screen.blit(lbl_apply, (self.btn_apply_tgt.centerx - lbl_apply.get_width()//2, self.btn_apply_tgt.centery - lbl_apply.get_height()//2))
        if self.btn_apply_tgt.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_apply_tgt, 2, border_radius=6)

        btn_h = int(26 * self.font_scale)
        y4 = int(100 * self.font_scale) + (btn_h + 6) * 3
        hdr_y = y4 + btn_h + int(6 * self.font_scale)
        
        if reset_needed:
            lbl_warn = self.font_subtitle.render("* Target settings changed. Click Apply Settings.", True, COLOR_YELLOW)
            self.screen.blit(lbl_warn, (sb_x + pad_x, hdr_y))
            hdr_y += int(14 * self.font_scale)

        # 4. Parameters Adjusters Header
        txt_param_hdr = self.font_header.render("Parameters & Target Constraints", True, COLOR_ACCENT)
        self.screen.blit(txt_param_hdr, (sb_x + pad_x, hdr_y))

        # Draw parameter rows
        row_h = int(24 * self.font_scale)
        adjust_start_y = hdr_y + int(20 * self.font_scale)
        visible_adjusters = self.get_visible_adjusters()
        
        for idx, adj in enumerate(visible_adjusters):
            key = adj["key"]
            if key == "grid_size":
                val = self.target_grid_size
            elif key == "nodes":
                val = self.target_num_nodes
            elif key == "aps":
                val = self.target_num_aps
            elif key == "show_links":
                val = self.show_links
            elif key == "throttle_speed":
                val = self.throttle_speed
            else:
                val = self.ga_config[key]
                
            y_pos = adjust_start_y + idx * row_h
            adj["rect_val"].y = y_pos
            if adj["type"] == "slider":
                adj["rect_minus"].y = y_pos
                adj["rect_plus"].y = y_pos
            else:
                adj["rect_toggle"].y = y_pos
            
            lbl_name = self.font_body.render(adj["name"], True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_name, (sb_x + pad_x, y_pos + 1))
            
            is_active = (self.active_input_key == key)
            val_bg_color = (45, 55, 75) if is_active else COLOR_BG
            pygame.draw.rect(self.screen, val_bg_color, adj["rect_val"], border_radius=4)
            if is_active or adj["rect_val"].collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_val"], 1, border_radius=4)
                
            if is_active:
                cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
                disp_val = self.placeholder_text if self.input_text == "" else self.input_text
                disp_color = COLOR_TEXT_MUTED if self.input_text == "" else COLOR_ACCENT
                lbl_val = self.font_mono.render(disp_val + cursor, True, disp_color)
            else:
                disp_val = adj["fmt"](val)
                lbl_val = self.font_mono.render(disp_val, True, COLOR_TEXT)
                
            self.screen.blit(lbl_val, (adj["rect_val"].x + 5, y_pos + 2))
            
            if adj["type"] == "slider":
                pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_minus"], border_radius=4)
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_minus"], 1, border_radius=4)
                lbl_minus = self.font_body.render("-", True, COLOR_TEXT)
                self.screen.blit(lbl_minus, (adj["rect_minus"].centerx - lbl_minus.get_width()//2, adj["rect_minus"].centery - lbl_minus.get_height()//2 - 1))
                if adj["rect_minus"].collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, COLOR_WHITE, adj["rect_minus"], 1, border_radius=4)
                
                pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_plus"], border_radius=4)
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_plus"], 1, border_radius=4)
                lbl_plus = self.font_body.render("+", True, COLOR_TEXT)
                self.screen.blit(lbl_plus, (adj["rect_plus"].centerx - lbl_plus.get_width()//2, adj["rect_plus"].centery - lbl_plus.get_height()//2))
                if adj["rect_plus"].collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, COLOR_WHITE, adj["rect_plus"], 1, border_radius=4)
            else:
                toggle_btn_color = COLOR_GREEN if val else COLOR_RED
                pygame.draw.rect(self.screen, toggle_btn_color, adj["rect_toggle"], border_radius=4)
                lbl_toggle = self.font_body.render("Toggle", True, COLOR_WHITE)
                self.screen.blit(lbl_toggle, (adj["rect_toggle"].centerx - lbl_toggle.get_width()//2, adj["rect_toggle"].centery - lbl_toggle.get_height()//2))
                if adj["rect_toggle"].collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, COLOR_WHITE, adj["rect_toggle"], 1, border_radius=4)

        # 5. Model Cost Breakdown
        breakdown_y = adjust_start_y + len(visible_adjusters) * row_h + int(4 * self.font_scale)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (sb_x + pad_x, breakdown_y), (right_edge, breakdown_y), 1)
        
        txt_costs = self.font_header.render(f"Best Model Cost: {int(best_ind.total_cost):,}", True, COLOR_ACCENT)
        self.screen.blit(txt_costs, (sb_x + pad_x, breakdown_y + 6))
        
        c_power = f"Power cost: {int(best_ind.power_cost * self.ga.power_weight):,}"
        c_overlap = f"Overlap penalty: {int(best_ind.overlap_cost * self.ga.overlap_weight):,}"
        c_cap = f"Capacity penalty: {int(best_ind.capacity_cost * self.ga.capacity_weight):,}"
        
        line_step = int(16 * self.font_scale)
        self.screen.blit(self.font_body.render(c_power, True, COLOR_TEXT_MUTED), (sb_x + pad_x, breakdown_y + 6 + line_step))
        self.screen.blit(self.font_body.render(c_overlap, True, COLOR_TEXT_MUTED), (sb_x + pad_x, breakdown_y + 6 + line_step * 2))
        self.screen.blit(self.font_body.render(c_cap, True, COLOR_TEXT_MUTED), (sb_x + pad_x, breakdown_y + 6 + line_step * 3))
        
        # 6. Dynamic Router load bars layout
        allocations_y = breakdown_y + 6 + line_step * 4 + int(6 * self.font_scale)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (sb_x + pad_x, allocations_y), (right_edge, allocations_y), 1)
        
        txt_routers = self.font_header.render("Router Load Allocations", True, COLOR_ACCENT)
        self.screen.blit(txt_routers, (sb_x + pad_x, allocations_y + 4))
        
        col_w = avail_w // 3
        bar_width = col_w - 20
        bar_height = 5
        ap_cap = self.ga.ap_capacity
        num_aps = self.ga.num_aps
        
        row_step_ap = int(24 * self.font_scale)
        for i in range(num_aps):
            col = i % 3
            row = i // 3
            
            x_bar = sb_x + pad_x + col * col_w
            y_bar = allocations_y + int(22 * self.font_scale) + row * row_step_ap
            
            load = best_ind.ap_loads[i] if i < len(best_ind.ap_loads) else 0
            percentage = min(1.0, load / ap_cap) if ap_cap > 0 else 0
            
            if load > ap_cap:
                bar_color = COLOR_RED
            elif load > ap_cap * 0.8:
                bar_color = COLOR_ORANGE
            else:
                bar_color = AP_COLORS[i % len(AP_COLORS)]
                
            lbl_ap = self.font_mono.render(f"R{i+1}:{load:02d}/{ap_cap}", True, COLOR_TEXT)
            self.screen.blit(lbl_ap, (x_bar, y_bar))
            
            pygame.draw.rect(self.screen, (20, 30, 45), (x_bar, y_bar + int(14 * self.font_scale), bar_width, bar_height), border_radius=2)
            if percentage > 0:
                pygame.draw.rect(self.screen, bar_color, (x_bar, y_bar + int(14 * self.font_scale), int(bar_width * percentage), bar_height), border_radius=2)

        # 7. Mini Cost History Graph
        chart_line_y = max(allocations_y + int(24 * self.font_scale) + ((num_aps + 2) // 3) * row_step_ap + 6, self.screen_height - int(90 * self.font_scale))
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (sb_x + pad_x, chart_line_y), (right_edge, chart_line_y), 1)
        
        chart_x = sb_x + pad_x
        chart_y = chart_line_y + 6
        chart_w = avail_w
        chart_h = min(int(70 * self.font_scale), self.screen_height - chart_y - 10)
        
        if chart_h > 20:
            pygame.draw.rect(self.screen, COLOR_GRID_BG, (chart_x, chart_y, chart_w, chart_h), border_radius=6)
            
            full_hist = self.ga.best_history
            hist = full_hist[-150:]
            avg_hist = self.ga.avg_history[-150:]
            
            if len(hist) > 1:
                max_val = max(max(hist), max(avg_hist))
                min_val = min(min(hist), min(avg_hist))
                val_range = max_val - min_val if max_val != min_val else 1.0
                
                pts_best = []
                pts_avg = []
                
                for gen_idx in range(len(hist)):
                    px = chart_x + int((gen_idx / (len(hist) - 1)) * chart_w)
                    py_best = chart_y + chart_h - int(((hist[gen_idx] - min_val) / val_range) * (chart_h - 6) + 3)
                    py_avg = chart_y + chart_h - int(((avg_hist[gen_idx] - min_val) / val_range) * (chart_h - 6) + 3)
                    
                    pts_best.append((px, py_best))
                    pts_avg.append((px, py_avg))
                    
                mid_y = chart_y + chart_h // 2
                pygame.draw.line(self.screen, (40, 50, 70), (chart_x + 5, mid_y), (chart_x + chart_w - 5, mid_y), 1)
                
                pygame.draw.lines(self.screen, COLOR_CHART_AVG, False, pts_avg, 1)
                pygame.draw.lines(self.screen, COLOR_ACCENT, False, pts_best, 2)
                
                label_prefix = f"Cost (last {len(hist)})" if len(full_hist) > 150 else "Cost History"
                cost_lbl = self.font_subtitle.render(f"{label_prefix}: {int(min_val):,} to {int(max_val):,}", True, COLOR_TEXT_MUTED)
                self.screen.blit(cost_lbl, (chart_x + 5, chart_y + 1))
            else:
                lbl_empty = self.font_subtitle.render("Waiting for history data...", True, COLOR_TEXT_MUTED)
                self.screen.blit(lbl_empty, (chart_x + 10, chart_y + chart_h//2 - 6))

if __name__ == '__main__':
    app = PygameApp()
    app.run()
