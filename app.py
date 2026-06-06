#!/usr/bin/env python3
import pygame
import sys
import time
import math
from typing import Dict, Any
from genetic_algorithm import generate_devices, GeneticAlgorithm, Individual, Device

# Initial default configuration (configurable via code constants)
DEFAULT_GRID_SIZE = 100
DEFAULT_NUM_DEVICES = 100
DEFAULT_NUM_APS = 5

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
        
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Wireless AP Genetic Algorithm Optimizer")
        self.clock = pygame.time.Clock()
        
        # Load premium system fonts safely
        self.font_title = pygame.font.SysFont("Trebuchet MS", 21, bold=True)
        self.font_subtitle = pygame.font.SysFont("Verdana", 11, italic=True)
        self.font_header = pygame.font.SysFont("Trebuchet MS", 15, bold=True)
        self.font_body = pygame.font.SysFont("Verdana", 12)
        self.font_mono = pygame.font.SysFont("Courier New", 12, bold=True)
        
        # Initialize active setup from constants
        self.current_seed = 42
        self.devices = generate_devices(num_devices=DEFAULT_NUM_DEVICES, grid_size=DEFAULT_GRID_SIZE, seed=self.current_seed)
        
        # Calculate initial capacity dynamically: ceil(total nodes / total APs)
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
        self.target_num_nodes = DEFAULT_NUM_DEVICES
        self.target_num_aps = DEFAULT_NUM_APS
        self.show_links = True
        self.throttle_speed = True
        
        # Textbox Input variables (for typing in numbers directly with placeholder support)
        self.active_input_key = None
        self.input_text = ""
        self.placeholder_text = ""
        
        # Execution control
        self.is_running = False
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0.0
        
        # Layouts
        self.rect_grid = pygame.Rect(0, 0, 800, 800)
        self.rect_sidebar = pygame.Rect(800, 0, 400, 800)
        
        # Dynamic 3-row layout for 6 control buttons
        self.btn_play = pygame.Rect(820, 105, 170, 26)
        self.btn_step = pygame.Rect(1010, 105, 170, 26)
        self.btn_step10 = pygame.Rect(820, 136, 170, 26)
        self.btn_reset_ga = pygame.Rect(1010, 136, 170, 26)      # Resets population, keeps nodes
        self.btn_rotate_nodes = pygame.Rect(820, 167, 170, 26)    # Regenerates nodes with new seed
        self.btn_apply_tgt = pygame.Rect(1010, 167, 170, 26)       # Applies target size/ap configs
        
        # 11 Parameter Adjusters
        self.adjusters = [
            {"name": "AP Radius", "key": "ap_radius", "fmt": lambda x: f"{x:.1f}", "step": 1.0, "type": "slider"},
            {"name": "Mutation Rate", "key": "mutation_rate", "fmt": lambda x: f"{x:.3f}", "step": 0.01, "type": "slider"},
            {"name": "Crossover Rate", "key": "crossover_rate", "fmt": lambda x: f"{x:.3f}", "step": 0.05, "type": "slider"},
            {"name": "Power Wt", "key": "power_weight", "fmt": lambda x: f"{x:.1f}", "step": 0.1, "type": "slider"},
            {"name": "Overlap Wt", "key": "overlap_weight", "fmt": lambda x: f"{x:.0f}", "step": 10.0, "type": "slider"},
            {"name": "Capacity Wt", "key": "capacity_weight", "fmt": lambda x: f"{x:.0f}", "step": 50.0, "type": "slider"},
            {"name": "Grid Size (Tgt)", "key": "grid_size", "fmt": lambda x: f"{x:d}", "step": 50, "type": "slider"},
            {"name": "Nodes (Tgt)", "key": "nodes", "fmt": lambda x: f"{x:d}", "step": 10, "type": "slider"},
            {"name": "AP Count (Tgt)", "key": "aps", "fmt": lambda x: f"{x:d}", "step": 1, "type": "slider"},
            {"name": "Show Links", "key": "show_links", "fmt": lambda x: "ON" if x else "OFF", "step": 0, "type": "toggle"},
            {"name": "Throttle Speed", "key": "throttle_speed", "fmt": lambda x: "ON" if x else "OFF", "step": 0, "type": "toggle"}
        ]
        
        # Position adjusters and register value-box zones for number clicks
        start_y = 222
        for idx, adj in enumerate(self.adjusters):
            y_pos = start_y + idx * 23
            if adj["type"] == "slider":
                adj["rect_minus"] = pygame.Rect(1110, y_pos, 30, 19)
                adj["rect_plus"] = pygame.Rect(1150, y_pos, 30, 19)
                adj["rect_val"] = pygame.Rect(965, y_pos, 140, 19)
            else: # toggle
                adj["rect_toggle"] = pygame.Rect(1110, y_pos, 70, 19)
                adj["rect_val"] = pygame.Rect(965, y_pos, 140, 19)

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
        """Changes the mouse cursor to a hand when hovering over interactive components."""
        pos = pygame.mouse.get_pos()
        hovering = False
        
        # Check main buttons
        for btn in [self.btn_play, self.btn_step, self.btn_step10, self.btn_reset_ga, self.btn_rotate_nodes, self.btn_apply_tgt]:
            if btn.collidepoint(pos):
                hovering = True
                break
                
        if not hovering:
            # Check adjusters
            for adj in self.adjusters:
                if adj["rect_val"].collidepoint(pos):
                    hovering = True
                    break
                if adj["type"] == "slider":
                    if adj["rect_minus"].collidepoint(pos) or adj["rect_plus"].collidepoint(pos):
                        hovering = True
                        break
                elif adj["type"] == "toggle":
                    if adj["rect_toggle"].collidepoint(pos):
                        hovering = True
                        break
                        
        if hovering:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
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
        """Dynamically calculates maximum bounds for parameter ranges to prevent crash/nonsense values."""
        if key == "ap_radius":
            return float(self.ga.grid_size)  # Cap at active grid size
        elif key == "nodes":
            return float(self.target_grid_size * self.target_grid_size)  # Cap at unique cells grid limit
        elif key == "aps":
            return 15.0  # Safe cap to fit sidebar columns load bars and AP color bounds
        elif key == "grid_size":
            return 1000.0  # Limit grid coordinate sizes to prevent rendering overload
        elif key in ["mutation_rate", "crossover_rate"]:
            return 1.0   # Cap probabilities strictly at 1.0 (100%)
        elif key in ["power_weight", "overlap_weight", "capacity_weight"]:
            return 10000.0  # Practical weight ceiling
        return float('inf')

    def get_min_bound(self, key: str) -> float:
        """Dynamically calculates minimum bounds for parameter ranges."""
        if key in ["grid_size", "nodes", "aps"]:
            return 10.0 if key == "grid_size" else 1.0
        elif key == "ap_radius":
            return 0.5
        return 0.0

    def apply_input_value(self):
        """Attempts to parse and apply the typed value from textbox."""
        if not self.active_input_key:
            return
        
        val_text = self.input_text.replace('%', '').strip()
        if not val_text:
            self.active_input_key = None
            return  # Empty string: keep previous value
            
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
                self.target_num_aps = int(val)
            else:
                self.update_param(key, val)
                
        except ValueError:
            pass
            
        self.active_input_key = None

    def perform_apply_and_reset(self, new_seed: bool = False):
        """Regenerates devices and initializes a new GA using target values."""
        self.is_running = False
        if new_seed:
            self.current_seed += 1
            
        # Generate new devices
        self.devices = generate_devices(
            num_devices=self.target_num_nodes,
            grid_size=self.target_grid_size,
            seed=self.current_seed
        )
        
        # Recalculate dynamic capacity: ceil(nodes / AP count)
        dynamic_capacity = int(math.ceil(self.target_num_nodes / self.target_num_aps))
        self.ga_config["ap_capacity"] = dynamic_capacity
        
        # Re-initialize Genetic Algorithm
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

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            # Direct numeric input typing loop (blocks hotkeys while editing)
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
                        for adj in self.adjusters:
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
                                self.input_text = ""  # Start empty: allow typing immediately
                                clicked_another = True
                                break
                        if not clicked_another:
                            self.apply_input_value()

            # Normal control hotkey mode
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_running = not self.is_running
                elif event.key == pygame.K_s:
                    self.is_running = False
                    self.ga.step()
                elif event.key == pygame.K_r:
                    self.is_running = False
                    self.ga.initialize_population()
                elif event.key == pygame.K_n:
                    self.perform_apply_and_reset(new_seed=True)
                elif event.key == pygame.K_a:
                    self.perform_apply_and_reset(new_seed=False)
                elif event.key == pygame.K_UP:
                    self.step_delay = max(0.005, self.step_delay - 0.01)
                elif event.key == pygame.K_DOWN:
                    self.step_delay = min(0.5, self.step_delay + 0.01)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = event.pos
                    
                    # 1. Action buttons
                    if self.btn_play.collidepoint(pos):
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
                    elif self.btn_rotate_nodes.collidepoint(pos):
                        self.perform_apply_and_reset(new_seed=True)
                    elif self.btn_apply_tgt.collidepoint(pos):
                        self.perform_apply_and_reset(new_seed=False)
                        
                    # 2. Adjusters buttons / Input activation
                    for adj in self.adjusters:
                        key = adj["key"]
                        if adj["rect_val"].collidepoint(pos):
                            # Clicked value to type directly
                            self.active_input_key = key
                            val = self.target_grid_size if key == "grid_size" else (
                                  self.target_num_nodes if key == "nodes" else (
                                  self.target_num_aps if key == "aps" else (
                                  self.show_links if key == "show_links" else (
                                  self.throttle_speed if key == "throttle_speed" else self.ga_config[key]
                            ))))
                            self.placeholder_text = adj["fmt"](val) if key not in ["show_links", "throttle_speed"] else ""
                            self.input_text = ""  # Start empty: allow typing immediately
                            break
                            
                        if adj["type"] == "toggle":
                            if adj["rect_toggle"].collidepoint(pos):
                                if key == "show_links":
                                    self.show_links = not self.show_links
                                elif key == "throttle_speed":
                                    self.throttle_speed = not self.throttle_speed
                        else: # slider minus/plus click
                            val = self.target_grid_size if key == "grid_size" else (
                                  self.target_num_nodes if key == "nodes" else (
                                  self.target_num_aps if key == "aps" else self.ga_config[key]
                            )
                            )
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
                                    self.target_num_aps = int(new_val)
                                else:
                                    self.update_param(key, new_val)
                            elif adj["rect_plus"].collidepoint(pos):
                                new_val = min(max_b, val + adj["step"])
                                if key == "grid_size":
                                    self.target_grid_size = int(new_val)
                                elif key == "nodes":
                                    self.target_num_nodes = int(new_val)
                                elif key == "aps":
                                    self.target_num_aps = int(new_val)
                                else:
                                    self.update_param(key, new_val)

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
        pygame.draw.rect(self.screen, COLOR_GRID_BG, self.rect_grid)
        
        best_ind = self.ga.get_best_individual()
        grid_size = self.ga.grid_size
        scale = 800.0 / grid_size  # Dynamic scale mapping grid to 800x800 screen space
        
        # 1. Draw dynamic background grid corresponding to real scale
        if scale >= 5.0:
            step = 1       # Draw every single unit line
        elif scale >= 2.0:
            step = 5       # Draw line every 5 units
        elif scale * 10 >= 4.0:
            step = 10      # Draw line every 10 units
        elif scale * 50 >= 4.0:
            step = 50      # Draw line every 50 units
        else:
            step = 100     # Draw line every 100 units
            
        for i in range(step, grid_size, step):
            screen_pos = int(i * scale)
            pygame.draw.line(self.screen, (20, 25, 40), (screen_pos, 0), (screen_pos, 800), 1)
            pygame.draw.line(self.screen, (20, 25, 40), (0, screen_pos), (800, screen_pos), 1)
            
        # 2. Draw AP coverage circles (translucent overlay)
        overlay = pygame.Surface((800, 800), pygame.SRCALPHA)
        ap_radius_screen = int(self.ga.ap_radius * scale)
        for idx, ap in enumerate(best_ind.aps):
            cx, cy = int(ap[0] * scale), int(ap[1] * scale)
            color = AP_COLORS[idx % len(AP_COLORS)]
            pygame.draw.circle(overlay, color + (20,), (cx, cy), ap_radius_screen)
            pygame.draw.circle(overlay, color + (50,), (cx, cy), ap_radius_screen, 1)
        self.screen.blit(overlay, (0, 0))
        
        # 3. Draw connection paths (only if show_links toggle is ON)
        if self.show_links:
            for dev in self.devices:
                ap_idx = best_ind.device_assignments[dev.id]
                if 0 <= ap_idx < len(best_ind.aps):
                    ap = best_ind.aps[ap_idx]
                    ap_color = AP_COLORS[ap_idx % len(AP_COLORS)]
                    start_pos = (int(dev.x * scale), int(dev.y * scale))
                    end_pos = (int(ap[0] * scale), int(ap[1] * scale))
                    pygame.draw.line(self.screen, ap_color + (45,), start_pos, end_pos, 1)
                
        # 4. Draw Device Nodes
        for dev in self.devices:
            ap_idx = best_ind.device_assignments[dev.id]
            color = AP_COLORS[ap_idx % len(AP_COLORS)] if (0 <= ap_idx < len(best_ind.aps)) else COLOR_TEXT_MUTED
            pos = (int(dev.x * scale), int(dev.y * scale))
            pygame.draw.circle(self.screen, color, pos, 3)
            pygame.draw.circle(self.screen, COLOR_GRID_BG, pos, 3, 1)
            
        # 5. Draw Access Point Centers
        for idx, ap in enumerate(best_ind.aps):
            cx, cy = int(ap[0] * scale), int(ap[1] * scale)
            color = AP_COLORS[idx % len(AP_COLORS)]
            load = best_ind.ap_loads[idx]
            is_overcapacity = load > self.ga.ap_capacity
            
            if is_overcapacity:
                pulse_r = 14 + int(3 * math.sin(time.time() * 10))
                pygame.draw.circle(self.screen, COLOR_RED, (cx, cy), pulse_r, 2)
                
            pygame.draw.circle(self.screen, color, (cx, cy), 9)
            pygame.draw.circle(self.screen, COLOR_WHITE, (cx, cy), 9, 2)
            pygame.draw.circle(self.screen, COLOR_WHITE, (cx, cy), 3)
            
            lbl = self.font_mono.render(str(idx + 1), True, COLOR_WHITE)
            self.screen.blit(lbl, (cx - lbl.get_width()//2, cy - 23))

    def draw_sidebar_panel(self):
        pygame.draw.rect(self.screen, COLOR_BG, self.rect_sidebar)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (800, 0), (800, 800), 2)
        
        best_ind = self.ga.get_best_individual()
        reset_needed = self.is_reset_required()
        mouse_pos = pygame.mouse.get_pos()
        
        # 1. Header Title (Verdana/Trebuchet Fonts)
        title = self.font_title.render("Wireless GA Optimizer", True, COLOR_ACCENT)
        self.screen.blit(title, (820, 15))
        
        # Render active parameters status
        sub_text = f"Active: Grid {self.ga.grid_size}x{self.ga.grid_size} | Nodes: {len(self.devices)} | APs: {self.ga.num_aps}"
        sub = self.font_subtitle.render(sub_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(sub, (820, 43))
        
        seed_txt = f"Seed: {self.current_seed}"
        lbl_seed = self.font_subtitle.render(seed_txt, True, COLOR_ACCENT)
        self.screen.blit(lbl_seed, (1120, 43))
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 65), (1180, 65), 1)
        
        # 2. Status details
        txt_gen = self.font_header.render(f"Generation: {self.ga.generation}", True, COLOR_TEXT)
        self.screen.blit(txt_gen, (820, 75))
        
        status_text = "RUNNING" if self.is_running else "PAUSED"
        status_color = COLOR_GREEN if self.is_running else COLOR_ORANGE
        txt_status = self.font_header.render(status_text, True, status_color)
        self.screen.blit(txt_status, (1100, 75))
        
        # 3. Play / Action Buttons with hover highlighting border
        play_btn_color = COLOR_ORANGE if self.is_running else COLOR_GREEN
        play_btn_lbl = "Pause (Space)" if self.is_running else "Play (Space)"
        pygame.draw.rect(self.screen, play_btn_color, self.btn_play, border_radius=6)
        lbl_play = self.font_header.render(play_btn_lbl, True, COLOR_WHITE)
        self.screen.blit(lbl_play, (self.btn_play.centerx - lbl_play.get_width()//2, self.btn_play.centery - lbl_play.get_height()//2))
        if self.btn_play.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_play, 2, border_radius=6)
        
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step, 1, border_radius=6)
        lbl_step = self.font_header.render("Step 1 (S)", True, COLOR_TEXT)
        self.screen.blit(lbl_step, (self.btn_step.centerx - lbl_step.get_width()//2, self.btn_step.centery - lbl_step.get_height()//2))
        if self.btn_step.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_step, 2, border_radius=6)
        
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step10, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step10, 1, border_radius=6)
        lbl_step10 = self.font_header.render("Step 10", True, COLOR_TEXT)
        self.screen.blit(lbl_step10, (self.btn_step10.centerx - lbl_step10.get_width()//2, self.btn_step10.centery - lbl_step10.get_height()//2))
        if self.btn_step10.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_step10, 2, border_radius=6)
        
        # Reset AP Positions (restarts search, keeps nodes)
        pygame.draw.rect(self.screen, COLOR_RED, self.btn_reset_ga, border_radius=6)
        lbl_reset_ga = self.font_header.render("Reset APs (R)", True, COLOR_WHITE)
        self.screen.blit(lbl_reset_ga, (self.btn_reset_ga.centerx - lbl_reset_ga.get_width()//2, self.btn_reset_ga.centery - lbl_reset_ga.get_height()//2))
        if self.btn_reset_ga.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_reset_ga, 2, border_radius=6)
        
        # Rotate Nodes (new random device layout, seed increment)
        pygame.draw.rect(self.screen, COLOR_CHART_AVG, self.btn_rotate_nodes, border_radius=6)
        lbl_rot = self.font_header.render("New Devices (N)", True, COLOR_WHITE)
        self.screen.blit(lbl_rot, (self.btn_rotate_nodes.centerx - lbl_rot.get_width()//2, self.btn_rotate_nodes.centery - lbl_rot.get_height()//2))
        if self.btn_rotate_nodes.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_rotate_nodes, 2, border_radius=6)
        
        # Apply Tgt Configurations
        apply_btn_color = COLOR_ORANGE if reset_needed else COLOR_PANEL_BG
        apply_text_color = COLOR_WHITE if reset_needed else COLOR_TEXT
        pygame.draw.rect(self.screen, apply_btn_color, self.btn_apply_tgt, border_radius=6)
        if not reset_needed:
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_apply_tgt, 1, border_radius=6)
        lbl_apply = self.font_header.render("Apply Settings (A)", True, apply_text_color)
        self.screen.blit(lbl_apply, (self.btn_apply_tgt.centerx - lbl_apply.get_width()//2, self.btn_apply_tgt.centery - lbl_apply.get_height()//2))
        if self.btn_apply_tgt.collidepoint(mouse_pos):
            pygame.draw.rect(self.screen, COLOR_WHITE, self.btn_apply_tgt, 2, border_radius=6)
        
        # Display warning label if apply settings is pending
        if reset_needed:
            lbl_warn = self.font_subtitle.render("* Target settings changed. Click Apply Settings.", True, COLOR_YELLOW)
            self.screen.blit(lbl_warn, (820, 196))
            
        # 4. Parameters Adjusters list
        txt_param_hdr = self.font_header.render("Parameters & Target Size Constraints", True, COLOR_ACCENT)
        self.screen.blit(txt_param_hdr, (820, 196 if not reset_needed else 208))
        
        lbl_note = self.font_subtitle.render("(Click values to type numbers directly)", True, COLOR_TEXT_MUTED)
        self.screen.blit(lbl_note, (820, 212 if not reset_needed else 224))
        
        # Draw parameter rows
        adjust_start_y = 225 if not reset_needed else 237
        for idx, adj in enumerate(self.adjusters):
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
                
            y_pos = adjust_start_y + idx * 23
            adj["rect_val"].y = y_pos
            if adj["type"] == "slider":
                adj["rect_minus"].y = y_pos
                adj["rect_plus"].y = y_pos
            else:
                adj["rect_toggle"].y = y_pos
            
            lbl_name = self.font_body.render(adj["name"], True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_name, (820, y_pos + 1))
            
            # Value Box Background
            is_active = (self.active_input_key == key)
            val_bg_color = (45, 55, 75) if is_active else COLOR_BG
            pygame.draw.rect(self.screen, val_bg_color, adj["rect_val"], border_radius=4)
            if is_active:
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_val"], 1, border_radius=4)
            elif adj["rect_val"].collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_val"], 1, border_radius=4)
                
            # Text rendering (supports greyed out placeholder text)
            if is_active:
                cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
                if self.input_text == "":
                    # Draw placeholder text in muted grey
                    disp_val = self.placeholder_text
                    disp_color = COLOR_TEXT_MUTED
                else:
                    disp_val = self.input_text
                    disp_color = COLOR_ACCENT
                lbl_val = self.font_mono.render(disp_val + cursor, True, disp_color)
            else:
                disp_val = adj["fmt"](val)
                lbl_val = self.font_mono.render(disp_val, True, COLOR_TEXT)
                
            self.screen.blit(lbl_val, (adj["rect_val"].x + 5, y_pos + 2))
            
            if adj["type"] == "slider":
                # Minus
                pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_minus"], border_radius=4)
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_minus"], 1, border_radius=4)
                lbl_minus = self.font_body.render("-", True, COLOR_TEXT)
                self.screen.blit(lbl_minus, (adj["rect_minus"].centerx - lbl_minus.get_width()//2, adj["rect_minus"].centery - lbl_minus.get_height()//2 - 2))
                if adj["rect_minus"].collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, COLOR_WHITE, adj["rect_minus"], 1, border_radius=4)
                
                # Plus
                pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_plus"], border_radius=4)
                pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_plus"], 1, border_radius=4)
                lbl_plus = self.font_body.render("+", True, COLOR_TEXT)
                self.screen.blit(lbl_plus, (adj["rect_plus"].centerx - lbl_plus.get_width()//2, adj["rect_plus"].centery - lbl_plus.get_height()//2 - 1))
                if adj["rect_plus"].collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, COLOR_WHITE, adj["rect_plus"], 1, border_radius=4)
            else: # toggle
                toggle_btn_color = COLOR_GREEN if val else COLOR_RED
                pygame.draw.rect(self.screen, toggle_btn_color, adj["rect_toggle"], border_radius=4)
                lbl_toggle = self.font_body.render("Toggle", True, COLOR_WHITE)
                self.screen.blit(lbl_toggle, (adj["rect_toggle"].centerx - lbl_toggle.get_width()//2, adj["rect_toggle"].centery - lbl_toggle.get_height()//2 - 1))
                if adj["rect_toggle"].collidepoint(mouse_pos):
                    pygame.draw.rect(self.screen, COLOR_WHITE, adj["rect_toggle"], 1, border_radius=4)
            
        # 5. Model Cost Breakdown (Dynamic position)
        breakdown_y = adjust_start_y + len(self.adjusters) * 23 + 3
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, breakdown_y), (1180, breakdown_y), 1)
        
        txt_costs = self.font_header.render(f"Best Model Cost: {int(best_ind.total_cost):,}", True, COLOR_ACCENT)
        self.screen.blit(txt_costs, (820, breakdown_y + 8))
        
        c_power = f"Power cost: {int(best_ind.power_cost * self.ga.power_weight):,}"
        c_overlap = f"Overlap penalty: {int(best_ind.overlap_cost * self.ga.overlap_weight):,}"
        c_cap = f"Capacity penalty: {int(best_ind.capacity_cost * self.ga.capacity_weight):,}"
        
        self.screen.blit(self.font_body.render(c_power, True, COLOR_TEXT_MUTED), (820, breakdown_y + 26))
        self.screen.blit(self.font_body.render(c_overlap, True, COLOR_TEXT_MUTED), (820, breakdown_y + 42))
        self.screen.blit(self.font_body.render(c_cap, True, COLOR_TEXT_MUTED), (820, breakdown_y + 58))
        
        # 6. Dynamic Router load bars layout
        allocations_y = breakdown_y + 76
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, allocations_y), (1180, allocations_y), 1)
        
        txt_routers = self.font_header.render("Router Load Allocations", True, COLOR_ACCENT)
        self.screen.blit(txt_routers, (820, allocations_y + 6))
        
        bar_width = 100
        bar_height = 5
        ap_cap = self.ga.ap_capacity
        num_aps = self.ga.num_aps
        
        for i in range(num_aps):
            col = i % 3
            row = i // 3
            
            x_bar = 820 + col * 123
            y_bar = allocations_y + 28 + row * 26
            
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
            
            pygame.draw.rect(self.screen, (20, 30, 45), (x_bar, y_bar + 13, bar_width, bar_height), border_radius=2)
            if percentage > 0:
                pygame.draw.rect(self.screen, bar_color, (x_bar, y_bar + 13, int(bar_width * percentage), bar_height), border_radius=2)

        # 7. Mini Cost History Graph with Guide Line
        chart_line_y = 720
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, chart_line_y), (1180, chart_line_y), 1)
        
        chart_x = 820
        chart_y = 728
        chart_w = 360
        chart_h = 60
        
        pygame.draw.rect(self.screen, COLOR_GRID_BG, (chart_x, chart_y, chart_w, chart_h), border_radius=6)
        
        hist = self.ga.best_history
        avg_hist = self.ga.avg_history
        
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
                
            # Draw mid guide line (median line)
            mid_y = chart_y + chart_h // 2
            pygame.draw.line(self.screen, (40, 50, 70), (chart_x + 5, mid_y), (chart_x + chart_w - 5, mid_y), 1)
            
            pygame.draw.lines(self.screen, COLOR_CHART_AVG, False, pts_avg, 1)
            pygame.draw.lines(self.screen, COLOR_ACCENT, False, pts_best, 2)
            
            cost_lbl = self.font_subtitle.render(f"Cost: {int(min_val):,} to {int(max_val):,}", True, COLOR_TEXT_MUTED)
            self.screen.blit(cost_lbl, (chart_x + 5, chart_y + 1))
        else:
            lbl_empty = self.font_subtitle.render("Waiting for history data...", True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_empty, (chart_x + 10, chart_y + 22))

if __name__ == '__main__':
    app = PygameApp()
    app.run()
