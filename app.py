#!/usr/bin/env python3
import pygame
import sys
import time
import math
from typing import Dict, Any
from genetic_algorithm import generate_devices, GeneticAlgorithm, Individual, Device

# Initial default configuration (configurable via code constants)
DEFAULT_GRID_SIZE = 500
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
        
        # Load fonts safely
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_subtitle = pygame.font.SysFont("Arial", 13, italic=True)
        self.font_header = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 13)
        self.font_mono = pygame.font.SysFont("Courier", 12, bold=True)
        
        # Initialize active setup from constants
        self.devices = generate_devices(num_devices=DEFAULT_NUM_DEVICES, grid_size=DEFAULT_GRID_SIZE, seed=42)
        
        # Initial GA configurations
        self.ga_config = {
            "pop_size": 100,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8,
            "elitism_count": 2,
            "ap_radius": 25.0,
            "ap_capacity": 22,
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
        
        # Execution control
        self.is_running = False
        self.step_delay = 0.05  # seconds
        self.last_step_time = 0.0
        
        # Layouts
        self.rect_grid = pygame.Rect(0, 0, 800, 800)
        self.rect_sidebar = pygame.Rect(800, 0, 400, 800)
        
        # Buttons
        self.btn_play = pygame.Rect(820, 110, 170, 35)
        self.btn_step = pygame.Rect(1010, 110, 170, 35)
        self.btn_step10 = pygame.Rect(820, 155, 170, 35)
        self.btn_reset = pygame.Rect(1010, 155, 170, 35)
        
        # 10 Parameter Adjusters
        self.adjusters = [
            {"name": "AP Radius", "key": "ap_radius", "fmt": lambda x: f"{x:.1f}", "step": 1.0, "min": 5.0, "max": 100.0},
            {"name": "AP Capacity", "key": "ap_capacity", "fmt": lambda x: f"{x:d}", "step": 1, "min": 5, "max": 100},
            {"name": "Mutation Rate", "key": "mutation_rate", "fmt": lambda x: f"{x*100:.0f}%", "step": 0.01, "min": 0.01, "max": 0.5},
            {"name": "Crossover Rate", "key": "crossover_rate", "fmt": lambda x: f"{x*100:.0f}%", "step": 0.05, "min": 0.1, "max": 1.0},
            {"name": "Power Wt", "key": "power_weight", "fmt": lambda x: f"{x:.1f}", "step": 0.1, "min": 0.0, "max": 10.0},
            {"name": "Overlap Wt", "key": "overlap_weight", "fmt": lambda x: f"{x:.0f}", "step": 10.0, "min": 0.0, "max": 1000.0},
            {"name": "Capacity Wt", "key": "capacity_weight", "fmt": lambda x: f"{x:.0f}", "step": 50.0, "min": 0.0, "max": 2000.0},
            {"name": "Grid Size (Tgt)", "key": "grid_size", "fmt": lambda x: f"{x:d}", "step": 50, "min": 50, "max": 1000},
            {"name": "Nodes (Tgt)", "key": "nodes", "fmt": lambda x: f"{x:d}", "step": 10, "min": 10, "max": 500},
            {"name": "AP Count (Tgt)", "key": "aps", "fmt": lambda x: f"{x:d}", "step": 1, "min": 1, "max": 15},
        ]
        
        # Position adjusters
        start_y = 210
        for idx, adj in enumerate(self.adjusters):
            y_pos = start_y + idx * 24
            adj["rect_minus"] = pygame.Rect(1110, y_pos, 30, 20)
            adj["rect_plus"] = pygame.Rect(1150, y_pos, 30, 20)

    def run(self):
        while True:
            self.handle_events()
            self.update_logic()
            self.draw()
            self.clock.tick(60)

    def update_logic(self):
        if self.is_running:
            current_time = time.time()
            if current_time - self.last_step_time >= self.step_delay:
                self.ga.step()
                self.last_step_time = current_time

    def is_reset_required(self) -> bool:
        """Checks if target setup parameters differ from active GA state."""
        return (
            self.target_grid_size != self.ga.grid_size or
            self.target_num_nodes != len(self.devices) or
            self.target_num_aps != self.ga.num_aps
        )

    def perform_apply_and_reset(self):
        """Regenerates devices and initializes a new GA using target values."""
        self.is_running = False
        # Generate new devices
        self.devices = generate_devices(
            num_devices=self.target_num_nodes,
            grid_size=self.target_grid_size,
            seed=42
        )
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
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_running = not self.is_running
                elif event.key == pygame.K_s:
                    self.is_running = False
                    self.ga.step()
                elif event.key == pygame.K_r:
                    self.perform_apply_and_reset()
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
                    elif self.btn_reset.collidepoint(pos):
                        self.perform_apply_and_reset()
                        
                    # 2. Adjusters buttons
                    for adj in self.adjusters:
                        key = adj["key"]
                        
                        # Handle target configuration changes separately
                        if key in ["grid_size", "nodes", "aps"]:
                            if adj["rect_minus"].collidepoint(pos):
                                if key == "grid_size":
                                    self.target_grid_size = max(adj["min"], self.target_grid_size - adj["step"])
                                elif key == "nodes":
                                    self.target_num_nodes = max(adj["min"], self.target_num_nodes - adj["step"])
                                elif key == "aps":
                                    self.target_num_aps = max(adj["min"], self.target_num_aps - adj["step"])
                            elif adj["rect_plus"].collidepoint(pos):
                                if key == "grid_size":
                                    self.target_grid_size = min(adj["max"], self.target_grid_size + adj["step"])
                                elif key == "nodes":
                                    self.target_num_nodes = min(adj["max"], self.target_num_nodes + adj["step"])
                                elif key == "aps":
                                    self.target_num_aps = min(adj["max"], self.target_num_aps + adj["step"])
                        else:
                            # Live parameter adjustment
                            val = self.ga_config[key]
                            if adj["rect_minus"].collidepoint(pos):
                                new_val = max(adj["min"], val - adj["step"])
                                self.update_param(key, new_val)
                            elif adj["rect_plus"].collidepoint(pos):
                                new_val = min(adj["max"], val + adj["step"])
                                self.update_param(key, new_val)

    def update_param(self, key: str, value: Any):
        if key in ["ap_capacity", "pop_size"]:
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
        
        # 1. Draw subtle grid coordinate lines dynamically
        if grid_size <= 100:
            step = 10
        elif grid_size <= 250:
            step = 25
        elif grid_size <= 500:
            step = 50
        else:
            step = 100
            
        for i in range(step, grid_size, step):
            screen_pos = int(i * scale)
            pygame.draw.line(self.screen, (20, 25, 40), (screen_pos, 0), (screen_pos, 800), 1)
            pygame.draw.line(self.screen, (20, 25, 40), (0, screen_pos), (800, screen_pos), 1)
            
        # 2. Draw AP coverage circles (translucent)
        overlay = pygame.Surface((800, 800), pygame.SRCALPHA)
        ap_radius_screen = int(self.ga.ap_radius * scale)
        for idx, ap in enumerate(best_ind.aps):
            cx, cy = int(ap[0] * scale), int(ap[1] * scale)
            color = AP_COLORS[idx % len(AP_COLORS)]
            # Semi-transparent coverage circle
            pygame.draw.circle(overlay, color + (20,), (cx, cy), ap_radius_screen)
            pygame.draw.circle(overlay, color + (50,), (cx, cy), ap_radius_screen, 1)
        self.screen.blit(overlay, (0, 0))
        
        # 3. Draw connection paths (device to AP)
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
        
        # 1. Header Title
        title = self.font_title.render("Wireless GA Optimizer", True, COLOR_ACCENT)
        self.screen.blit(title, (820, 15))
        
        # Render active parameters status
        sub_text = f"Active: Grid {self.ga.grid_size}x{self.ga.grid_size} | Nodes: {len(self.devices)} | APs: {self.ga.num_aps}"
        sub = self.font_subtitle.render(sub_text, True, COLOR_TEXT_MUTED)
        self.screen.blit(sub, (820, 45))
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 70), (1180, 70), 1)
        
        # 2. Status details
        txt_gen = self.font_header.render(f"Generation: {self.ga.generation}", True, COLOR_TEXT)
        self.screen.blit(txt_gen, (820, 80))
        
        status_text = "RUNNING" if self.is_running else "PAUSED"
        status_color = COLOR_GREEN if self.is_running else COLOR_ORANGE
        txt_status = self.font_header.render(status_text, True, status_color)
        self.screen.blit(txt_status, (1100, 80))
        
        # 3. Play / Action Buttons
        play_btn_color = COLOR_ORANGE if self.is_running else COLOR_GREEN
        play_btn_lbl = "Pause (Space)" if self.is_running else "Play (Space)"
        pygame.draw.rect(self.screen, play_btn_color, self.btn_play, border_radius=6)
        lbl_play = self.font_header.render(play_btn_lbl, True, COLOR_WHITE)
        self.screen.blit(lbl_play, (self.btn_play.centerx - lbl_play.get_width()//2, self.btn_play.centery - lbl_play.get_height()//2))
        
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step, 1, border_radius=6)
        lbl_step = self.font_header.render("Step 1 (S)", True, COLOR_TEXT)
        self.screen.blit(lbl_step, (self.btn_step.centerx - lbl_step.get_width()//2, self.btn_step.centery - lbl_step.get_height()//2))
        
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step10, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step10, 1, border_radius=6)
        lbl_step10 = self.font_header.render("Step 10", True, COLOR_TEXT)
        self.screen.blit(lbl_step10, (self.btn_step10.centerx - lbl_step10.get_width()//2, self.btn_step10.centery - lbl_step10.get_height()//2))
        
        reset_btn_color = COLOR_ORANGE if reset_needed else COLOR_RED
        reset_btn_lbl = "Apply & Reset" if reset_needed else "Reset (R)"
        pygame.draw.rect(self.screen, reset_btn_color, self.btn_reset, border_radius=6)
        lbl_reset = self.font_header.render(reset_btn_lbl, True, COLOR_WHITE)
        self.screen.blit(lbl_reset, (self.btn_reset.centerx - lbl_reset.get_width()//2, self.btn_reset.centery - lbl_reset.get_height()//2))
        
        # Display warning label if reset is pending
        if reset_needed:
            lbl_warn = self.font_subtitle.render("* Target parameters changed. Requires Reset.", True, COLOR_YELLOW)
            self.screen.blit(lbl_warn, (820, 192))
            
        # 4. Parameters Adjusters list
        txt_param_hdr = self.font_header.render("Parameters", True, COLOR_ACCENT)
        self.screen.blit(txt_param_hdr, (820, 192 if not reset_needed else 206)) # Shift slightly if warn visible
        
        for adj in self.adjusters:
            key = adj["key"]
            if key == "grid_size":
                val = self.target_grid_size
            elif key == "nodes":
                val = self.target_num_nodes
            elif key == "aps":
                val = self.target_num_aps
            else:
                val = self.ga_config[key]
                
            y_pos = adj["rect_minus"].y
            
            lbl_name = self.font_body.render(adj["name"], True, COLOR_TEXT_MUTED)
            lbl_val = self.font_mono.render(adj["fmt"](val), True, COLOR_TEXT)
            self.screen.blit(lbl_name, (820, y_pos + 1))
            self.screen.blit(lbl_val, (970, y_pos + 1))
            
            # Minus
            pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_minus"], border_radius=4)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_minus"], 1, border_radius=4)
            lbl_minus = self.font_body.render("-", True, COLOR_TEXT)
            self.screen.blit(lbl_minus, (adj["rect_minus"].centerx - lbl_minus.get_width()//2, adj["rect_minus"].centery - lbl_minus.get_height()//2 - 2))
            
            # Plus
            pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_plus"], border_radius=4)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_plus"], 1, border_radius=4)
            lbl_plus = self.font_body.render("+", True, COLOR_TEXT)
            self.screen.blit(lbl_plus, (adj["rect_plus"].centerx - lbl_plus.get_width()//2, adj["rect_plus"].centery - lbl_plus.get_height()//2 - 1))
            
        # 5. Model Cost Breakdown
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 460), (1180, 460), 1)
        
        txt_costs = self.font_header.render(f"Best Model Cost: {int(best_ind.total_cost):,}", True, COLOR_ACCENT)
        self.screen.blit(txt_costs, (820, 470))
        
        c_power = f"Power cost: {int(best_ind.power_cost * self.ga.power_weight):,}"
        c_overlap = f"Overlap penalty: {int(best_ind.overlap_cost * self.ga.overlap_weight):,}"
        c_cap = f"Capacity penalty: {int(best_ind.capacity_cost * self.ga.capacity_weight):,}"
        
        self.screen.blit(self.font_body.render(c_power, True, COLOR_TEXT_MUTED), (820, 492))
        self.screen.blit(self.font_body.render(c_overlap, True, COLOR_TEXT_MUTED), (820, 510))
        self.screen.blit(self.font_body.render(c_cap, True, COLOR_TEXT_MUTED), (820, 528))
        
        # 6. Dynamic Router load bars layout (3 Columns Grid)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 552), (1180, 552), 1)
        
        txt_routers = self.font_header.render("Router Load Allocations", True, COLOR_ACCENT)
        self.screen.blit(txt_routers, (820, 560))
        
        bar_width = 100
        bar_height = 5
        ap_cap = self.ga.ap_capacity
        num_aps = self.ga.num_aps
        
        for i in range(num_aps):
            col = i % 3
            row = i // 3
            
            x_bar = 820 + col * 123
            y_bar = 585 + row * 30
            
            load = best_ind.ap_loads[i] if i < len(best_ind.ap_loads) else 0
            percentage = min(1.0, load / ap_cap) if ap_cap > 0 else 0
            
            if load > ap_cap:
                bar_color = COLOR_RED
            elif load > ap_cap * 0.8:
                bar_color = COLOR_ORANGE
            else:
                bar_color = AP_COLORS[i % len(AP_COLORS)]
                
            # Render label: "R1: 18/22"
            lbl_ap = self.font_mono.render(f"R{i+1}:{load:02d}/{ap_cap}", True, COLOR_TEXT)
            self.screen.blit(lbl_ap, (x_bar, y_bar))
            
            # Load bar
            pygame.draw.rect(self.screen, (20, 30, 45), (x_bar, y_bar + 14, bar_width, bar_height), border_radius=2)
            if percentage > 0:
                pygame.draw.rect(self.screen, bar_color, (x_bar, y_bar + 14, int(bar_width * percentage), bar_height), border_radius=2)

        # 7. Mini Cost History Graph
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 736), (1180, 736), 1)
        
        chart_x = 820
        chart_y = 744
        chart_w = 360
        chart_h = 50
        
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
                
            pygame.draw.lines(self.screen, COLOR_CHART_AVG, False, pts_avg, 1)
            pygame.draw.lines(self.screen, COLOR_ACCENT, False, pts_best, 2)
            
            cost_lbl = self.font_subtitle.render(f"Cost: {int(min_val):,} to {int(max_val):,}", True, COLOR_TEXT_MUTED)
            self.screen.blit(cost_lbl, (chart_x + 5, chart_y + 1))
        else:
            lbl_empty = self.font_subtitle.render("Waiting for history data...", True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_empty, (chart_x + 10, chart_y + 18))

if __name__ == '__main__':
    # Initialize defaults from constants which can be overwritten directly in the script
    # e.g., to override via code: set DEFAULT_GRID_SIZE = 500, etc.
    app = PygameApp()
    app.run()
