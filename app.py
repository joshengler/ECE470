#!/usr/bin/env python3
import pygame
import sys
import time
import math
from typing import Dict, Any
from genetic_algorithm import generate_devices, GeneticAlgorithm, Individual, Device

# Color definitions (Premium Slate/Purple Theme)
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
COLOR_WHITE = (255, 255, 255)

# AP Colors
AP_COLORS = [
    (59, 130, 246),  # Blue
    (168, 85, 247), # Purple
    (16, 185, 129), # Green
    (249, 115, 22),  # Orange
    (6, 182, 212)   # Cyan
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
        self.font_title = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_subtitle = pygame.font.SysFont("Arial", 14, italic=True)
        self.font_header = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 14)
        self.font_mono = pygame.font.SysFont("Courier", 13, bold=True)
        
        # Initialize problem setup
        self.devices = generate_devices(num_devices=100, grid_size=500, seed=42)
        
        # Initial GA config
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
        self.ga = GeneticAlgorithm(devices=self.devices, **self.ga_config)
        
        # Execution control variables
        self.is_running = False
        self.step_delay = 0.05  # seconds between generations when playing
        self.last_step_time = 0.0
        
        # Define layout rectangles
        self.rect_grid = pygame.Rect(0, 0, 800, 800)
        self.rect_sidebar = pygame.Rect(800, 0, 400, 800)
        
        # Interactive UI buttons
        self.btn_play = pygame.Rect(820, 110, 170, 35)
        self.btn_step = pygame.Rect(1010, 110, 170, 35)
        self.btn_step10 = pygame.Rect(820, 155, 170, 35)
        self.btn_reset = pygame.Rect(1010, 155, 170, 35)
        
        # Parameter Adjuster rects list (name, key, value_formatter, step, min_val, max_val, rect_minus, rect_plus)
        self.adjusters = [
            {"name": "AP Radius", "key": "ap_radius", "fmt": lambda x: f"{x:.1f}", "step": 1.0, "min": 5.0, "max": 50.0},
            {"name": "AP Capacity", "key": "ap_capacity", "fmt": lambda x: f"{x:d}", "step": 1, "min": 5, "max": 50},
            {"name": "Mutation Rate", "key": "mutation_rate", "fmt": lambda x: f"{x*100:.0f}%", "step": 0.01, "min": 0.01, "max": 0.5},
            {"name": "Crossover Rate", "key": "crossover_rate", "fmt": lambda x: f"{x*100:.0f}%", "step": 0.05, "min": 0.1, "max": 1.0},
            {"name": "Power Wt", "key": "power_weight", "fmt": lambda x: f"{x:.1f}", "step": 0.1, "min": 0.0, "max": 10.0},
            {"name": "Overlap Wt", "key": "overlap_weight", "fmt": lambda x: f"{x:.0f}", "step": 10.0, "min": 0.0, "max": 1000.0},
            {"name": "Capacity Wt", "key": "capacity_weight", "fmt": lambda x: f"{x:.0f}", "step": 50.0, "min": 0.0, "max": 2000.0},
        ]
        
        # Add positioning to adjusters dynamically
        start_y = 245
        for idx, adj in enumerate(self.adjusters):
            y_pos = start_y + idx * 30
            adj["rect_minus"] = pygame.Rect(1110, y_pos, 30, 22)
            adj["rect_plus"] = pygame.Rect(1150, y_pos, 30, 22)

    def run(self):
        while True:
            self.handle_events()
            self.update_logic()
            self.draw()
            self.clock.tick(60)  # Cap display at 60 FPS

    def update_logic(self):
        # Step GA if running and delay expired
        if self.is_running:
            current_time = time.time()
            if current_time - self.last_step_time >= self.step_delay:
                self.ga.step()
                self.last_step_time = current_time

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
                    self.is_running = False
                    self.ga.initialize_population()
                elif event.key == pygame.K_UP:
                    self.step_delay = max(0.005, self.step_delay - 0.01)
                elif event.key == pygame.K_DOWN:
                    self.step_delay = min(0.5, self.step_delay + 0.01)
                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
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
                        self.is_running = False
                        self.ga.initialize_population()
                        
                    # 2. Adjusters buttons
                    for adj in self.adjusters:
                        val = self.ga_config[adj["key"]]
                        if adj["rect_minus"].collidepoint(pos):
                            new_val = max(adj["min"], val - adj["step"])
                            self.update_param(adj["key"], new_val)
                        elif adj["rect_plus"].collidepoint(pos):
                            new_val = min(adj["max"], val + adj["step"])
                            self.update_param(adj["key"], new_val)

    def update_param(self, key: str, value: Any):
        if key in ["ap_capacity", "pop_size"]:
            value = int(value)
        self.ga_config[key] = value
        self.ga.update_parameters({key: value})

    def draw(self):
        # Draw background
        self.screen.fill(COLOR_BG)
        
        # Draw panels
        self.draw_grid_panel()
        self.draw_sidebar_panel()
        
        pygame.display.flip()

    def draw_grid_panel(self):
        # Fill grid panel background
        pygame.draw.rect(self.screen, COLOR_GRID_BG, self.rect_grid)
        
        best_ind = self.ga.get_best_individual()
        scale = 8.0 # Map 100x100 to 800x800
        
        # 1. Draw subtle grid coordinate lines (every 10 units)
        for i in range(10, 100, 10):
            screen_pos = int(i * scale)
            pygame.draw.line(self.screen, (20, 25, 40), (screen_pos, 0), (screen_pos, 800), 1)
            pygame.draw.line(self.screen, (20, 25, 40), (0, screen_pos), (800, screen_pos), 1)
            
        # 2. Draw AP coverage circles (semi-transparent overlay)
        # Create transparent canvas for circles overlay
        overlay = pygame.Surface((800, 800), pygame.SRCALPHA)
        ap_radius_screen = int(self.ga.ap_radius * scale)
        for idx, ap in enumerate(best_ind.aps):
            cx, cy = int(ap[0] * scale), int(ap[1] * scale)
            color = AP_COLORS[idx]
            # Draw translucent coverage area
            pygame.draw.circle(overlay, color + (20,), (cx, cy), ap_radius_screen)
            # Draw dashed/faint outer circle boundary
            pygame.draw.circle(overlay, color + (50,), (cx, cy), ap_radius_screen, 1)
        self.screen.blit(overlay, (0, 0))
        
        # 3. Draw connecting links (lines from device to closest AP)
        for dev in self.devices:
            ap_idx = best_ind.device_assignments[dev.id]
            if ap_idx >= 0 and ap_idx < len(best_ind.aps):
                ap = best_ind.aps[ap_idx]
                ap_color = AP_COLORS[ap_idx]
                start_pos = (int(dev.x * scale), int(dev.y * scale))
                end_pos = (int(ap[0] * scale), int(ap[1] * scale))
                pygame.draw.line(self.screen, ap_color + (40,), start_pos, end_pos, 1)
                
        # 4. Draw Device Nodes
        for dev in self.devices:
            ap_idx = best_ind.device_assignments[dev.id]
            color = AP_COLORS[ap_idx] if (0 <= ap_idx < len(AP_COLORS)) else COLOR_TEXT_MUTED
            pos = (int(dev.x * scale), int(dev.y * scale))
            pygame.draw.circle(self.screen, color, pos, 3)
            pygame.draw.circle(self.screen, COLOR_GRID_BG, pos, 3, 1) # Outline for contrast
            
        # 5. Draw Access Point Centers
        for idx, ap in enumerate(best_ind.aps):
            cx, cy = int(ap[0] * scale), int(ap[1] * scale)
            color = AP_COLORS[idx]
            load = best_ind.ap_loads[idx]
            is_overcapacity = load > self.ga.ap_capacity
            
            # Draw capacity warning halo
            if is_overcapacity:
                # Pulsating warning ring (time-based)
                pulse_r = 16 + int(3 * math.sin(time.time() * 10))
                pygame.draw.circle(self.screen, COLOR_RED, (cx, cy), pulse_r, 2)
                
            # AP solid center
            pygame.draw.circle(self.screen, color, (cx, cy), 9)
            pygame.draw.circle(self.screen, COLOR_WHITE, (cx, cy), 9, 2)
            pygame.draw.circle(self.screen, COLOR_WHITE, (cx, cy), 3)
            
            # Label
            lbl = self.font_mono.render(str(idx + 1), True, COLOR_WHITE)
            self.screen.blit(lbl, (cx - lbl.get_width()//2, cy - 23))

    def draw_sidebar_panel(self):
        # Draw background and left border
        pygame.draw.rect(self.screen, COLOR_BG, self.rect_sidebar)
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (800, 0), (800, 800), 2)
        
        best_ind = self.ga.get_best_individual()
        
        # 1. Title section
        title = self.font_title.render("Wireless GA Optimizer", True, COLOR_ACCENT)
        self.screen.blit(title, (820, 20))
        sub = self.font_subtitle.render("Optimizing placement of 5 access points on 100x100 grid", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub, (820, 50))
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 75), (1180, 75), 1)
        
        # 2. Performance Metrics
        txt_gen = self.font_header.render(f"Gen: {self.ga.generation}", True, COLOR_TEXT)
        self.screen.blit(txt_gen, (820, 85))
        
        status_text = "RUNNING" if self.is_running else "PAUSED"
        status_color = COLOR_GREEN if self.is_running else COLOR_ORANGE
        txt_status = self.font_header.render(status_text, True, status_color)
        self.screen.blit(txt_status, (1100, 85))
        
        # 3. Action Buttons
        # Play/Pause
        play_btn_color = COLOR_ORANGE if self.is_running else COLOR_GREEN
        play_btn_lbl = "Pause (Space)" if self.is_running else "Play (Space)"
        pygame.draw.rect(self.screen, play_btn_color, self.btn_play, border_radius=6)
        lbl_play = self.font_header.render(play_btn_lbl, True, COLOR_WHITE)
        self.screen.blit(lbl_play, (self.btn_play.centerx - lbl_play.get_width()//2, self.btn_play.centery - lbl_play.get_height()//2))
        
        # Step 1
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step, 1, border_radius=6)
        lbl_step = self.font_header.render("Step 1 (S)", True, COLOR_TEXT)
        self.screen.blit(lbl_step, (self.btn_step.centerx - lbl_step.get_width()//2, self.btn_step.centery - lbl_step.get_height()//2))
        
        # Step 10
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, self.btn_step10, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, self.btn_step10, 1, border_radius=6)
        lbl_step10 = self.font_header.render("Step 10", True, COLOR_TEXT)
        self.screen.blit(lbl_step10, (self.btn_step10.centerx - lbl_step10.get_width()//2, self.btn_step10.centery - lbl_step10.get_height()//2))
        
        # Reset
        pygame.draw.rect(self.screen, COLOR_RED, self.btn_reset, border_radius=6)
        lbl_reset = self.font_header.render("Reset (R)", True, COLOR_WHITE)
        self.screen.blit(lbl_reset, (self.btn_reset.centerx - lbl_reset.get_width()//2, self.btn_reset.centery - lbl_reset.get_height()//2))
        
        # 4. Parameter Adjustment Interface
        txt_param_hdr = self.font_header.render("Parameters", True, COLOR_ACCENT)
        self.screen.blit(txt_param_hdr, (820, 215))
        
        for adj in self.adjusters:
            val = self.ga_config[adj["key"]]
            y_pos = adj["rect_minus"].y
            
            # Label & Value
            lbl_name = self.font_body.render(adj["name"], True, COLOR_TEXT_MUTED)
            lbl_val = self.font_mono.render(adj["fmt"](val), True, COLOR_TEXT)
            self.screen.blit(lbl_name, (820, y_pos + 2))
            self.screen.blit(lbl_val, (970, y_pos + 2))
            
            # Draw minus button
            pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_minus"], border_radius=4)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_minus"], 1, border_radius=4)
            lbl_minus = self.font_body.render("-", True, COLOR_TEXT)
            self.screen.blit(lbl_minus, (adj["rect_minus"].centerx - lbl_minus.get_width()//2, adj["rect_minus"].centery - lbl_minus.get_height()//2 - 2))
            
            # Draw plus button
            pygame.draw.rect(self.screen, COLOR_PANEL_BG, adj["rect_plus"], border_radius=4)
            pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, adj["rect_plus"], 1, border_radius=4)
            lbl_plus = self.font_body.render("+", True, COLOR_TEXT)
            self.screen.blit(lbl_plus, (adj["rect_plus"].centerx - lbl_plus.get_width()//2, adj["rect_plus"].centery - lbl_plus.get_height()//2 - 1))
            
        # 5. Model Cost Breakdown
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 465), (1180, 465), 1)
        
        txt_costs = self.font_header.render(f"Best Model Cost: {int(best_ind.total_cost):,}", True, COLOR_ACCENT)
        self.screen.blit(txt_costs, (820, 475))
        
        c_power = f"Power cost: {int(best_ind.power_cost * self.ga.power_weight):,}"
        c_overlap = f"Overlap penalty: {int(best_ind.overlap_cost * self.ga.overlap_weight):,}"
        c_cap = f"Capacity penalty: {int(best_ind.capacity_cost * self.ga.capacity_weight):,}"
        
        self.screen.blit(self.font_body.render(c_power, True, COLOR_TEXT_MUTED), (820, 500))
        self.screen.blit(self.font_body.render(c_overlap, True, COLOR_TEXT_MUTED), (820, 520))
        self.screen.blit(self.font_body.render(c_cap, True, COLOR_TEXT_MUTED), (820, 540))
        
        # 6. Router Load bars
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 570), (1180, 570), 1)
        
        txt_routers = self.font_header.render("Router Allocations", True, COLOR_ACCENT)
        self.screen.blit(txt_routers, (820, 580))
        
        bar_width = 130
        bar_height = 8
        ap_cap = self.ga.ap_capacity
        
        # Layout allocations in a compact grid: 5 routers arranged horizontally or compact list
        for i in range(5):
            x_bar = 820 + (i % 2) * 190
            y_bar = 610 + (i // 2) * 35
            if i == 4:
                x_bar = 820  # Center or align last AP
                
            load = best_ind.ap_loads[i]
            percentage = min(1.0, load / ap_cap)
            
            # Pick color for bar
            if load > ap_cap:
                bar_color = COLOR_RED
            elif load > ap_cap * 0.8:
                bar_color = COLOR_ORANGE
            else:
                bar_color = AP_COLORS[i]
                
            # Render label: "R1: 18/22"
            lbl_ap = self.font_mono.render(f"R{i+1}:{load:02d}/{ap_cap}", True, COLOR_TEXT)
            self.screen.blit(lbl_ap, (x_bar, y_bar))
            
            # Draw bar background
            pygame.draw.rect(self.screen, (20, 30, 45), (x_bar, y_bar + 16, bar_width, bar_height), border_radius=4)
            # Draw bar load
            if percentage > 0:
                pygame.draw.rect(self.screen, bar_color, (x_bar, y_bar + 16, int(bar_width * percentage), bar_height), border_radius=4)

        # 7. Mini Cost History Graph
        pygame.draw.line(self.screen, COLOR_PANEL_BORDER, (820, 715), (1180, 715), 1)
        
        chart_x = 820
        chart_y = 725
        chart_w = 360
        chart_h = 65
        
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
                # Invert y because pygame screen coordinates have y=0 at top
                py_best = chart_y + chart_h - int(((hist[gen_idx] - min_val) / val_range) * (chart_h - 10) + 5)
                py_avg = chart_y + chart_h - int(((avg_hist[gen_idx] - min_val) / val_range) * (chart_h - 10) + 5)
                
                pts_best.append((px, py_best))
                pts_avg.append((px, py_avg))
                
            # Draw average cost lines (purple)
            pygame.draw.lines(self.screen, COLOR_CHART_AVG, False, pts_avg, 1)
            # Draw best cost lines (cyan)
            pygame.draw.lines(self.screen, COLOR_ACCENT, False, pts_best, 2)
            
            # Label
            cost_lbl = self.font_subtitle.render(f"Cost History: {int(min_val):,} to {int(max_val):,}", True, COLOR_TEXT_MUTED)
            self.screen.blit(cost_lbl, (chart_x + 5, chart_y + 2))
        else:
            # Draw placeholder label
            lbl_empty = self.font_subtitle.render("Waiting for history data...", True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_empty, (chart_x + 10, chart_y + 20))

if __name__ == '__main__':
    app = PygameApp()
    app.run()
