"""
Dijkstra Path Planning Algorithm for Robot Navigation
This program finds the shortest path between start and goal points
while avoiding obstacles shaped as text characters.
"""

import math
import time
import numpy as np
import pygame
from queue import PriorityQueue
from collections import deque 

class PathPlanner:
    def __init__(self, canvas_width=680, canvas_height=189, clearance= 7 ):
        """Initialize the path planner with canvas dimensions and robot clearance"""
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.clearance = clearance
        self.grid = np.zeros((canvas_width, canvas_height))
        
        # Visualization parameters
        self.colors = {
            'background': (200, 200, 200),
            'obstacle': (225, 50, 50),
            'clearance': (0, 0, 0),
            'visited': (105, 135, 235),
            'path': (0, 0, 0),
            'start': (50, 220, 50),
            'goal': (225, 50, 50)
        }
        
        # Initialize data structures for search
        self.open_list = PriorityQueue()
        self.closed_list = []
        self.closed_states = set()
        
        # Cost for different movements
        self.straight_cost = 1.0
        self.diagonal_cost = 1.4
        
        # Create the obstacle map
        print("Creating Obstacle Map ....")
        self._create_obstacle_map()
        
    def _create_obstacle_map(self):
        """Create the obstacle map with obstacles and clearance zones"""
        # First pass: mark obstacles
        for i in range(self.canvas_width):
            for j in range(self.canvas_height):
                if self._is_obstacle(i, j):
                    self.grid[i, j] = 1
        
        # Second pass: add clearance around obstacles
        for i in range(self.clearance, self.canvas_width - self.clearance):
            for j in range(self.clearance, self.canvas_height - self.clearance):
                if self.grid[i, j] == 1:
                    for i1 in range(i - self.clearance, i + self.clearance + 1):
                        for j1 in range(j - self.clearance, j + self.clearance + 1):
                            if ((i1-i)**2 + (j1-j)**2) <= self.clearance**2:
                                if self.grid[i1, j1] == 0:
                                    self.grid[i1, j1] = 2
    
    def _is_obstacle(self, x, y):
        """Check if a point is part of an obstacle"""
        is_in_obstacle = 0
        
        # Text display parameters
        base_x = 135  # Starting position for text
        base_y = 57  #  Starting position for text
        letter_height = 94  # Proportional height
        spacing = 25  # Proportional spacing
        thickness = int(letter_height * 0.1)  # Proportional thickness
        
        # Calculate letter positions and sizes
        letter_e_width = letter_height * 0.33
        letter_n_width = letter_height * 0.5
        letter_p_width = letter_height * 0.4
        letter_m_width = letter_height * 0.67
        number_6_width = letter_height * 0.4
        number_1_width = letter_height 
        
        # Position each character and check if point falls within
        pos = base_x
        is_in_obstacle += self.create_letter_E(pos, x, y, base_y, letter_height, thickness)
        
        pos += letter_e_width + spacing
        is_in_obstacle += self.create_letter_N(pos, x, y, base_y, letter_height, thickness)
        
        pos += letter_n_width + spacing
        is_in_obstacle += self.create_letter_P(pos, x, y, base_y, letter_height, thickness)
        
        pos += letter_p_width + spacing
        is_in_obstacle += self.create_letter_M(pos, x, y, base_y, letter_height, thickness)
        
        pos += letter_m_width + spacing
        is_in_obstacle += self.create_number_6(pos, x, y, base_y, letter_height, thickness)
        
        pos += number_6_width + spacing
        is_in_obstacle += self.create_number_6(pos, x, y, base_y, letter_height, thickness)
        
        pos += number_1_width + spacing
        is_in_obstacle += self.create_number_1(pos, x, y, base_y, letter_height, thickness)
        
        # Check if point is on canvas border
        if x < self.clearance or x >= self.canvas_width - self.clearance or y < self.clearance or y >= self.canvas_height - self.clearance:
            is_in_obstacle += 1
            
        return is_in_obstacle


     # Function to create a letter obstacle using relative positioning
    def create_letter_E(self, x_pos, x, y, base_y, letter_height, thickness):
        e_left = x_pos
        e_top = base_y
        e_width = letter_height * 0.33  
        e_height = letter_height
        e_thickness = thickness
        
        # Main vertical line of E
        if (e_left <= x <= e_left + e_thickness) and (e_top <= y <= e_top + e_height):
            return 1
        
        # Top horizontal line of E
        if (e_left <= x <= e_left + e_width) and (e_top <= y <= e_top + e_thickness):
            return 1
        
        # Middle horizontal line of E
        if (e_left <= x <= e_left + e_width * 0.8) and (e_top + e_height/2 - e_thickness/2 <= y <= e_top + e_height/2 + e_thickness/2):
            return 1
        
        # Bottom horizontal line of E
        if (e_left <= x <= e_left + e_width) and (e_top + e_height - e_thickness <= y <= e_top + e_height):
            return 1
        
        return 0
    
    def create_letter_N(self, x_pos, x, y, base_y, letter_height, thickness):
        n_left = x_pos
        n_top = base_y
        n_width = letter_height * 0.5
        n_height = letter_height
        n_thickness = thickness
        
        # Left vertical line of N
        if (n_left <= x <= n_left + n_thickness) and (n_top <= y <= n_top + n_height):
            return 1
        
        # Right vertical line of N
        if (n_left + n_width - n_thickness <= x <= n_left + n_width) and (n_top <= y <= n_top + n_height):
            return 1
        
        # Define the two endpoints of the diagonal
        x1, y1 = n_left + n_thickness, n_top + n_height  
        x2, y2 = n_left + n_width - n_thickness, n_top   
        
        # Vector along the line
        dx, dy = x2 - x1, y2 - y1
        line_length = math.sqrt(dx**2 + dy**2)
        
        # Unit normal vector to the line (perpendicular)
        nx, ny = -dy/line_length, dx/line_length
        
        # Check if point is within the thick diagonal line
        px, py = x - x1, y - y1
        
        # Calculate projection length along the line
        proj_len = (px * dx + py * dy) / line_length
        
        # Calculate perpendicular distance from line
        perp_dist = abs(px * nx + py * ny)
        
        # Point is on diagonal if:
        # 1. Projection falls within the line segment
        # 2. Perpendicular distance is less than half the thickness
        half_thickness = n_thickness / 2
        
        if (0 <= proj_len <= line_length) and (perp_dist <= half_thickness):
            return 1
        return 0
            
    def create_letter_P(self, x_pos, x, y, base_y, letter_height, thickness):
        six_left = x_pos
        six_top = base_y
        six_width = letter_height * 0.4
        six_height = letter_height
        six_thickness = thickness
        
        # Full height left vertical line
        if (six_left <= x <= six_left + six_thickness) and (six_top <= y <= six_top + six_height):
            return 1
        
        # Bottom half - create curved portion 
        bottom_mid_height = six_height * 0.5  # Bottom half starts at middle of height
        
        # Bottom horizontal line
        if (six_left <= x <= six_left + six_width) and \
        (six_top + six_height - six_thickness <= y <= six_top + six_height):
            return 1
        
        # Right vertical half-line (only bottom half)
        if (six_left + six_width - six_thickness <= x <= six_left + six_width) and \
        (six_top + bottom_mid_height <= y <= six_top + six_height - six_thickness):
            return 1
        
        # Middle horizontal line that makes the bottom loop
        if (six_left <= x <= six_left + six_width) and \
        (six_top + bottom_mid_height <= y <= six_top + bottom_mid_height + six_thickness):
            return 1
        
        # Curved portion in bottom right
        corner_radius = six_height * 0.15
        
        # Bottom-right corner (quarter circle)
        br_center_x = six_left + six_width - corner_radius
        br_center_y = six_top + six_height - corner_radius
        
        dist_from_br = math.sqrt((x - br_center_x)**2 + (y - br_center_y)**2)
        if (corner_radius - six_thickness <= dist_from_br <= corner_radius) and \
        (x >= br_center_x) and (y >= br_center_y):
            return 1
        
        # Middle-right curved connection (quarter circle)
        mr_center_x = six_left + six_width - corner_radius
        mr_center_y = six_top + bottom_mid_height + corner_radius
        
        dist_from_mr = math.sqrt((x - mr_center_x)**2 + (y - mr_center_y)**2)
        if (corner_radius - six_thickness <= dist_from_mr <= corner_radius) and \
        (x >= mr_center_x) and (y <= mr_center_y and y >= mr_center_y - 2*corner_radius):
            return 1
        
        return 0
    
    def create_letter_M(self, x_pos, x, y, base_y, letter_height, thickness):
        m_left = x_pos
        m_top = base_y
        m_width = letter_height * 0.67  # Wider for M
        m_height = letter_height
        m_thickness = thickness
        
        # Left vertical line of M
        if (m_left <= x <= m_left + m_thickness) and (m_top <= y <= m_top + m_height):
            return 1
        
        # Right vertical line of M
        if (m_left + m_width - m_thickness <= x <= m_left + m_width) and (m_top <= y <= m_top + m_height):
            return 1
        
        # Left diagonal of M
        slope_left = (m_height/2) / (m_width/2)
        if (m_left + 15 <= x <= m_left + m_width/2) and \
           (m_top + m_height - slope_left * (x - m_left) - m_thickness <= y <= m_top + m_height - slope_left * (x - m_left) + m_thickness):
            return 1
        
        # Right diagonal of M
        slope_right = -(m_height/2) / (m_width/2)
        if (m_left + m_width/2 <= x <= m_left + m_width - 15) and \
           (m_top + m_height/2 - slope_right * (x - (m_left + m_width/2)) - m_thickness <= y <= m_top + m_height/2 - slope_right * (x - (m_left + m_width/2)) + m_thickness):
            return 1
        
        return 0
    
    def create_number_6(self, x_pos, x, y, base_y, letter_height, thickness):
        six_left = x_pos
        six_top = base_y
        six_width = letter_height * 0.4
        six_height = letter_height
        six_thickness = thickness
        
        # Full height left vertical line
        if (six_left <= x <= six_left + six_thickness) and (six_top <= y <= six_top + six_height):
            return 1
        
        # TOP half - create curved portion (for proper "6")
        top_mid_height = six_height * 0.5  
        # Top horizontal line
        if (six_left <= x <= six_left + six_width) and \
        (six_top <= y <= six_top + six_thickness):
            return 1
        
        # Right vertical half-line (only top half)
        if (six_left + six_width - six_thickness <= x <= six_left + six_width) and \
        (six_top + six_thickness <= y <= six_top + top_mid_height):
            return 1
        
        # Middle horizontal line that makes the top loop
        if (six_left <= x <= six_left + six_width) and \
        (six_top + top_mid_height - six_thickness <= y <= six_top + top_mid_height):
            return 1
        
        # Curved portion in top right
        corner_radius = six_height * 0.15
        
        # Top-right corner (quarter circle)
        tr_center_x = six_left + six_width - corner_radius
        tr_center_y = six_top + corner_radius
        
        dist_from_tr = math.sqrt((x - tr_center_x)**2 + (y - tr_center_y)**2)
        if (corner_radius - six_thickness <= dist_from_tr <= corner_radius) and \
        (x >= tr_center_x) and (y <= tr_center_y):
            return 1
        
        # Middle-right curved connection (quarter circle)
        mr_center_x = six_left + six_width - corner_radius
        mr_center_y = six_top + top_mid_height - corner_radius
        
        dist_from_mr = math.sqrt((x - mr_center_x)**2 + (y - mr_center_y)**2)
        if (corner_radius - six_thickness <= dist_from_mr <= corner_radius) and \
        (x >= mr_center_x) and (y >= mr_center_y and y <= mr_center_y + 2*corner_radius):
            return 1
        
        return 0
    
    def create_number_1(self, x_pos, x, y, base_y, letter_height, thickness):
        m_left = x_pos
        m_top = base_y
        m_width = letter_height * 0.67
        m_height = letter_height
        m_thickness = thickness
        
        # Right vertical line of M
        if (m_left + m_width - m_thickness - 100 <= x <= m_left + m_width - 100) and (m_top <= y <= m_top + m_height):
            return 1
        
        return 0
  
    def _get_neighbors(self, state, cost):
        """Get all valid neighboring states and their costs"""
        neighbors = []
        moves = [
            # [dx, dy, cost_factor]
            [0, 1, self.straight_cost],    # Up
            [1, 1, self.diagonal_cost],    # Up-Right
            [1, 0, self.straight_cost],    # Right
            [1, -1, self.diagonal_cost],   # Down-Right
            [0, -1, self.straight_cost],   # Down
            [-1, -1, self.diagonal_cost],  # Down-Left
            [-1, 0, self.straight_cost],   # Left
            [-1, 1, self.diagonal_cost]    # Up-Left
        ]
        
        for dx, dy, cost_factor in moves:
            new_x = state[0] + dx
            new_y = state[1] + dy
            new_cost = cost + cost_factor
            
            # Check if new position is valid
            if 0 <= new_x < self.canvas_width and 0 <= new_y < self.canvas_height:
                neighbors.append((new_cost, [new_x, new_y]))
                
        return neighbors
    
    def _backtrack(self, start_state):
        """Reconstruct the path from goal to start"""
        print(" Backtracking ....")
        path = []
        parent_state = self.closed_list[-1][1]  
        
        while parent_state != tuple(start_state):
            for node in self.closed_list:
                if node[2] == parent_state:
                    path.append(parent_state)
                    parent_state = node[1]
                    break
        
        path.reverse()
        return path
    
    def get_start_goal_inputs(self):
        """
        Get valid start and goal positions from user input
        
        Accepts user coordinates in mm (5-175 X, 5-45 Y) and converts to pixels
        """
        # Conversion factor from mm to pixels
        x_scale = self.canvas_width / 180  # 680 pixels / 180 mm
        y_scale = self.canvas_height / 50  # 189 pixels / 50 mm
        
        # Get valid start position
        while True:
            print("\n===== START NODE =====")
            try:
                start_x_mm = float(input("Enter Start Node 'X' coordinate (5-175 mm): "))
                start_y_mm = float(input("Enter Start Node 'Y' coordinate (5-45 mm): "))
                print("__________________________")
                
                # Check if coordinates are within valid range
                if not (5 <= start_x_mm <= 175 and 5 <= start_y_mm <= 45):
                    print("Coordinates out of range. X must be 5-175 mm, Y must be 5-45 mm")
                    continue
                    
                # Convert mm to pixel coordinates
                start_x_px = int(start_x_mm * x_scale)
                start_y_px = int(start_y_mm * y_scale)
                
                # Check if position is valid in grid
                if self.grid[start_x_px, start_y_px] == 0:
                    start_state = [start_x_px, start_y_px]
                    break
                else:
                    print("Start Node is in obstacle or clearance zone, please enter different coordinates")
            except ValueError:
                print("Please enter valid numerical coordinates")
        
        # Get valid goal position
        while True:
            print("\n===== GOAL NODE =====")
            try:
                goal_x_mm = float(input("Enter Goal Node 'X' coordinate (5-175 mm): "))
                goal_y_mm = float(input("Enter Goal Node 'Y' coordinate (5-45 mm): "))
                
                # Check if coordinates are within valid range
                if not (5 <= goal_x_mm <= 175 and 5 <= goal_y_mm <= 45):
                    print("Coordinates out of range. X must be 5-175 mm, Y must be 5-45 mm")
                    continue
                    
                # Convert mm to pixel coordinates
                goal_x_px = int(goal_x_mm * x_scale)
                goal_y_px = int(goal_y_mm * y_scale)
                
                # Check if position is valid in grid
                if self.grid[goal_x_px, goal_y_px] == 0:
                    goal_state = [goal_x_px, goal_y_px]
                    break
                else:
                    print("Goal Node is in obstacle or clearance zone, please enter different coordinates")
            except ValueError:
                print("Please enter valid numerical coordinates")
                
        print("__________________________")
        print("  Nodes have been accepted  ")
        print(f"  Start: ({start_x_mm:.1f}, {start_y_mm:.1f}) mm → ({start_x_px}, {start_y_px}) pixels")
        print(f"  Goal: ({goal_x_mm:.1f}, {goal_y_mm:.1f}) mm → ({goal_x_px}, {goal_y_px}) pixels")
        print("  Calculating the path...  ")
        
        return start_state, goal_state
    def find_path_dijsktra(self, start_state, goal_state):
        """Find shortest path from start to goal using Dijkstra's algorithm"""
        # Initialize search structures
        self.open_list = PriorityQueue()
        self.closed_list = []
        self.closed_states = set()
        
        # Add start node to open list
        self.open_list.put([0, start_state, start_state])
        
        while not self.open_list.empty():
            # Get node with lowest cost
            current_cost, parent, current_state = self.open_list.get()
            
            # Skip if already processed
            if tuple(current_state) in self.closed_states:
                continue
                
            # Add to closed list
            self.closed_list.append((current_cost, tuple(parent), tuple(current_state)))
            self.closed_states.add(tuple(current_state))
            
            # Check if goal reached
            if current_state == goal_state:
                print("\nGoal Reached")
                return self._backtrack(start_state)
                
            # Explore neighbors
            neighbors = self._get_neighbors(current_state, current_cost)
            for neighbor_cost, neighbor_state in neighbors:
                # Skip if obstacle or already processed
                if (self.grid[neighbor_state[0], neighbor_state[1]] != 0 or 
                    tuple(neighbor_state) in self.closed_states):
                    continue
                    
                # Add to open list
                self.open_list.put([neighbor_cost, current_state, neighbor_state])
                
        print("No path found!")
        return None
    
    def visualize(self, start_state, goal_state, path=None):
        """Visualize the map, obstacles, explored nodes and path"""
        # Initialize pygame
        pygame.init()
        window = pygame.display.set_mode((self.canvas_width, self.canvas_height))
        pygame.display.set_caption("Dijkstra Path Planning")
        window.fill(self.colors['background'])
        
        # Draw obstacles and clearance zones
        for i in range(self.canvas_width):
            for j in range(self.canvas_height):
                if self.grid[i, j] == 1:  # Obstacle
                    window.set_at((i, self.canvas_height-1-j), self.colors['obstacle'])
                elif self.grid[i, j] == 2:  # Clearance
                    window.set_at((i, self.canvas_height-1-j), self.colors['clearance'])
        
        pygame.display.flip()
        
        # Visualize exploration
        for idx, node in enumerate(self.closed_list):
            point = node[2]
            window.set_at((point[0], self.canvas_height-1-point[1]), self.colors['visited'])
            if idx % 80 == 0:  # Update display periodically for smoother visualization
                pygame.display.flip()
        
        # Draw start point
        pygame.draw.circle(window, self.colors['clearance'], 
                          (start_state[0], self.canvas_height-1-start_state[1]), 5)
        pygame.draw.circle(window, self.colors['start'], 
                          (start_state[0], self.canvas_height-1-start_state[1]), 3)
        
        pygame.display.flip()
        time.sleep(1)
        
        # Visualize path
        if path:
            for point in path:
                time.sleep(0.01)  # Slow down to see the path forming
                window.set_at((point[0], self.canvas_height-1-point[1]), self.colors['path'])
                pygame.display.flip()
        
        # Draw goal point
        pygame.draw.circle(window, self.colors['clearance'], 
                          (goal_state[0], self.canvas_height-1-goal_state[1]), 5)
        pygame.draw.circle(window, self.colors['goal'], 
                          (goal_state[0], self.canvas_height-1-goal_state[1]), 3)
        
        pygame.display.flip()
        
        print(" ------------------------ PATH SUCCESSFULLY TRACKED ------------------------")
        print("-------  PATH PLANNING COMPLETED -------")
        
        # Keep window open until closed
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.update()
        
        pygame.quit()
    def find_path_bfs(self, start_state, goal_state):
        """Find shortest path from start to goal using Breadth-First Search (BFS)"""
        # Initialize search structures
        queue = deque([start_state])  # Use deque for efficient queue operations
        visited = set()  # Set to track visited states
        visited.add(tuple(start_state))
        
        # Dictionary to store parent nodes for backtracking
        parent = {tuple(start_state): None}
        
        # Store all visited nodes for visualization (similar to closed_list in Dijkstra)
        self.closed_list = []  # Clear existing data
        self.closed_list.append((0, None, tuple(start_state)))  # Add start node
        
        print("Starting BFS search...")
        nodes_explored = 0
        
        while queue:
            # Get the next state from the queue
            current_state = queue.popleft()
            nodes_explored += 1
            
            # Check if goal reached
            if current_state == goal_state:
                print(f"\nGoal Reached! Explored {nodes_explored} nodes.")
                # Add goal to closed list for visualization
                self.closed_list.append((0, parent[tuple(current_state)], tuple(current_state)))
                # Backtrack to find path
                return self._backtrack_bfs(start_state, goal_state, parent)
            
            # Get all valid neighbors (8-connected grid)
            moves = [
                [0, 1],     # Up
                [1, 1],     # Up-Right
                [1, 0],     # Right
                [1, -1],    # Down-Right
                [0, -1],    # Down
                [-1, -1],   # Down-Left
                [-1, 0],    # Left
                [-1, 1]     # Up-Left
            ]
            
            for dx, dy in moves:
                new_x = current_state[0] + dx
                new_y = current_state[1] + dy
                new_state = [new_x, new_y]
                
                # Check if the new state is valid
                if (0 <= new_x < self.canvas_width and 
                    0 <= new_y < self.canvas_height and 
                    self.grid[new_x, new_y] == 0 and
                    tuple(new_state) not in visited):
                    
                    queue.append(new_state)
                    visited.add(tuple(new_state))
                    parent[tuple(new_state)] = tuple(current_state)
                    
                    # Add to closed list for visualization (with cost 0 since BFS doesn't use costs)
                    self.closed_list.append((0, tuple(current_state), tuple(new_state)))
                
            # Print progress periodically
            if nodes_explored % 1000 == 0:
                print(f"Explored {nodes_explored} nodes, Queue size: {len(queue)}")
        
        print("No path found!")
        return None
            

    def _backtrack_bfs(self, start_state, goal_state, parent_dict):
        """Reconstruct path from start to goal using parent dictionary"""
        path = []
        current = tuple(goal_state)
        
        while current != tuple(start_state):
            path.append(current)
            current = parent_dict[current]
        
        path.reverse()
        print(f"Path found with {len(path)} steps")
        return path

    def visualize_and_record(self, start_state, goal_state, path=None, save_video=True, video_filename="dijkstra_planning.mp4"):
        """Visualize the map, obstacles, explored nodes and path, and optionally save as MP4"""
        # Initialize pygame
        pygame.init()
        window = pygame.display.set_mode((self.canvas_width, self.canvas_height))
        pygame.display.set_caption("Dijkstra Path Planning")
        window.fill(self.colors['background'])
        
        # Video recording setup
        if save_video:
            try:
                import cv2
                fps = 30
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (self.canvas_width, self.canvas_height))
                print(f"Recording video to {video_filename}...")
            except ImportError:
                print("OpenCV (cv2) not found. Video will not be saved.")
                save_video = False
        
        # Draw obstacles and clearance zones
        for i in range(self.canvas_width):
            for j in range(self.canvas_height):
                if self.grid[i, j] == 1:  # Obstacle
                    window.set_at((i, self.canvas_height-1-j), self.colors['obstacle'])
                elif self.grid[i, j] == 2:  # Clearance
                    window.set_at((i, self.canvas_height-1-j), self.colors['clearance'])
        
        pygame.display.flip()
        
        # Capture initial frame
        if save_video:
            self._capture_frame(window, video_writer)
        
        # Visualize exploration (in chunks to speed up video)
        chunk_size = 80
        for i in range(0, len(self.closed_list), chunk_size):
            # Process a chunk of nodes
            for idx in range(i, min(i + chunk_size, len(self.closed_list))):
                point = self.closed_list[idx][2]
                window.set_at((point[0], self.canvas_height-1-point[1]), self.colors['visited'])
            
            pygame.display.flip()
            if save_video:
                self._capture_frame(window, video_writer)
        
        # Draw start point
        pygame.draw.circle(window, self.colors['clearance'], 
                        (start_state[0], self.canvas_height-1-start_state[1]), 5)
        pygame.draw.circle(window, self.colors['start'], 
                        (start_state[0], self.canvas_height-1-start_state[1]), 3)
        
        pygame.display.flip()
        if save_video:
            self._capture_frame(window, video_writer)
        
        # Visualize path
        if path:
            for point in path:
                window.set_at((point[0], self.canvas_height-1-point[1]), self.colors['path'])
                pygame.display.flip()
                if save_video:
                    self._capture_frame(window, video_writer)
        
        # Draw goal point
        pygame.draw.circle(window, self.colors['clearance'], 
                        (goal_state[0], self.canvas_height-1-goal_state[1]), 5)
        pygame.draw.circle(window, self.colors['goal'], 
                        (goal_state[0], self.canvas_height-1-goal_state[1]), 3)
        
        pygame.display.flip()
        if save_video:
            # Capture final frame multiple times to make it stay longer
            for _ in range(fps * 3):  
                self._capture_frame(window, video_writer)
            video_writer.release()
            print(f"Video saved to {video_filename}")
        print(" ------------------------ PATH SUCCESSFULLY TRACKED ------------------------")
        print("------- DIJKSTRA PATH PLANNING COMPLETED -------")
        
        # Keep window open until closed
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            pygame.display.update()
        
        pygame.quit()

    def _capture_frame(self, surface, video_writer):
        """Capture the current pygame surface as a frame for the video"""
        # Convert pygame surface to numpy array for OpenCV
        import numpy as np
        frame = pygame.surfarray.array3d(surface)
        frame = frame.transpose([1, 0, 2])
        # Convert from RGB to BGR (OpenCV format)
        frame = np.flip(frame, axis=2)
        video_writer.write(frame)

def main():
    print("\n PATH PLANNING ALGORITHMS ")
    print(" ----------------------- ")
    print(" 1. Start Node : Green ")
    print(" 2. Goal Node : Red ")
    print(" 3. Obstacles : Red ")
    print(" 4. Path : Black ")
    print(" 5. Visited Nodes : Blue ")
    print(" 6. Clearance Zones : Black ")
    print(" ----------------------- ")
    
    # Select algorithm
    while True:
        algo_choice = input("Select algorithm:\n1. Dijkstra\n2. BFS \nChoice: ")
        if algo_choice in ['1', '2']:
            break
        print("Invalid choice. Please select 1 or 2.")
    
    start_time = time.time()
    
    # Create path planner with default canvas size and clearance
    planner = PathPlanner()
    
    # Get start and goal positions from user
    start, goal = planner.get_start_goal_inputs()
    
    # Find the shortest path using selected algorithm
    if algo_choice == '1':
        print("Running Dijkstra's algorithm...")
        path = planner.find_path_dijsktra(start, goal)
    else:
        print("Running BFS algorithm...")
        path = planner.find_path_bfs(start, goal)
    
    # Display execution time
    print(f"Process finished --- {time.time() - start_time:.2f} seconds ---\n")
    
    # Generate video filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    algorithm_name = "dijkstra" if algo_choice == '1' else "bfs"
    video_filename = f"{algorithm_name}_path_{timestamp}.mp4"
    
    # Ask user if they want to save video
    save_choice = input("Save visualization as video? (y/n): ").lower()
    save_video = save_choice == 'y' or save_choice == 'yes'
    
    if save_video:
        # Visualize and save as video
        planner.visualize_and_record(start, goal, path, save_video=True, video_filename=video_filename)
    else:
        # Just visualize without saving
        planner.visualize(start, goal, path)

    # Ensure all pygame resources are released
    pygame.quit()

if __name__ == "__main__":
    main()