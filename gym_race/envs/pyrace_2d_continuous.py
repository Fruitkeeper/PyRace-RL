import pygame
import math
import numpy as np

screen_width = 1500
screen_height = 800
check_point = ((1370, 675), (1370, 215), (935, 465), (630, 180), (320, 160), (130, 675), (550, 702)) # race_track_ie.png
"""
Area for text boxes:
750,0 - 1500,100
1055,290 - 1300,605
"""

class Car:
    def __init__(self, car_file, map, pos): # map_file
        # self.map = pygame.image.load(map_file)
        self.map = map
        self.surface = pygame.image.load(car_file)
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = pos
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.check_flag = False
        self.last_checkpoint_time = 0  # Time when the last checkpoint was reached
        self.turn_rate = 5  # Default turn rate in degrees

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars: # or self.radars_for_draw
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def pixel_at(self,x,y):
        try:
            return self.map.get_at((x,y))
        except:
            return (255, 255, 255, 255)

    def check_collision(self, map=None):
        self.is_alive = True
        for p in self.four_points:
            if self.pixel_at(int(p[0]), int(p[1])) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map=None):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not self.pixel_at(x, y) == (255, 255, 255, 255) and len < 200:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def check_checkpoint(self):
        p = check_point[self.current_check]
        self.prev_distance = self.cur_distance
        dist = get_distance(p, self.center)
        if dist < 70:
            self.current_check += 1
            self.prev_distance = 9999
            self.check_flag = True
            self.last_checkpoint_time = self.time_spent  # Record the time when checkpoint was reached
            if self.current_check >= len(check_point):
                self.current_check = 0
                self.goal = True
            else:
                self.goal = False

        self.cur_distance = dist

    def update(self, map=None):
        # Get the minimum radar distance to detect proximity to walls
        min_radar = 200
        if self.radars:
            min_radar = min([r[1] for r in self.radars])
        
        # Natural deceleration - lighter in general
        base_deceleration = 0.2  # Reduced from 0.3
        
        # Milder deceleration in curves
        if min_radar < 70:  # Only apply extra deceleration when very close to walls
            # Less aggressive curve deceleration
            curve_deceleration = base_deceleration * (1 + (70 - min_radar) / 100)
            self.speed -= curve_deceleration
        else:
            # Normal deceleration on straight sections
            self.speed -= base_deceleration
        
        # Speed limits - increased back to original with small buffer
        if self.speed > 9.5: self.speed = 9.5  # Allow speeds close to max (10)
        if self.speed < 0: self.speed = 0  # Allow car to come to a complete stop
        
        # Update position based on speed and angle
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        
        # Only update position if speed > 0
        if self.speed > 0:
            self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
            self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
            self.distance += self.speed
        
        # Track time regardless of movement
        self.time_spent += 1
        
        # Boundary checking
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120
            
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # Calculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        # Required for car operation
        self.check_collision(self.map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, self.map)

    def get_data(self):
        # Return raw radar distances (continuous values)
        radars = self.radars
        ret = []
        for r in radars:
            ret.append(r[1])  # Raw distance in pixels
        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        # Base reward on distance traveled
        return self.distance / 50.0

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


class PyRace2DContinuous:
    def __init__(self, is_render = True, car = True, mode = 0):
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.game_speed = 60
        self.mode = mode
        self.font = pygame.font.SysFont("Arial", 30)
        
        # Load game map
        try:
            self.map = pygame.image.load('race_track_ie.png').convert_alpha()
        except:
            self.map = pygame.image.load('./race_track_ie.png').convert_alpha()
        
        # Initialize car
        try:
            self.car = Car('car.png', self.map, [700, 650])
        except:
            self.car = Car('./car.png', self.map, [700, 650])
            
        self.cars = [self.car]

    def action(self, action):
        # Get minimum radar distance to detect proximity to walls
        min_radar = 200
        if self.car.radars:
            min_radar = min([r[1] for r in self.car.radars])
            
        # Speed factor based on proximity to walls - less restrictive
        # Only slightly reduce acceleration near walls
        wall_factor = min(1.0, min_radar / 70)  # Less aggressive reduction
        
        # Expanded action space with less restrictive adjustments
        if action == 0:   # Accelerate
            # Higher base acceleration with milder wall penalty
            acceleration = 2.2 * (0.7 + 0.3 * wall_factor)  # At least 70% effective even near walls
            self.car.speed += acceleration
        elif action == 1: # Turn left
            # Turning is more effective at higher speeds but capped
            turn_rate = 5 + min(3, self.car.speed / 3)  # Cap turn enhancement to avoid over-steering
            self.car.angle += turn_rate
        elif action == 2: # Turn right
            # Turning is more effective at higher speeds but capped
            turn_rate = 5 + min(3, self.car.speed / 3)  # Cap turn enhancement to avoid over-steering
            self.car.angle -= turn_rate
        elif action == 3: # Brake - less powerful to encourage speed
            self.car.speed -= 1.5  # Reduced from 2.0

        self.car.update()
        self.car.check_collision()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(-90, 120, 45):
            self.car.check_radar(d)

    def evaluate(self):
        reward = 0
        
        # Crashed - still penalize crashes but less severely
        if not self.car.is_alive:
            # Less aggressive crash penalty
            speed_penalty = self.car.speed * self.car.speed * 50  # Reduced from 100
            reward = -5000 - speed_penalty + self.car.distance * 0.8  # More credit for distance
        
        # Completed lap - big reward with time bonus
        elif self.car.goal:
            time_factor = max(0, 1 - (self.car.time_spent / 2000))  # Time efficiency bonus
            reward = 10000 + (5000 * time_factor)
            
        # Checkpoint reached - provide progress reward
        elif self.car.check_flag:
            self.car.check_flag = False
            checkpoint_reward = 1000  # Base reward for reaching checkpoint
            
            # Add time efficiency bonus for reaching checkpoint quickly
            time_since_last = self.car.time_spent - self.car.last_checkpoint_time
            if time_since_last > 0:
                time_bonus = max(0, 500 - time_since_last)
                checkpoint_reward += time_bonus
                
            reward = checkpoint_reward
            
        # Small positive reward for moving forward and staying alive
        else:
            # Calculate how close to walls the car is (using minimum radar distance)
            min_radar_distance = min([r[1] for r in self.car.radars]) if self.car.radars else 200
            
            # More generous speed reward - higher optimal speed
            if self.car.speed <= 8:
                # Reward speeds up to 8 linearly
                reward = self.car.speed * 1.0  # Increased multiplier
            else:
                # Only apply speed penalties when very close to walls at high speed
                speed_over_limit = self.car.speed - 8
                
                # Less aggressive wall-speed penalty
                wall_factor = max(0, (60 - min_radar_distance) / 60)  # Only penalize when very close
                speed_penalty = speed_over_limit * wall_factor * 2.0  # Reduced from 5.0
                
                # Base reward for speed minus the smaller penalty
                reward = self.car.speed * 1.0 - speed_penalty
            
            # Smaller penalty for very slow movement
            if self.car.speed < 2:
                reward -= 0.5  # Reduced from 1.0
                
        return reward

    def is_done(self):
        if not self.car.is_alive or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            return True
        return False

    def observe(self):
        # Return continuous state observation
        # [radar1, radar2, radar3, radar4, radar5, speed, angle]
        radars = self.car.radars
        state = []
        
        # Ensure we have exactly 5 radar readings
        # If fewer are available, fill with max range values (200)
        for i in range(5):
            if i < len(radars):
                state.append(radars[i][1])  # Raw distance in pixels
            else:
                state.append(200.0)  # Default to max distance
            
        # Add current speed and angle to the state
        state.append(self.car.speed)      # Current speed (0-10)
        state.append(self.car.angle % 360)  # Current angle normalized to 0-360
        
        # Ensure all values are float32
        return np.array(state, dtype=np.float32)

    def view_(self, msgs=[]):
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3
                if event.key == pygame.K_p:
                    self.mode += 1
                    self.mode = self.mode % 3
                elif event.key == pygame.K_q:
                    done = True
                    exit()

        self.screen.blit(self.map, (0, 0))

        if self.mode == 1:
            self.screen.fill((0, 0, 0))
            
        if len(self.cars) == 1:
            pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        
        # Draw cars
        for car in self.cars:
            if car.get_alive():
                car.draw(self.screen)

        # Display messages
        for k, msg in enumerate(msgs):
            myfont = pygame.font.SysFont("impact", 20)
            label = myfont.render(msg, 1, (0, 0, 0))
            self.screen.blit(label, (1055, 290 + k * 25))

        text = self.font.render("Press 'm' to change view mode", True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.topleft = (750, 0)
        self.screen.blit(text, text_rect)

        # Update the display
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
    return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)) 