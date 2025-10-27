import pygame
import random
import sys
import os

pygame.init()

WIDTH, HEIGHT = 800, 600
GRID_SIZE = 30
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 12

DARK_GREEN = (0, 100, 0)
LIGHT_GREEN = (50, 205, 50)
RED = (255, 0, 0)
DARK_BLUE = (25, 25, 112)
LIGHT_BLUE = (135, 206, 250)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🐍 Professional Snake Game")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.reset_game()
    
    def reset_game(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)
        self.next_direction = (1, 0)
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
        self.game_started = False
        self.speed = FPS
    
    def generate_food(self):
        while True:
            food = (random.randint(1, GRID_WIDTH - 2), 
                   random.randint(1, GRID_HEIGHT - 2))
            if food not in self.snake:
                return food
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if not self.game_started and event.key == pygame.K_SPACE:
                    self.game_started = True
                elif event.key == pygame.K_r and self.game_over:
                    self.reset_game()
                    self.game_started = True  
                elif self.game_started and not self.game_over:
                    if event.key == pygame.K_UP and self.direction != (0, 1):
                        self.next_direction = (0, -1)
                    elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                        self.next_direction = (0, 1)
                    elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                        self.next_direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                        self.next_direction = (1, 0)
                    elif event.key == pygame.K_p:  
                        self.game_started = False
    
    def update(self):
        if not self.game_started or self.game_over:
            return
        
        self.direction = self.next_direction
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.game_over = True
            return
        
        if new_head in self.snake:
            self.game_over = True
            return
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 10
            self.food = self.generate_food()
            if self.score % 50 == 0 and self.speed < 20:
                self.speed += 1
        else:
            self.snake.pop()
    
    def draw_background(self):
        self.screen.fill(BLACK)
        
        for y in range(0, HEIGHT, 2):
            color_value = max(0, 50 - y // 20)
            pygame.draw.line(self.screen, (color_value, color_value, 50 + color_value), 
                           (0, y), (WIDTH, y), 2)
        
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 80), (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (40, 40, 80), (0, y), (WIDTH, y), 1)
    
    def draw_snake(self):
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                color = LIGHT_GREEN
            else:
                color_intensity = max(100, 255 - (i * 5))
                color = (0, color_intensity, 0)
            
            pygame.draw.rect(self.screen, color, 
                           (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(self.screen, WHITE, 
                           (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)
        
        if self.snake:
            head_x, head_y = self.snake[0]
            eye_size = GRID_SIZE // 5
            dx, dy = self.direction
            
            if dx == 1: 
                left_eye = (head_x * GRID_SIZE + GRID_SIZE - eye_size - 2, head_y * GRID_SIZE + eye_size * 2)
                right_eye = (head_x * GRID_SIZE + GRID_SIZE - eye_size - 2, head_y * GRID_SIZE + GRID_SIZE - eye_size * 3)
            elif dx == -1: 
                left_eye = (head_x * GRID_SIZE + 2, head_y * GRID_SIZE + eye_size * 2)
                right_eye = (head_x * GRID_SIZE + 2, head_y * GRID_SIZE + GRID_SIZE - eye_size * 3)
            elif dy == 1:  
                left_eye = (head_x * GRID_SIZE + eye_size * 2, head_y * GRID_SIZE + GRID_SIZE - eye_size - 2)
                right_eye = (head_x * GRID_SIZE + GRID_SIZE - eye_size * 3, head_y * GRID_SIZE + GRID_SIZE - eye_size - 2)
            else:  #
                left_eye = (head_x * GRID_SIZE + eye_size * 2, head_y * GRID_SIZE + 2)
                right_eye = (head_x * GRID_SIZE + GRID_SIZE - eye_size * 3, head_y * GRID_SIZE + 2)
            
            pygame.draw.circle(self.screen, WHITE, left_eye, eye_size)
            pygame.draw.circle(self.screen, WHITE, right_eye, eye_size)
            pygame.draw.circle(self.screen, BLACK, left_eye, eye_size // 2)
            pygame.draw.circle(self.screen, BLACK, right_eye, eye_size // 2)
    
    def draw_food(self):
        x, y = self.food
        pygame.draw.circle(self.screen, RED, 
                         (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2), 
                         GRID_SIZE // 2 - 2)
        pygame.draw.rect(self.screen, DARK_GREEN, 
                       (x * GRID_SIZE + GRID_SIZE // 2 - 1, y * GRID_SIZE + 2, 2, 4))
        pygame.draw.circle(self.screen, (255, 150, 150), 
                         (x * GRID_SIZE + GRID_SIZE // 2 - 3, y * GRID_SIZE + GRID_SIZE // 2 - 3), 
                         3)
    
    def draw_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score}", True, GOLD)
        self.screen.blit(score_text, (20, 20))

        speed_text = self.font_small.render(f"Speed: {self.speed}", True, LIGHT_BLUE)
        self.screen.blit(speed_text, (20, 60))
 
        if not self.game_started and not self.game_over:
            title = self.font_large.render("SNAKE GAME", True, GOLD)
            start_text = self.font_medium.render("Press SPACE to Start", True, WHITE)
            controls = self.font_small.render("Arrow Keys: Move • P: Pause • R: Restart", True, LIGHT_BLUE)
            
            self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 3))
            self.screen.blit(start_text, (WIDTH // 2 - start_text.get_width() // 2, HEIGHT // 2))
            self.screen.blit(controls, (WIDTH // 2 - controls.get_width() // 2, HEIGHT // 2 + 50))
        
        elif self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))  
            self.screen.blit(overlay, (0, 0))
            
            game_over = self.font_large.render("GAME OVER", True, RED)
            score_final = self.font_medium.render(f"Final Score: {self.score}", True, GOLD)
            restart = self.font_medium.render("Press R to Play Again", True, WHITE)
            
            self.screen.blit(game_over, (WIDTH // 2 - game_over.get_width() // 2, HEIGHT // 3))
            self.screen.blit(score_final, (WIDTH // 2 - score_final.get_width() // 2, HEIGHT // 2))
            self.screen.blit(restart, (WIDTH // 2 - restart.get_width() // 2, HEIGHT // 2 + 50))
        
        elif self.game_started and not self.game_over:
            controls_hint = self.font_small.render("P: Pause • R: Restart", True, LIGHT_BLUE)
            self.screen.blit(controls_hint, (WIDTH - controls_hint.get_width() - 20, 20))
    
    def draw(self):
        self.draw_background()  
        self.draw_snake()
        self.draw_food()
        self.draw_ui()
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(self.speed)

if __name__ == "__main__":
    try:
        game = SnakeGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pygame.quit()