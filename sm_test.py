import pygame
import pyspacemouse

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))

# Open the first detected SpaceMouse
with pyspacemouse.open() as device:
    running = True
    while running:
        # Standard Pygame Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Read SpaceMouse state (returns 6DoF data)
        state = device.read()
        
        # Example: Use X and Y axis to move an object
        # Values typically range from -1.0 to 1.0
        pos_x = state.x * 10
        pos_y = state.y * 10
        
        # Update your Pygame surfaces here...
        pygame.display.flip()

