import pygame
pygame.init()
print("init done")

screen=pygame.display.set_mode((400, 300))
print("window created")

running = True
while running:
    for event in pygame.event.get():
        print(event)
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 0, 0))
    pygame.display.flip()