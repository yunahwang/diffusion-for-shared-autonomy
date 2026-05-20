import pygame

pygame.init()
pygame.joystick.init()
print("init done")

screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Joystick Axis Tester")
font = pygame.font.SysFont("monospace", 20)

# Connect joystick
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
if not joysticks:
    print("No joystick found!")
    pygame.quit()
    exit()
joy = joysticks[-1]
joy.init()
print(f"Using joystick: {joy.get_name()}")

WATCH_AXES = {3, 4}
axis_values = {3: 0.0, 4: 0.0}
DEADZONE = 0.1

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.JOYAXISMOTION:
            if event.axis in WATCH_AXES:
                v = 0.0 if abs(event.value) < DEADZONE else event.value
                axis_values[event.axis] = v
                print(f"Axis {event.axis}: raw={event.value:+.3f}  deadzoned={v:+.3f}")

    screen.fill((30, 30, 30))

    # Draw axis 3 (SIDE): horizontal bar
    cx, cy = 200, 120
    bar_w = 150
    pygame.draw.rect(screen, (80, 80, 80), (cx - bar_w, cy - 15, bar_w * 2, 30))
    side_x = int(axis_values[3] * bar_w)
    pygame.draw.rect(screen, (0, 200, 255), (cx, cy - 15, side_x, 30))
    screen.blit(font.render(f"Axis 3 (SIDE): {axis_values[3]:+.2f}", True, (255, 255, 255)), (10, cy - 40))

    # Draw axis 4 (UP): vertical bar
    cx2, cy2 = 200, 220
    bar_w2 = 150
    pygame.draw.rect(screen, (80, 80, 80), (cx2 - bar_w2, cy2 - 15, bar_w2 * 2, 30))
    up_x = int(axis_values[4] * bar_w2)
    pygame.draw.rect(screen, (255, 100, 100), (cx2, cy2 - 15, up_x, 30))
    screen.blit(font.render(f"Axis 4 (UP):   {axis_values[4]:+.2f}", True, (255, 255, 255)), (10, cy2 - 40))

    # Legend
    screen.blit(font.render("Axis 3: left=neg, right=pos", True, (180, 180, 180)), (10, 265))

    pygame.display.flip()

pygame.quit()