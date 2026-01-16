#!/home/curl/anaconda3/envs/accel_challenge/bin/python
import keyboard

# raw materials: https://github.com/boppreh/keyboard

while True:
    # Wait for the next event.
    event = keyboard.read_event()
    print('event: ', event)
    # KEY_DOWN: means the button has been pressed
    # KEY_UP: means the button has been released
    if event.event_type == keyboard.KEY_DOWN and event.name == 'space':
        print('space was pressed')
    if event.event_type == keyboard.KEY_UP and event.name == 'space':
        print('space was released')
    if event.event_type == keyboard.KEY_DOWN and event.name == 'up':
        print('up was pressed')
    if event.event_type == keyboard.KEY_DOWN and event.name == 'down':
        print('down was pressed')
    if event.event_type == keyboard.KEY_DOWN and event.name == 'left':
        print('left was pressed')
    if event.event_type == keyboard.KEY_DOWN and event.name == 'right':
        print('right was pressed')