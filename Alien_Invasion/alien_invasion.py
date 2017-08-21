import sys
import pygame
from settings import Settings
from game_stats import GameStats
from ship import Ship
#from alien import Alien
import game_functions as gf
from pygame.sprite import Group
from button import Button
from scoreboard import Scoreboard


#This is the main class of the alien invasion game project of the PythonCrashCourse 

def run_game():
    #Initialize game and  create a screen object
    pygame.init()
    ai_settings = Settings()
    screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption("Alien Invasion")

    #Set the background color
    bg_color = (0,255,0)

    #Make the Play button.
    play_button = Button(ai_settings, screen, "Play")

    #Make a ship.
    ship = Ship(ai_settings,screen)

    #Make a group to store bullets and aliens in.
    bullets = Group()
    aliens = Group()

    #Create an instance to store game statistics.
    stats = GameStats(ai_settings)
    sb = Scoreboard(ai_settings, screen, stats)

    #Create the fleet of aliens.
    gf.create_fleet(ai_settings, screen, ship, aliens)
     
    #Start the main loop for the game.
    while True:
        gf.check_events(ai_settings, screen, stats, sb, play_button, ship, aliens, bullets)

        if stats.game_active:
            ship.update()
            gf.update_bullets(ai_settings, screen,stats, sb, ship, aliens, bullets)
            gf.update_aliens(ai_settings, stats, screen, sb, ship, aliens, bullets)

        gf.update_screen(ai_settings, screen, stats, sb, ship, aliens, bullets, play_button)
        
 
run_game()
