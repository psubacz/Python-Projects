import sys
import pygame
from bullet import Bullet
from alien import Alien
from time import sleep

def check_events(ai_settings, screen, stats, sb, play_button,ship, aliens, bullets):
    """Respond to keypresses and mouse events."""
    #Watch for keyboard and mouse events.
    for event in pygame.event.get():    
        if event.type == pygame.QUIT:
            print("Debug:System Exit")
            pygame.display.quit()
            sys.exit()
            
        elif event.type == pygame.KEYDOWN:
            check_keydown_events(event, ai_settings, screen, ship, bullets)
            
        elif event.type == pygame.KEYUP:
            check_keyup_events(event, ship,)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            check_play_button(ai_settings, screen, stats, sb, play_button, ship, aliens, bullets, mouse_x, mouse_y)

        

def check_keydown_events(event, ai_settings, screen, ship, bullets):
    """Respond to keypresses"""
   
    if event.key == pygame.K_RIGHT:
        #Move the ship to the right
        #print("Debug: Function move right.")
##      ship.rect.centerx +=1
        ship.moving_right = True
        
    elif event.key == pygame.K_LEFT:
        #Move the ship to the LEFT
        #print("Debug: Function move left.")
##      ship.rect.centerx += -1
        ship.moving_left = True
        
    elif event.key == pygame.K_SPACE:
        fire_bullet(ai_settings,screen,ship,bullets)

def create_fleet(ai_settings, screen, ship, aliens):
    """Create a full fleet of aliens"""
    # Create an alien and find the number of aliends in a row.
    # Spacing Between each alien os equal to one alien width.
    alien = Alien(ai_settings, screen)
    number_aliens_x = get_number_aliens_x(ai_settings, alien.rect.width)
    number_rows = get_number_rows(ai_settings, ship.rect.height, alien.rect.height)
    #number_rows= 3
    
    #Create the first row of aliens.
    for row_number in range(number_rows):
        
        for alien_number in range(number_aliens_x):
            create_alien(ai_settings, screen, aliens, alien_number, row_number)

def get_number_rows(ai_settings, alien_height, ship_height):
    """Determine the number of rows of aliens that fit on the screen."""
    available_space_y = (ai_settings.screen_height - (3 * alien_height) -ship_height)
    number_rows = int(available_space_y/(2*alien_height))
    if number_rows > 4:
        number_rows = 4
    return number_rows

def create_alien(ai_settings, screen, aliens, alien_number, row_number):
    """Create an  alien and place it in the  row."""
    alien = Alien(ai_settings, screen)
    alien_width = alien.rect.width
    alien.x = alien_width + 2 * alien_width * alien_number
    alien.rect.x = alien.x
    alien.rect.y = alien.rect.height + 2 * alien.rect.height * row_number
    aliens.add(alien)

def get_number_aliens_x(ai_settings, alien_width):
    """Determine the number of aliens that fit in a row."""
    available_space_x = ai_settings.screen_width - 2*alien_width
    number_aliens_x = int(available_space_x /(2 * alien_width))
    return number_aliens_x
    
def update_bullets(ai_settings, screen,stats, sb, ship, aliens, bullets):
    """Update position of bullets and get rif of old bullets."""
    # Update bullet position
    bullets.update()
    
    #get rid of bullets that have disappeared.
    for bullet in bullets.copy():
        if bullet.rect.bottom <= 0:
            bullet.remove(bullets)
    #    print("Bullets on screen: ",len(bullets))
    #Check for any bullets that have hit the aliens.
    #if som delete the bullet and the alien.

    check_bullet_alien_collision(ai_settings, screen, stats, sb, ship, aliens, bullets)

def check_bullet_alien_collision(ai_settings, screen, stats, sb, ship, aliens, bullets):
    """Respond to bullet collisions."""
    collision = pygame.sprite.groupcollide(bullets, aliens, True, True)

    if len(aliens)== 0:
        ai_settings.initialize_dynamic_settings()
        #destroy existing bullets and create a new fleet.
        bullets.empty()
        create_fleet(ai_settings, screen, ship, aliens)
        print("The Alien Fleet is relentless, More Approach!")

        #Increase level.
        stats.level += 1
        sb.prep_level()

    if collision:
        for aliens in collision.values():
            stats.score+=ai_settings.alien_points * len(aliens)
            sb.prep_score()
        check_high_score(stats, sb)

            
def check_high_score(stats,sb):
    """Check to see if there's a new high score."""
    if stats.score > stats.high_score:
        stats.high_score = stats.score
        sb.prep_high_score()
        
        
def fire_bullet(ai_settings,screen,ship,bullets):
    """Fire a bullet."""
    #Create a new bullet and add it to the bullets group
    if len(bullets) < ai_settings.bullets_allowed:
        new_bullet = Bullet(ai_settings, screen, ship)
        bullets.add(new_bullet)

def check_keyup_events(event, ship):
    """Respond to key release"""
    if event.key == pygame.K_RIGHT:
        ship.moving_right = False
    
    elif event.key == pygame.K_LEFT:
        ship.moving_left = False
                    
def update_screen(ai_settings,screen, stats, sb, ship, aliens, bullets, play_button):
    """Update images on the screen and flip to the new screen."""
    #Draw the play button if the game is inactive

    #Redraw the screen during each pass through the loop.
    screen.fill(ai_settings.bg_color)
    ship.blitme()
    aliens.draw(screen)
 
    #Draw the score information.
    sb.show_score()

    if not stats.game_active:
        play_button.draw_button()
        
    for bullet in bullets.sprites():
        bullet.draw_bullet()


    #Make the most recently drawn screen visible.
    pygame.display.flip()


    
def check_fleet_edges(ai_settings, aliens):
    """Respond to aliens reaching the edge."""
    for alien in aliens.sprites():
        if alien.check_edges():
            change_fleet_direction(ai_settings, aliens)
            break
        
def change_fleet_direction(ai_settings, aliens):
    """Drop the entire fleet and change the fleet's direction."""
    for alien in aliens.sprites():
        alien.rect.y += ai_settings.fleet_drop_speed
    ai_settings.fleet_direction *= -1

def update_aliens(ai_settings, stats, screen, sb, ship, aliens, bullets):
    """check if th fleet is at the edge and then update the position if al aliends in the fleet."""
    check_fleet_edges(ai_settings, aliens)
    aliens.update()

    #Look for alien collisions.
    if pygame.sprite.spritecollideany(ship,aliens):
        ship_hit(ai_settings, stats, screen, sb, ship, aliens, bullets)
        
    check_aliens_buttom(ai_settings, screen, stats ,sb, ship, aliens, bullets)


def ship_hit(ai_settings, stats, screen, sb, ship, aliens, bullets):
    """Respond to the ship being hit."""
    #decrement the number of ships left.
    if stats.ships_left > 0:
        stats.ships_left -= 1

        # Empty the list of aliens and bullets.
        aliens.empty()
        bullets.empty()

        #Create a new fleet and center the ship.
        create_fleet(ai_settings, screen, ship, aliens)
        ship.center_ship()

        #Update scoreboard.
        sb.prep_ships()

        #Pause.
        sleep(0.5)

    else:
        stats.game_active = False

def check_aliens_buttom(ai_settings, screen, stats ,sb, ship, aliens, bullets):
    """Check if any aliens have reached the buttom of the screen."""
    screen_rect = screen.get_rect()
    for alien in aliens.sprites():
        if alien.rect.bottom >= screen_rect.bottom:
            #treat this the same as if the ship got hit.
            ship_hit(ai_settings, screen, stats, sb, ship, aliens, bullets)
            break
        

def check_play_button(ai_settings, screen, stats, sb, play_button, ship, aliens, bullets, mouse_x, mouse_y):
    """Start a new game when the player clikcs Play."""
    button_clicked = play_button.rect.collidepoint(mouse_x, mouse_y)
    if button_clicked and not stats.game_active:
        #hide the mouse cursor
        pygame.mouse.set_visible(False)
    #Reset the game statistics
    stats.reset_stats()
    stats.game_active = True

    #Reset the scoreboard images.
    sb.prep_score()
    sb.prep_high_score()
    sb.prep_level()
    sb.prep_ships()
    
    #empty the list of aliens and bullets.
    aliens.empty()
    bullets.empty()

    # Create a new fleet and center the ship.
    create_fleet(ai_settings,screen,ship,aliens)
    ship.center_ship()
    
