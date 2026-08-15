import math

def planet_strategy(state, enemy_x, enemy_y, target_x, target_y):
    ship_x = state["position"][0]
    ship_y = state["position"][1]

    near_planet = None
    min_distance = 999999

    for planet in state["obstacles"]:
        planet_x = planet["center"][0]
        planet_y = planet["center"][1]

        dx = planet_x - enemy_x
        dy = planet_y - enemy_y

        distance = math.sqrt(dx * dx + dy * dy)

        if distance < min_distance:
            min_distance = distance
            near_planet = planet

    if near_planet == None:
        return target_x, target_y

    if min_distance > 100:
        return target_x, target_y

    planet_x = near_planet["center"][0]
    planet_y = near_planet["center"][1]

    dx = planet_x - ship_x
    dy = planet_y - ship_y

    distance = math.sqrt(dx * dx + dy * dy)

    if distance == 0:
        return target_x, target_y

    dir_x = dx / distance
    dir_y = dy / distance

    side_x = -dir_y
    side_y = dir_x

    avoid_distance = 100

    new_target_x = planet_x + side_x * avoid_distance
    new_target_y = planet_y + side_y * avoid_distance

    return new_target_x, new_target_y