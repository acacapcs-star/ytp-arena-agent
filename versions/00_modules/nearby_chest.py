import math
from acceleration import acceleration_control


target_x = None
target_y = None


def nearby_chest(state):
    global target_x
    global target_y

    ship_x = state["position"][0]
    ship_y = state["position"][1]
    ship_radius = state["config"]["ship_radius"]

    # 已經有鎖定的 chest
    if target_x != None and target_y != None:

        dx = target_x - ship_x
        dy = target_y - ship_y

        distance = math.sqrt(dx * dx + dy * dy)

        target_chest = None

        for chest in state["chests"]:
            chest_x = chest["center"][0]
            chest_y = chest["center"][1]

            if chest_x == target_x and chest_y == target_y:
                target_chest = chest
                break

        # chest 還存在
        if target_chest != None:
            chest_radius = target_chest["radius"]

            # 還沒碰到，繼續追同一顆
            if distance > chest_radius + ship_radius:
                return acceleration_control(
                    state,
                    target_x,
                    target_y
                )

        # 已經吃到或 chest 不見了
        target_x = None
        target_y = None


    # 找新的最近 chest
    min_distance = 999999
    target_chest = None

    for chest in state["chests"]:
        chest_x = chest["center"][0]
        chest_y = chest["center"][1]
        chest_radius = chest["radius"]

        dx = chest_x - ship_x
        dy = chest_y - ship_y

        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= chest_radius + ship_radius:
            continue

        if distance < min_distance:
            min_distance = distance
            target_chest = chest


    if target_chest == None:
        return 0, 0


    target_x = target_chest["center"][0]
    target_y = target_chest["center"][1]

    return acceleration_control(
        state,
        target_x,
        target_y
    )