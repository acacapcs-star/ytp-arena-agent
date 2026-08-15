import math


def acceleration_control(state, target_x, target_y):
    ship_x = state["position"][0]
    ship_y = state["position"][1]

    vx = state["velocity"][0]
    vy = state["velocity"][1]

    max_accel = state["config"]["max_accel"]

    dx = target_x - ship_x
    dy = target_y - ship_y

    distance = math.sqrt(dx * dx + dy * dy)

    if distance == 0:
        return 0, 0

    dir_x = dx / distance
    dir_y = dy / distance

    speed = math.sqrt(vx * vx + vy * vy)

    speed_towards_target = vx * dir_x + vy * dir_y

    if speed_towards_target > 0:
        brake_distance = (
            speed_towards_target * speed_towards_target
        ) / (2 * max_accel)
    else:
        brake_distance = 0


    # 靠近 chest，開始繞圈減速
    if distance < 100:

        side_x = -dir_y
        side_y = dir_x

        ax = side_x * max_accel
        ay = side_y * max_accel

        if speed > 0:
            ax += -vx / speed * max_accel * 0.5
            ay += -vy / speed * max_accel * 0.5


    elif distance <= brake_distance and speed_towards_target > 0:
        ax = -vx / speed * max_accel
        ay = -vy / speed * max_accel

    else:
        ax = dir_x * max_accel
        ay = dir_y * max_accel


    accel = math.sqrt(ax * ax + ay * ay)

    if accel > max_accel:
        ax = ax / accel * max_accel
        ay = ay / accel * max_accel

    return ax, ay