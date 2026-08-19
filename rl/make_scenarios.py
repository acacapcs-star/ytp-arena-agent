import random, os
os.makedirs("scenarios", exist_ok=True)

def write(name, obstacles, seed):
    rng = random.Random(seed)
    lines = ["L,100", "player,5,5", "player,95,95"]
    for _ in range(20):
        x = rng.uniform(10, 90); y = rng.uniform(10, 90)
        lines.append(f"chest,{x:.2f},{y:.2f},1.20")
    for (ox, oy, orr) in obstacles:
        lines.append(f"obstacle,{ox},{oy},{orr}")
    open(f"scenarios/{name}.csv","w").write("\n".join(lines)+"\n")

write("open", [], 1)
write("gravity", [(30,30,10),(70,70,10),(30,70,10),(70,30,10),(50,50,12)], 7)
print(open("scenarios/gravity.csv").read()[:200])
