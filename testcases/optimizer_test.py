decay = .75
decay_step = 1000
l = lambda step: decay ** (step // decay_step)
print(10//decay_step)
for step in range(10000):
    if step % 1000 == 0:
        print(l(step))