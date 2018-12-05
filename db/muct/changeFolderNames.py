import os

start = 467
end = 477
move = 202

for i in range(start, end+1):
    old_name = str(i)
    new_name = str(i - move)
    os.rename(old_name, new_name)
    print(i)
