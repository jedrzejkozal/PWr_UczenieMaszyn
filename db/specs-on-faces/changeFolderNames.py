import os

start = 74
end = 102
move = 1

for i in range(start, end+1):
    old_name = str(i)
    new_name = str(i - move)
    os.rename(old_name, new_name)
    print(i)
