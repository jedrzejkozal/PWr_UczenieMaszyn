import os

start = 383
end = 397
for i in range(start, end):
    old_name = str(i)
    new_name = str(i-4)
    os.rename(old_name, new_name)
    print(i)
