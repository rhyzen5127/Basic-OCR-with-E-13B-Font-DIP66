import sys

for idx, path in enumerate(sys.path, 1):
    print(f'{idx} - {path}')