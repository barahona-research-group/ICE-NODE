from time import sleep
from tqdm import tqdm
# from tqdm.notebook import tqdm
# print("Manual nesting")
print(get_ipython())
for i in tqdm(range(16)):
    for j in tqdm(range(16), leave=False):
        sleep(0.05)
