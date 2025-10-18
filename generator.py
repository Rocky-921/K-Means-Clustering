import os
import random
import time
import sys

def generate_test_case(N, d, k, file_path, value_range=(-10**2, 10**2), seed=None):
    if seed is not None:
        random.seed(seed)

    with open(file_path, "w") as f:
        f.write(f"{N}\n")
        f.write(f"{d}\n")
        f.write(f"{k}\n")

        for _ in range(N):
            point = [str(random.randint(*value_range)) for _ in range(d)]
            f.write(" ".join(point) + "\n")


# Example parameters — can be randomized per case if desired
N = int(sys.argv[1])
d = int(sys.argv[2])
k = int(sys.argv[3])
folder = sys.argv[4]
tc_no = sys.argv[5]

file_path = os.path.join(folder, f"{tc_no}")
generate_test_case(N, d, k, file_path, seed=time.time())
print(f"Generated test case {tc_no} -> {file_path}")


