
import ray
import time

ray.init()

# A regular Python function.
def regular_function():
    return 1

# A Ray remote function.
@ray.remote
def remote_function():
    print('done.')
    return 1

remote_function.remote()

result = ray.get(remote_function.remote())

time.sleep(3)
print(result)