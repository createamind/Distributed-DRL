
import ray, os, time

ray.init(object_store_memory=1000000000, redis_max_memory=1000000000)

print('main.pid:', os.getpid())

@ray.remote
def f(x):
    print('f.pid:', os.getpid())
    return x

@ray.remote
class Foo():
    def __init__(self, f):
        self.x = ray.get(f.remote(100))

    # @ray.remote     # AttributeError: 'ActorHandle' object has no attribute 'bar'
    def bar(self):
        print('bar.pid:', os.getpid())
        return 1

foo = Foo.remote(f)

obj_id1 = foo.bar.remote()

print(ray.get(obj_id1))


''' outputs:
main.pid: 9521
1
(pid=9593) bar.pid: 9593
(pid=9602) f.pid: 9602
'''



@ray.remote
class Counter(object):
    def __init__(self):
        self.counter = 0

    def inc(self):
        self.counter += 1

    def get_counter(self):
        return self.counter

@ray.remote
def g(counter):
    print('g.pid:', os.getpid())
    for _ in range(1000):
        time.sleep(0.1)
        counter.inc.remote()

counter = Counter.remote()

# Start some tasks that use the actor.
[g.remote(counter) for _ in range(3)]

# Print the counter value.
for _ in range(10):
    time.sleep(1)
    print(ray.get(counter.get_counter.remote()))