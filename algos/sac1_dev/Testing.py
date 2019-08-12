
import ray
import time

ray.init(object_store_memory=1000000000, redis_max_memory=1000000000)

y=111
y_id = ray.put(y)


@ray.remote
class Cat:
    def __init__(self):
        self.cnt = 0
        global y_id
        y_id = ray.put(2)
    def incre(self):
        print('done.')
        time.sleep(1)
        self.cnt += ray.get(y_id)
    def get_cnt(self):
        return self.cnt

cat = Cat.remote()



class Dog:
    def __init__(self):
        self.cnt = 0
        global y_id
        y_id = ray.put(2)
    def incre(self):
        print('done.')
        time.sleep(1)
        self.cnt += ray.get(y_id)
    def get_cnt(self):
        return self.cnt

dog = Dog()


@ray.remote
def remote_cat(cls1):
    cls1.incre.remote()  # self.cnt will increase
    return 1 # cls1.get_cnt.remote()

@ray.remote
def remote_dog(cls1):
    cls1.incre()    # self.cnt will not increase
    return 1 # cls1.get_cnt.remote()



result_id = [remote_dog.remote(dog) for _ in range(5)]

result = ray.get(result_id)

print(result)

time.sleep(5)
# print(ray.get(cat.get_cnt.remote()))
print(dog.get_cnt())