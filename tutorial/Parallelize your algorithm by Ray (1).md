

Ray是一个实现分布式python程序的通用框架。Ray提供了统一的任务并行和actor抽象，并通过共享内存、零拷贝序列化和分布式调度实现了高性能。

Ray里面还有用来调超参数的库[Tune](http://ray.readthedocs.io/en/latest/tune.html)和可扩展规模的强化学习库[Rllib](http://ray.readthedocs.io/en/latest/rllib.html)。

ray的必备知识：

1. 使用远程方程（任务） [`ray.remote`]
2. 通过object IDs获取结果 [`ray.put`, `ray.get`, `ray.wait`]
3. 使用远程类 (actors) [`ray.remote`]

使用Ray，可以使你的代码从单机运行轻松地扩展到大集群上运行。

使用该命令安装Ray：`pip install -U ray`



开始使用ray，导入ray，然后初始化。

```python
import ray

# Start Ray. If you're connecting to an existing cluster, you would use
# ray.init(address=<cluster-address>) instead.
ray.init()
```



1. 使用远程方程（任务） [`ray.remote`]

将python函数转换为远程函数的标准方法使在函数上面添加一个`@ray.remote`装饰器。下面看一个例子。

```python
# A regular Python function.
def regular_function():
    return 1

# A Ray remote function.
@ray.remote
def remote_function():
    return 1
```

```python
assert regular_function() == 1

object_id = remote_function.remote()

# The value of the original `regular_function`
assert ray.get(object_id) == 1
```

**Parallelism:** Invocations of `regular_function` happen **serially**, for example

在调用的时候，普通函数将串行运行。

```python
# These happen serially.
for _ in range(4):
    regular_function()
```

 

whereas invocations of `remote_function` happen in **parallel**, for example

调用远程函数时，程序将并行运行。

```python
# These happen in parallel.
for _ in range(4):
    remote_function.remote()
```



Oftentimes, you may want to specify a task’s resource requirements (for example
one task may require a GPU). The `ray.init()` command will automatically
detect the available GPUs and CPUs on the machine. However, you can override
this default behavior by passing in specific resources, e.g.

运行`ray.init()`后，ray将自动检查可用的GPU和CPU。我们也可以给我们传入参数设置特定的资源需求量。

`ray.init(num_cpus=8, num_gpus=4, resources={'Custom': 2})`

远程函数/类也可以设置资源需求量，像这样`@ray.remote(num_cpus=2, num_gpus)`

如果没有设置，默认设置为1个CPU。

If you do not specify any resources in the `@ray.remote` decorator, the
default is 1 CPU resource and no other resources.



远程函数执行后并不会直接返回结果，而是会立即返回一个object ID。远程函数会在后台并行处理，等执行得到最终结果后，可以通过返回的object ID取得这个结果。

`ray.put(*value*)`也会返回object ID

put操作将对象存入object store里，然后返回它的object ID。

Store an object in the object store.  return: The object ID assigned to this value.

```python
y = 1
object_id = ray.put(y)
```







通过object IDs获取结果 [`ray.put`, `ray.get`, `ray.wait`]

ray.get(obj_id)

从object store获取远程对象或者一个列表的远程对象。

Get a remote object or a list of remote objects from the object store.

Then, if the object is a numpy array or a collection of numpy arrays, the `get` call is zero-copy and returns arrays backed by shared object store memory.
Otherwise, we deserialize the object data into a Python object.

This method blocks until the object corresponding to the object ID is
available in the local object store.

需要注意的是，使用get方法时会锁，直到要取得的对象在本地的object store里可用。

调用remote操作是异步的，他们会返回object IDs而不是结果。想要得到真的的结果我们需要使用ray.get()。

我们之前写的这段语句，实际上results是一个由object IDs组成的列表。

`results = [do_some_work.remote(x) for x in range(4)]`

如果改为下面，ray.get()将通过object ID取得真实的结果。

`results = [ray.get(do_some_work.remote(x)) for x in range(4)]`

但是，这样写会有一个问题。ray.get()会锁进程，这意味着，ray.get()会一直等到do_some_work这个函数执行完返回结果后才执行结束然后进入下一个循环。这样的话，4次调用do_some_work函数就不再是并行运行的了。

为了可以并行运算，我们需要在调用完所有的任务后再调用ray.get()。像下面这样。

`results = ray.get([do_some_work.remote(x) for x in range(4)])`

所以，需要小心使用ray.get()。因为它是一个锁进程的操作。如果太频繁调用ray.get()，将会影响并行性能。同时，尽可能的晚使用ray.get()以防止不必要的等待。



Recall that remote operations are asynchronous and they return futures (i.e., object IDs) instead of the results themselves.To get the actual results,  we need to use ray.get(), and here the first instinct is to just call ray.get() on the remote operation invocation i.e., replace line “results = [do_some_work.remote(x) for x in range(4)]” with: results = [ray.get(do_some_work.remote(x)) for x in range(4)]

The observant reader will already have the answer: ray.get() is blocking, so calling it after each remote operation means that we wait for that operation to complete, which essentially means that we execute one operation at a time, hence no parallelism!

To enable parallelism, we need to call ray.get() *after* invoking all tasks. We can easily do so in our example by replacing line “results = [do_some_work.remote(x) for x in range(4)]” with:

 

```python
results = ray.get([do_some_work.remote(x) for x in range(4)])
```

always keep in mind that ray.get() is a blocking operation, and thus if called eagerly it can hurt the parallelism. Instead, you should try to write your program such that ray.get() is called as late as possible.

**Tip 1:** ***Delay calling ray.get() as much as possible.***



远程类

通过远程类，我们可以实现一个共享的参数服务器。

 remote classes (Actors)

我们在类的定义上面加上修饰器ray.remote。这个类的实例就会是一个Ray的actor。每一个actor运行在自己的python进程上。

Actors extend the Ray API from functions (tasks) to classes. The `ray.remote` decorator indicates that instances of the `Counter` class will be actors. An actor is essentially a stateful worker. Each actor runs in its own Python process.



```python
@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value
```

 You can specify resource requirements in Actors too (see the [Actors section](https://ray.readthedocs.io/en/latest/actors.html) for more details.)

 同样可以给actor设置资源请求量。

```python
@ray.remote(num_cpus=2, num_gpus=0.5)
class Actor(object):
    pass
```

 

We can interact with the actor by calling its methods with the `.remote` operator. We can then call `ray.get` on the object ID to retrieve the actual value.

 在调用类的方法时加上`.remote`，然后使用`ray.get`获取实际的值。

```
obj_id = a1.increment.remote()
ray.get(obj_id) == 1
```

Actor handles can be passed into other tasks. To illustrate this with a
simple example, consider a simple actor definition.

Actor可以作为参数传给别的任务，下面的例子就是实现一个参数服务器。不同的参数就可以公用一个参数服务器了。



ps

The @ray.remote decorator defines a service. It takes the
`ParameterServer` class and allows it to be instantiated as a remote service or
actor.



**Sharding Across Multiple Parameter Servers:** When your parameters are large and your cluster is large, a single parameter server may not suffice because the application could be bottlenecked by the network bandwidth into and out of the machine that the parameter server is on (especially if there are many workers).

当你的参数特别大，而且你的集群也很大，一个parameter server可能就不够了。特别是有很多worker的时候，因为向一个parameter server的数据传输就会成为瓶颈。

简单的解决办法就是把参数分散在多个parameter server上。可以通过创建多个actor来实现。

A natural solution in this case is to shard the parameters across multiple parameter servers. This can be achieved by simply starting up multiple parameter server actors. An example of how to do this is shown in the code example at the bottom.



为了保证ray并行的性能，远程任务应该花费至少几毫秒的时间。

当需要重复向不同远程任务传入相同对象时，可以先用ray.put()把类存入object store，然后传入它的object id。

**Tip 2:** **For exploiting Ray’s parallelism, remote tasks should take at least several milliseconds.**

**Tip 3:** ***When passing the same object repeatedly as an argument to a remote operation, use ray.put() to store it once in the object store and then pass its ID.***

**Tip 4:** **Use ray.wait() to process results as soon as they become available.**





https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users/

https://ray-project.github.io/2018/07/15/parameter-server-in-fifteen-lines.html