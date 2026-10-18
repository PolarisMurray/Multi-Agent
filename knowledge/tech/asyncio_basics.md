# asyncio 常见问题

asyncio 适合 I/O 密集型任务，例如同时调用多个大模型接口。CPU 密集型计算仍然会阻塞事件循环，应该放到线程池或进程池里。

asyncio.gather 可以并发等待多个协程；如果需要限制并发数，用 asyncio.Semaphore 包住每个任务。

在协程里调用同步的阻塞函数（例如 time.sleep、同步数据库驱动）会卡住整个事件循环，可以用 asyncio.to_thread 把它放到线程里执行。

asyncio.Lock 只在同一个事件循环内有效，不能跨进程使用；跨进程或跨机器的互斥需要分布式锁，例如 Redis 的 SET NX PX。
