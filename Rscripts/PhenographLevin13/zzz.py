import matplotlib.pyplot as plt
import numpy as np
import functools
import time

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        res = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        return (res, elapsedTime)
    return newfunc

@timeit
def naive_power(m, n):
    m = np.asarray(m)
    res = m.copy()
    for i in range(1,n):
        res *= m
    return res

@timeit
def fast_power(m, n):
    # elementwise power
    return np.power(m, n)

m = np.random.random((100,100))
n = 10

rs1 = []
ts1 = []
ts2 = []
for i in range(1, n):
    r1, t1 = naive_power(m, i)
    ts1.append(t1)

for i in range(1, n):
    r2, t2 = fast_power(m, i)
    ts2.append(t2)

plt.figure()
plt.plot(ts1, label='naive')
plt.plot(ts2, label='numpy')
plt.xlabel('exponent')
plt.ylabel('time')
plt.legend(loc='upper left')