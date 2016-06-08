import numpy as np
import time
import copy

rng = np.random.RandomState(333)

length = 1000000
a = []
for i in xrange(length):
	a.append(rng.randint(length))

b = copy.deepcopy(a)
c = copy.deepcopy(a)
print "Finished building list"

ops_count = 10000

start_time = time.time()
# for i in xrange(ops_count):
# 	a = a[1 :]
# 	a.append(rng.randint(length))
# 	if (i + 1) % (ops_count // 10) == 0:
# 		print "Finished %d%%" % ((i + 1) * 100 // ops_count)
end_time = time.time()

slice_time = end_time - start_time

print "ops_count =", ops_count
print "slice_time =", slice_time
print "slice_time per op =", slice_time / float(ops_count)

start_time = time.time()
for i in xrange(ops_count):
	del b[0]
	b.append(rng.randint(length))
	if (i + 1) % (ops_count // 10) == 0:
		print "Finished %d%%" % ((i + 1) * 100 // ops_count)
end_time = time.time()

del_time = end_time - start_time

print "ops_count =", ops_count
print "del_time =", del_time
print "del_time per op =", del_time / float(ops_count)

start_time = time.time()
for i in xrange(ops_count):
	c.pop(0)
	c.append(rng.randint(length))
	if (i + 1) % (ops_count // 10) == 0:
		print "Finished %d%%" % ((i + 1) * 100 // ops_count)
end_time = time.time()

pop_time = end_time - start_time

print "ops_count =", ops_count
print "pop_time =", pop_time
print "pop_time per op =", pop_time / float(ops_count)
