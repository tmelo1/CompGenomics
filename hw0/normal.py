from numpy import *

sum = 0;
for i in range(11):
    r = random.normal(0, 1)
    sum = sum + r
    print(r)
print 'sum = %d' % (sum) 

    
