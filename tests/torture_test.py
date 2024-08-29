import matplotlib.pyplot as plt
import numpy as np
import time
i=0
time_elapsed=0
timetaken=[]
max_whack=[]
begin=time.time()
while np.average(timetaken[:-5:-1])<1 or i<1000:
    time_start = time.time()
    i=i+50
    dim=i
    a=np.random.random((dim,dim))

    time_end = time.time()
    time_elapsed=time_end-time_start
    timetaken.append(time_elapsed)
    max_whack.append(i^2)
    if i % 1000 == 0 or time_elapsed>0.5:
        print(i,time_elapsed)
print("total time taken: ",time_end-begin)
plt.plot(max_whack,timetaken)
plt.ylabel('Time Taken')
plt.xlabel('Cells in array')
plt.show()