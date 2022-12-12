import os

f = open("val_frames.txt", "r")
f2 = open("val_frames_cut.txt","w")
out=[]
for line in f:
	temp=line.split('/')[4]+'/'+line.rsplit('/', 1)[-1]
	out.append(temp)
	print(temp)

for item in out:
	f2.write(item)
