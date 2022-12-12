import json
  
# Opening JSON file
f = open('/home/shounak_rtml/11777/visual-comet/data/visualcomet/val_annots.json','r')
f2=open('/home/shounak_rtml/11777/visual-comet/scripts/error_file.txt','r')  
# returns JSON object as 
# a dictionary
data = json.load(f)
# Iterating through the json
# list
error_file=f2.readline()
print(error_file)
out=[]
flag=0
for i in range(len(data)):
  
  if(flag==1):
    flag=0
    continue
  out.append(data[i])
  #print(i)
  if(str(data[i]['img_fn'])==error_file):
    flag=1
    print("err")

  #print(out)
#     print(i)

#f3=open('/home/shounak_rtml/11777/visual-comet/data/visualcomet/val_annots2.json','w')
json.dump(out, open('/home/shounak_rtml/11777/visual-comet/data/visualcomet/val_annots.json','w'))
# for i in data['img_fn']:
#     print(i)

# for i in data['generations']:
#     print(i)

# Closing file
f.close()
