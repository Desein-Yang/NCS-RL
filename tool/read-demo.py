import pickle
import sys

with open('./demos/'+sys.argv[1]+'NoFrameskip-v4-1000.pickle','rb+') as f:
    buf = pickle.load(f)

print("Buffer length %d" % buf.length)
action_list = [buf.act(i) for i in range(buf.length)]
#print(action_list)
with open('./demos/'+sys.argv[1] + 'action.pickle','wb+') as f:
    pickle.dump(action_list,f)
