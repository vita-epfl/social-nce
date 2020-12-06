import torch
import pdb

def convert(data):
    len_data = len(data)
    robot_data = torch.empty(len_data, data[0][0][0].size(0))
    human_data = torch.empty(len_data, data[0][0][1].size(0), data[0][0][1].size(1))
    action_data = torch.empty(len_data, data[0][1].size(0))
    value_data = torch.empty(len_data, data[0][2].size(0))
    for i, (obs, act, val) in enumerate(data):
        robot_data[i] = obs[0]
        human_data[i] = obs[1]
        action_data[i] = act
        value_data[i] = val
    return [robot_data, human_data, action_data, value_data]

def combine(flist):
    data_all = []
    for fname in flist:
        print(fname)
        data_sub = torch.load(fname)
        print("# frame:", len(data_sub))
        data_all.extend(data_sub)
    return data_all

if __name__ == "__main__":
    foldername = 'data/demonstration/'
    flist = [foldername+'data_imit_invisible.pt']
    data_raw = combine(flist)
    data_convert = convert(data_raw)
    torch.save(data_convert, foldername+'data_imit.pt')
