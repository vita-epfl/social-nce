def freeze_model(model, keys):
    for name, p in model.named_parameters():
        if any(key in name for key in keys):
            print("frozen parameter: ", name)
            p.requires_grad = False

def count_parameters(model, key='lstm'):
    cnt_total_params, cnt_trainable_params, cnt_key_params = 0, 0, 0
    for name, p in model.named_parameters():
        num_param = p.numel()
        cnt_total_params += num_param
        if p.requires_grad: cnt_trainable_params += num_param
        if key in name: cnt_key_params += num_param
    return cnt_total_params, cnt_trainable_params, cnt_key_params

def trim_model_dict(pretrain, name='encoder'):
    for key in list(pretrain):
        if not name in key: del pretrain[key]
    trim_state = dict((key.replace(name+'.',''), value) for (key, value) in pretrain.items())
    return trim_state