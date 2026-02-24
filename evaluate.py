from retrieval.engine_eval import *
from dinov2.models.vision_transformer import *
from retrieval.retrieval_tools import *
import random


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.stack([torch.tensor(target[1], dtype=torch.int64) for target in batch]).squeeze()
    index = torch.tensor([index[2] for index in batch], dtype=torch.int64)  # qss
    w = imgs[0].shape[1]
    h = imgs[0].shape[2]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.float)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=float)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets, index

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config():
    config={
        'random_seed': 0, 
        'g_seed': 0, 
        'level': 0,
        'num_classes':1000,
        'batch_size': 64,
        'num_workers': 0,
        'pin_mem': False,
        # saved "teacher_checkpoint.pth" in the output_dir
        'pretrain_path': "", # TODO set this before evaluation
        'hash_bit': 256,
        'do_hash': 0,
        'use_bn':False,
        'lmdb': False,
        }
    return config

def get_data_config(config):
    if config['level'] == 0:
        config['data_path'] = '' # path to imagenet_val
        config['test_list_path'] = './imagenet_cog/cog_val_level0.txt'
        # you can use lmdb format after following "./dir2lmdb.py" 
        # NOT using lmdb by default
        if config['lmdb']:
            config['data_path_lmdb'] = '' # path to imagenet_val-1k.lmdb
    if config['level'] != 0:
        config['data_path'] = '' # path to imagenet-21k
        config['test_list_path'] = './imagenet_cog/cog_val_level'+str(config['level'])+'.txt'
        if config['lmdb']:
            config['data_path_lmdb'] = 'imagenet_cog_val_level' + str(config['level']) + '-1k.lmdb'
    return config

def get_dataloader(config):
    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    pin_mem = config['pin_mem']
    if config['lmdb']:
        dataset_test = ImageFolderLMDB(
             config['data_path_lmdb'],transform=transform_test,num_class=num_classes
         )
    else:
        dataset_test =ImageList(config['data_path'], open(config['test_list_path']).readlines(),
                              transform=transform_test, num_class=num_classes)
    g = torch.Generator()
    g.manual_seed(config['g_seed'])
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        generator=g,
        collate_fn=fast_collate
    )
    return data_loader_test

def test_mAP(config):
    set_random_seed(config['random_seed'])
    pretrain_path = config['pretrain_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_base(block_chunks=0, patch_size=16, img_size=224, init_values=1)
    checkpoint = torch.load(pretrain_path)
    teacher_state_dict = checkpoint['teacher']
    new_state_dict = {}
    for key, value in teacher_state_dict.items():
        new_key = key
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
        if new_key in model.state_dict():
            new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict,strict=False)
    model.to(device)
    if config['lmdb']:
        print("using lmdb")
    print("Testing of level%1d set..." % (config['level']))
    query_num = 50000
    topK = -1
    model.eval()
    data_loader_test = get_dataloader(config)
    mAP = evaluate(data_loader_test, model, device, query_num=query_num, topK=topK)
    print('mAP of level%1d: %.6f' % (config['level'], mAP))
    return mAP

if __name__ == '__main__':
    config = get_config()
    mAP_list = []
    for level in range(0, 6):
        config['level'] = level
        config = get_data_config(config)
        mAP = test_mAP(config)
        mAP_list.append(mAP)
    formatted_list = '\t'.join(map(str, mAP_list))
    print("mAP of ImageNet-CoG:")
    print(formatted_list)
