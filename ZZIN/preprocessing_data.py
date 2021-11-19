import os.path as osp
import json
import os
from tqdm import tqdm
import copy

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

add_data_dir = os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean')

with open(osp.join(add_data_dir, 'ufo/{}.json'.format('annotation')), 'r') as f:
    anno = json.load(f)
ille = False
anno = anno['images']

anno_temp = copy.deepcopy(anno)

count = 0
count_normal = 0
count_none_anno = 0

for img_name, img_info in tqdm(anno.items()) :
    if img_info['words'] == {} :
        del(anno_temp[img_name])
        count_none_anno += 1
        continue
    for obj, obj_info in img_info['words'].items() :
        if len(img_info['words'][obj]['points']) == 4 :
            count_normal += 1
            continue  
        
        elif len(img_info['words'][obj]['points'])%2 == 1 :
            del anno_temp[img_name]['words'][obj]
            
            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
            count += 1

        # 다음 예제는 polygon이 넘치거나 모자를 경우 해당 폴리곤을 object를 삭제처리
        elif len(img_info['words'][obj]['points']) < 4 :
            over_polygon_temp = copy.deepcopy(anno_temp[img_name]['words'][obj])
            over_poly_region = copy.deepcopy(over_polygon_temp)
            over_poly_region['points'] = []

            for index in range(len(img_info['words'][obj]['points'])//2 -1):
                over_poly_region['points'].append(over_polygon_temp['points'][index])
                over_poly_region['points'].append(over_polygon_temp['points'][index+1])
                over_poly_region['points'].append(over_polygon_temp['points'][-index-1])
                over_poly_region['points'].append(over_polygon_temp['points'][-index])
                anno_temp[img_name]['words'][obj+f'{index+911}'] = copy.deepcopy(over_poly_region)
                over_poly_region['points'] = []
            del anno_temp[img_name]['words'][obj]

            if anno_temp[img_name]['words'] == {} :
                del(anno_temp[img_name])
                count_none_anno += 1
                continue
            count += 1


            
print(f'normal polygon count : {count_normal}')
print(f'deleted {count} over polygon')

anno = {'images': anno_temp}

ufo_dir = osp.join('../input/data/ICDAR17_Korean', 'ufo')
maybe_mkdir(ufo_dir)
with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
    json.dump(anno, f, indent=4)