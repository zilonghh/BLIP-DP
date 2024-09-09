import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder as CaptionModel
from models.blip_vqa import blip_vqa as VQAModel
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval

def get_answer(image, BLIP_VQA, device, question):
    with torch.no_grad():
        answer = BLIP_VQA(image, question, train=False, inference='generate')
    return answer[0]



def train(model_caption, data_loader, optimizer, epoch, device):
    # 训练
    model_caption.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)

        loss = model_caption(image, caption)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 收集所有进程的统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model_caption, model_vqa,model_vqa_leaf, data_loader, device, config):
    # 评估
    model_caption.eval()
    model_vqa.eval()
    model_vqa_leaf.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 1

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):

        image = image.to(device)
        question_vqa = 'What is the category of this image?'  # 适用于你的VQA模型的问题
        answer_vqa = get_answer(image, model_vqa, device, question_vqa)
        
        question_leaf = 'What type of leaf is this?'
        answer_leaf = get_answer(image, model_vqa_leaf, device, question_leaf)

        # 使用VQA答案作为文本生成的 prompt
        current_prompt = f'a {answer_leaf} affected by {answer_vqa} of '
        
        model_caption.prompt = current_prompt
        captions = model_caption.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # 设置种子以确保结果可重复
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # 创建数据集
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks,
                                  global_rank)
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size']] * 3, num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    # 创建模型
    print("Creating model")
    model_caption = CaptionModel(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                                 prompt=config['prompt'])  
    model_caption = model_caption.to(device)
    print("Creating VQA model")
    image_size_vqa = 384
    model_path_vqa = 'output/VQA1/checkpoint_09.pth'
    model_vqa = VQAModel(pretrained=model_path_vqa, image_size=image_size_vqa, vit='base')
    model_vqa.eval()
    model_vqa = model_vqa.to(device)

    print("Creating VQA model for leaf classification")
    model_vqa_leaf = VQAModel(pretrained='output/VQA/checkpoint_09.pth', image_size=384, vit='base')
    model_vqa_leaf.eval()
    model_vqa_leaf = model_vqa_leaf.to(device)


    optimizer = torch.optim.AdamW(params=model_caption.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model_caption, train_loader, optimizer, epoch, device)

        val_result = evaluate(model_caption, model_vqa,model_vqa_leaf, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d' % epoch, remove_duplicate='image_id')

        test_result = evaluate(model_caption, model_vqa,model_vqa_leaf, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d' % epoch,
                                       remove_duplicate='image_id')

        if utils.is_main_process():
            coco_val = coco_caption_eval(config['coco_gt_root'], val_result_file, 'val')
            coco_test = coco_caption_eval(config['coco_gt_root'], test_result_file, 'test')

            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             }
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    'model': model_caption.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/caption_coco.yaml')
    parser.add_argument('--output_dir', default='output/Caption_leaf_des')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
