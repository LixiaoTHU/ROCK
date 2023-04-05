import time, datetime, shutil, os
import numpy as np
import mlconfig
import models
import argparse
import utils
from dataset import *
from loss import FocalLoss
from thop import profile
from judgement import Judgement
from partattack import PartAttack
from trades import TradesLoss
from pgdat import PGDATLoss
from trades_awp import TradesAWP
import knowledge
import cv2
import torch
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

from ema import EMA


mlconfig.register(VOCDataset)
mlconfig.register(FaceVOCDataset)
mlconfig.register(CocoDataset)
mlconfig.register(TradesLoss)
mlconfig.register(PGDATLoss)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Training a part segmentation network.") 
    parser.add_argument('--config_path', type=str, default='configs')
    parser.add_argument('--exp_name', type=str, default="/home/lixiao/data/partdefense_nips")
    parser.add_argument('--version', type=str, default="newtrial")
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--load_model', action='store_true', default=False)
    parser.add_argument('--load_best_model', action='store_true', default=False)
    parser.add_argument('--data_parallel', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)

    parser.add_argument('--epsilon', default=1.0, type=float, help='perturbation')
    parser.add_argument('--attack_choice', default='PGD', choices=['PGD', 'AA', 'None', 'Query'])
    parser.add_argument('--attack_type', default='random', choices=["untargeted", "targeted", "background", "random", "weight", "SPSA", "NES"])
    parser.add_argument('--dump_path', default='None', type = str)
    parser.add_argument('--other_data', default=None)
    return parser.parse_args()

args = get_arguments()

# Set up
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
search_results_checkpoint_file_name = None

checkpoint_path_file = os.path.join(checkpoint_path, args.version)
utils.build_dirs(exp_path)
utils.build_dirs(checkpoint_path)
logger = utils.setup_logger(name=args.version, log_file=log_file_path + ".log")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))

scaler = GradScaler()


def adjust_learning_rate_poly(optimizer, epoch, power=0.9):
    lr = config.learning_rate * (1-epoch/config.epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch, model, optimizer, trainloader, criterion, ENV, awp_adversary, ema):
    logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
    model.train()
    # interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
    log_frequency = config.log_frequency if config.log_frequency is not None else 100
    part_loss = 0
    part_tot = 1e-8
    part_correct = 0
    for batch_idx, batch in enumerate(trainloader):
        start = time.time()

        imgs, labels, cids, names, has_part = batch
        num_has_part = has_part.sum().item()
        labels = F.interpolate(labels.unsqueeze(1), size=[28, 28])
        labels = labels[:,0,:,:]
        imgs, labels, cids = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), cids.to(device)

        optimizer.zero_grad()
        labels = labels.long()
        if "amp" in config and config.amp:
            with autocast():
                if "adv_train_type" in config and epoch >= config.warm_epochs:
                    x_adv = criterion(model, imgs, labels, cids)
                else:
                    print("error")
                    exit(-1)
                
                model.train()
                if epoch >= config.awp_warmup:
                    awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=imgs,
                                         targets=labels,
                                         beta=config.beta)
                    awp_adversary.perturb(awp)
                optimizer.zero_grad()
                x_natural = imgs
                ori_pred = model(x_natural)
                loss_natural = criterion.base_criterion(ori_pred, labels)
                pred = model(x_adv)
                N, C, H, W = pred.shape
                logits = ori_pred.permute(0,2,3,1)
                logits = logits.reshape(N*H*W, C)
                adv_logits = pred.permute(0,2,3,1)
                adv_logits = adv_logits.reshape(N*H*W, C)
                loss_robust = (1.0 / (N*H*W)) * criterion.criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                                F.softmax(logits, dim=1))
                optimizer.zero_grad()
                loss = loss_natural + config.beta * loss_robust





            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # loss.backward()
            if "grad_clip" in config:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    
            scaler.step(optimizer)
            scaler.update()

            if epoch >= config.awp_warmup:
                awp_adversary.restore(awp)

            if "ema" in config and config.ema == True:
                ema.update()
        
        part_loss += loss.item()*num_has_part*100
        _, predicted = pred.max(1)
        part_tot += 28 * 28 * cids.size(0)
        part_tot = part_tot - labels.eq(0).sum().item() - labels.eq(config.ignare_label).sum().item()

        labels[labels==0] = -1
        labels[labels==config.ignare_label] = -1
        part_correct += predicted.eq(labels).sum().item()

        end = time.time()
        time_used = end - start
        
        if ENV["global_step"] % log_frequency == 0:
            log_payload = {"part loss": part_loss / part_tot, "part acc": 100.*(part_correct/part_tot)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                           **log_payload)
            logger.info(display)

        ENV["global_step"] += 1
    return part_loss/part_tot, 100.*(part_correct/part_tot)


def test(epoch, model, testloader, criterion, judger, attacker, ENV, ema):
    logger.info("="*20 + "Test Epoch %d" % (epoch) + "="*20)
    if "ema" in config and config.ema == True:
        ema.apply_shadow()
    model.eval()
    category_total = 0 
    part_category_correct = 0
    part_category_correct_adv = 0
    part_tot = 1e-8
    part_correct = 0
    log_frequency = config.log_frequency if config.log_frequency is not None else 100
    for batch_idx, batch in enumerate(testloader):
        start = time.time()

        imgs, labels, cids, names, has_part = batch
        labels = F.interpolate(labels.unsqueeze(1), size=[28, 28])
        labels = labels[:,0,:,:]
        imgs, labels, cids = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), cids.to(device)

        with torch.no_grad():
            pred = model(imgs)
        labels = labels.long()
        labels_copy = labels.clone()

        # pixel 
        _, predicted = pred.max(1)
        part_tot += 28 * 28 * cids.size(0)
        part_tot = part_tot - labels.eq(0).sum().item() - labels.eq(config.ignare_label).sum().item()

        labels[labels==0] = -1
        labels[labels==config.ignare_label] = -1
        part_correct += predicted.eq(labels).sum().item()


        category_total += cids.size(0)
        if epoch >= config.adv_eval_epoch * config.epochs:

            score = judger.eval(pred)
            _, predicted = score.max(1)
            part_category_correct += predicted.eq(cids).sum().item()

            adv_imgs = attacker.DAG_attack(model, imgs, labels_copy, cids, attacktype = "random")
            with torch.no_grad():
                pred = model(adv_imgs)
            score = judger.eval(pred)
            _, predicted = score.max(1)
            part_category_correct_adv += predicted.eq(cids).sum().item()
        
        end = time.time()


        time_used = end - start
        if (batch_idx+1) % log_frequency == 0:
            log_payload = {"part category acc": 100.* (part_category_correct/category_total), 
                        "part adv category acc": 100.* (part_category_correct_adv/category_total)}
            display = utils.log_display(epoch=epoch,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                            **log_payload)
            logger.info(display)
    if "ema" in config and config.ema == True:
        ema.restore()
    return 100.* (part_category_correct/category_total), 100.* (part_category_correct_adv/category_total), 100.*(part_correct/part_tot)

def query_single(judger, model, x, ids, sigma = 0.001, q = 128, dist = "gauss"):
    g = torch.zeros(x.shape).to(x.device)
    with torch.no_grad():
        for t in range(q):
            if dist == "gauss":
                u = torch.randn(x.shape).to(x.device)
            elif dist == "Rademacher":
                u = torch.empty(x.shape).uniform_(0, 1).to(x.device)
                u = torch.bernoulli(u)
                u = (u - 0.5) * 2
            z1 = model(x + sigma * u)
            z1 = judger.eval(z1)
            z2 = model(x - sigma * u)
            z2 = judger.eval(z2)

            Jlist = []
            for i in range(z1.shape[0]):
                z1y = z1[i][ids[i]].clone()
                z1[i][ids[i]] = -10000
                J1 = torch.max(z1[i]) - z1y

                z2y = z2[i][ids[i]].clone()
                z2[i][ids[i]] = -10000
                J2 =  torch.max(z2[i]) - z2y
                Jlist.append(J1 - J2)
            J = torch.Tensor(Jlist).to(x.device)
            J = J.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            g += J * u
    g = g / (q * 2 * sigma)
    return g

def query(judger, model, imgs, ids, query_type):
    model.eval()
    adv_imgs = imgs.clone()
    for i in range(40):
        print("step: ", i)
        if query_type == "NES":
            est_grad = query_single(judger, model, adv_imgs, ids, dist = "gauss")
        elif query_type == "SPSA":
            est_grad = query_single(judger, model, adv_imgs, ids, dist = "Rademacher")
        adv_imgs = adv_imgs.detach() + (args.epsilon / (8*255)) * est_grad.sign()
        delta = torch.clamp(adv_imgs - imgs, min=-args.epsilon / 255, max=args.epsilon / 255)
        adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
    return adv_imgs



def evaluate_attack(model, testloader, judger, attacker, ENV):
    logger.info("="*20 + "Evaluate %s" % (args.attack_type) + "="*20)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    category_total = 0 
    part_category_correct = 0
    part_category_correct_adv = 0
    log_frequency = config.log_frequency if config.log_frequency is not None else 100
    
    for batch_idx, batch in enumerate(testloader):
        start = time.time()

        imgs, labels, cids, names, has_part = batch
        labels = F.interpolate(labels.unsqueeze(1), size=[28, 28])
        labels = labels[:,0,:,:]
        imgs, labels, cids = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), cids.to(device)
        labels = labels.long()

        ori_imgs = imgs.clone()

        with torch.no_grad():
            pred = model(imgs)

        score = judger.eval(pred)
        _, predicted = score.max(1)

        category_total += cids.size(0)

        part_category_correct += (predicted).eq(cids).sum().item()
        if args.attack_choice == "PGD" or args.attack_choice == "Query":
            if args.attack_type == "weight":
                adv_imgs = attacker.opt_attack(model, imgs, cids)
            elif args.attack_type == "NES":
                adv_imgs = query(judger, model, imgs, cids, args.attack_type)
            elif args.attack_type == "SPSA":
                adv_imgs = query(judger, model, imgs, cids, args.attack_type)
            else:
                adv_imgs = attacker.DAG_attack(model, imgs, labels, cids, attacktype = args.attack_type)
            if not os.path.exists(args.dump_path):
                os.mkdir(args.dump_path)
            if args.dump_path != "None":
                s_adv_imgs = adv_imgs * 255
                for idx in range(len(names)):
                    name = names[idx].split("/")[-1]
                    name = name.split(".")[0]
                    name = os.path.join(args.dump_path,name) + ".jpg"
                    cv2.imwrite(name, s_adv_imgs[idx,:,:,:].cpu().detach().numpy().transpose((1,2,0)))
            with torch.no_grad():
                pred = model(adv_imgs)
            score = judger.eval(pred)
            _, predicted = score.max(1)
            part_category_correct_adv += (predicted).eq(cids).sum().item()
        
        end = time.time()
        time_used = end - start
        if (batch_idx) % log_frequency == 0:
            log_payload = {"part category acc": 100.* (part_category_correct/category_total), 
                        "part adv category acc": 100.* (part_category_correct_adv/category_total)}
            display = utils.log_display(epoch=0,
                                        global_step=ENV["global_step"],
                                        time_elapse=time_used,
                                            **log_payload)
            logger.info(display)
    return 100.* (part_category_correct/category_total), 100.* (part_category_correct_adv/category_total)




def main():
    model = config.model().to(device)
    proxy = config.model().to(device)

    if "transformer" in config:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.03)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,weight_decay=5e-4)
    criterion = FocalLoss(2, 1000) 



    # define judgement
    names = ["stat_train.txt", "partlabel.txt", "categorylabel.txt"]
    contents = []
    for n in names:
        f = open(os.path.join(config.root, config.subroot, n), "r")
        s = eval(f.read())
        f.close()
        contents.append(s)
    stat, plabels_, clabels_ = contents[0], list(contents[1]), list(contents[2])

    categ2pids = defaultdict(list)
    pid2name = dict()
    for pid, plabel in enumerate(plabels_):
        categ = plabel.split('_')[0]
        categ_id = clabels_.index(categ)
        categ2pids[categ_id].append(pid+1)
        pid2name[pid+1] = plabel

    judger = Judgement(categ2pids, pid2name, clabels_, plabels_, getattr(knowledge, config.knowledge), stat, 
                        finegrained = config.finegrained, withknowledge = config.withknowledge, withat = config.withat) 




    starting_epoch = 0

    config.set_immutable()
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
    logger.info("PyTorch Version: %s" % (torch.__version__))
    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

    ENV = { 'global_step': 0,
            'best_acc': 0.0,
            'curren_acc': 0.0,
            'best_pgd_acc': 0.0,
            'train_history': [],
            'eval_history': [],
            'pgd_eval_history': []}

    if args.load_model or args.load_best_model:
        filename = checkpoint_path_file + '_best.pth' if args.load_best_model else checkpoint_path_file + '.pth'
        checkpoint = utils.load_model(filename=filename,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=None)
        starting_epoch = checkpoint['epoch'] + 1
        ENV = checkpoint['ENV']
        logger.info("File %s loaded!" % (filename))
    
    if args.data_parallel:
        print('data_parallel')
        model = torch.nn.DataParallel(model).to(device)
        proxy = torch.nn.DataParallel(proxy).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=config.learning_rate)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, 
                    gamma=config.awp_gamma, base_criterion = criterion, ctype = "seg")

    if "ema" in config and config.ema == True:
        ema = EMA(model, 0.995)
        ema.register()

    else:
        ema = None

    trainloader = data.DataLoader(config.traindataset(root = config.root), 
                            batch_size= config.batch_per_gpu * len(device_list), shuffle=True, 
                            num_workers=4, pin_memory=True)
    testloader = data.DataLoader(config.testdataset(root = config.root, other_data = args.other_data), 
                            batch_size= config.batch_per_gpu * len(device_list) // 2, shuffle=True, 
                            num_workers=4, pin_memory=True)
    logger.info("Starting Epoch: %d" % (starting_epoch))
    if args.train:
        # define attacker
        eps, alpha = config.epsilon / 255, config.epsilon / (255*8)
        if "single_step" in config:
            alpha = config.single_step / 255
        attacker = PartAttack(judger, categ2pids = categ2pids, num_class = config.num_classes, c_classes = len(clabels_), eps = eps, alpha = alpha)

        for epoch in range(starting_epoch, config.epochs):
            adjust_learning_rate_poly(optimizer, epoch)

            if "adv_train_type" in config and epoch >= config.warm_epochs:
                train_criterion = config.adv_train_type(categ2pids = categ2pids, base_criterion = criterion, attacker = attacker)
            else:
                train_criterion = criterion

            tp_loss, tp_acc = train(epoch, model, optimizer, trainloader, train_criterion, ENV, awp_adversary, ema)
            vpc_acc, vpc_acc_adv, vp_acc = test(epoch, model, testloader, criterion, judger, attacker, ENV, ema)
            if "ema" in config and config.ema == True:
                ema.apply_shadow()

            is_best = True if vpc_acc > ENV['best_acc'] else False
            ENV['train_history'].append(tp_acc)
            ENV['eval_history'].append(vpc_acc)
            ENV['best_acc'] = max(ENV['best_acc'], vpc_acc)
            ENV['curren_acc'] = vpc_acc
            if epoch >= config.adv_eval_epoch * config.epochs:
                ENV['best_pgd_acc'] = max(ENV['best_pgd_acc'], vpc_acc_adv)
                ENV['pgd_eval_history'].append(vpc_acc_adv)

            logger.info('Current pixel accuracy: %.2f' % (vp_acc))
            logger.info('Current classification accuracy: %.2f' % (ENV['curren_acc']))
            logger.info('Current PGD classification accuracy: %.2f' % (vpc_acc_adv))
            
            target_model = model.module if args.data_parallel else model
            filename = checkpoint_path_file + '.pth'
            utils.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            alpha_optimizer=None,
                            scheduler=None,
                            genotype=None,
                            save_best=is_best,
                            filename=filename)
            logger.info('Model Saved at %s\n', filename)
            if "ema" in config and config.ema == True:
                ema.restore()
    
    elif args.attack_choice == 'PGD' or args.attack_choice == 'None' or args.attack_choice == 'Query':
        # define attacker

        eps, alpha = args.epsilon / 255, args.epsilon / (255*8)
        attacker = PartAttack(judger, categ2pids = categ2pids, num_class = config.num_classes, c_classes = len(clabels_), eps = eps, alpha = alpha)
        vpc_acc, vpc_acc_adv = evaluate_attack(model, testloader, judger, attacker, ENV)
        print(vpc_acc_adv)
        logger.info('Current classification accuracy: %.2f' % (vpc_acc))
        logger.info('Current PGD classification accuracy: %.2f' % (vpc_acc_adv))

if __name__ == "__main__":
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)
