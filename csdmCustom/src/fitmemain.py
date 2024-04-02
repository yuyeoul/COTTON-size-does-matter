import torch
import torch.autograd as autograd

from torch.utils.data import DataLoader, Dataset

import os
import time

from csdmCustom.src.dataLoader import TryonDataset
# from dataLoader_processed import TryonDataset

# from util.utils import *
from datetime import datetime

from csdmCustom.src.Focal_Loss import focal_loss

from tqdm import tqdm
import csdmCustom.src.util.utils as utils
import numpy as np
from pytorch_fid import fid_score
from csdmCustom.src.model_end2end import COTTON, FashionOn_MultiD


class fitmeMain:
    def __init__(self,dressroom_id,train_dir,data_dir):
        dressroom_result_dir = "dressroomResult"
        current_path = os.getcwd()
        fitmeDir = current_path + "/fitme/" + dressroom_result_dir + "/" + dressroom_id
        record_file = os.path.join(train_dir,
                                   'FID_score_{}.txt'.format('val'))
        f = open(record_file, 'a')

        weight_dir = os.path.join(fitmeDir,'result/fitme', 'weights')
        weight_path = os.path.join(weight_dir, '{}.pkl'.format('fitme'))

        val_folder = os.path.join(train_dir, 'val')
        GT_folder = os.path.join(val_folder, 'GT')
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(GT_folder, exist_ok=True)

        dataset = TryonDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=1, \
                                shuffle=True, num_workers=12)

        print('Size of the dataset: %d, dataloader: %d' % (len(dataset), len(dataloader)))
        model = COTTON().cuda().train()

        best_score = np.inf
        best_epoch = 0
        # pg_unet_wo_warp 30->
        for e in range(1, 2, 1):
            weight_name = '{}_{}.pkl'.format(weight_path.split('.')[0], e)
            if not os.path.isfile(weight_name):
                print("weight not found | {}".format(weight_name))
                break
            checkpoint = torch.load(weight_name, map_location='cpu')

            fid_pred_folder = os.path.join(val_folder, '{}'.format(e)) if False else os.path.join(val_folder,
                                                                                                           '{}_untucked'.format(
                                                                                                               e))
            os.makedirs(fid_pred_folder, exist_ok=True)

            epoch_num = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
            model.cuda().eval()
            start_time = time.time()

            for step, batch in enumerate(tqdm(dataloader)):

                # Name
                human_name = batch["human_name"]
                c_name = batch["c_name"]

                # Input
                human_masked = batch['human_masked'].cuda()
                human_pose = batch['human_pose'].cuda()
                human_parse_masked = batch['human_parse_masked'].cuda()
                c_aux = batch['c_aux_warped'].cuda()
                c_torso = batch['c_torso_warped'].cuda()
                c_rgb = batch['c_rgb'].cuda()

                # Supervision
                human_img = batch['human_img'].cuda()
                human_parse_label = batch['human_parse_label'].cuda()
                human_parse_masked_label = batch['human_parse_masked_label'].cuda()

                # print("c_torso.size() = {} [{}, {}]".format(c_torso.size(), torch.min(c_torso), torch.max(c_torso)))
                # print("c_aux.size() = {} [{}, {}]".format(c_aux.size(), torch.min(c_aux), torch.max(c_aux)))
                # print("human_parse_masked.size() = {} [{}, {}]".format(human_parse_masked.size(), torch.min(human_parse_masked), torch.max(human_parse_masked)))
                # print("human_masked.size() = {} [{}, {}]".format(human_masked.size(), torch.min(human_masked), torch.max(human_masked)))
                # print("human_pose.size() = {} [{}, {}]".format(human_pose.size(), torch.min(human_pose), torch.max(human_pose)))
                # exit()

                with torch.no_grad():
                    c_img = torch.cat([c_torso, c_aux], dim=1)
                    parsing_pred, parsing_pred_hard, tryon_img_fakes = model(c_img, human_parse_masked, human_masked,
                                                                             human_pose)

                for idx, tryon_img_fake in enumerate(tryon_img_fakes):
                    utils.imsave_trainProcess([utils.remap(tryon_img_fake)], os.path.join(fid_pred_folder, c_name[idx]))
                    utils.imsave_trainProcess([utils.remap(human_img)], os.path.join(GT_folder, c_name[idx]))
                    # utils.imsave_trainProcess([utils.remap(tryon_img_fake)], os.path.join(fid_pred_folder, human_name[idx].replace('.jpg','') + '_' + c_name[idx]))
                    # utils.imsave_trainProcess([utils.remap(human_img)], os.path.join(GT_folder, human_name[idx].replace('.jpg','') + '_' + c_name[idx]))

            print("cost {}/images secs [with average of {} images]".format((time.time() - start_time) / len(dataset),
                                                                           len(dataset)))

            fid = fid_score.calculate_fid_given_paths(paths=[GT_folder, fid_pred_folder], batch_size=50,
                                                      device=torch.device(0), dims=2048, num_workers=0)

            if fid < best_score:
                best_score, best_epoch = fid, e
            print(e, fid)
            f.write('epoch:{} fid:{}\n'.format(e, fid))

        print('Best epoch:{}, Best fid:{}'.format(best_epoch, best_score))