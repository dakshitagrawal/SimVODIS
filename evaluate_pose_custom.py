# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F
import networks
from datasets import KITTIOdomDataset
from layers import transformation_from_parameters
from maskrcnn_benchmark.config import cfg
from options import MonodepthOptions
from utils import readlines


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# def compute_pose_loss(pred, target):
# return ((pred - target) ** 2).mean(axis=0)
def compute_pose_loss(pred_axis, target_axis, pred_trans, target_trans):
    # return ((pred_axis - target_axis) ** 2).mean() + (
    # (pred_trans - target_trans) ** 2
    # ).mean()
    return F.mse_loss(
        pred_axis, target_axis, size_average=True, reduce=True
    ) + F.mse_loss(pred_trans, target_trans, size_average=True, reduce=True)


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz**2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error**2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset"""
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(
        opt.load_weights_folder
    )

    # assert (
    # opt.eval_split == "odom_9" or opt.eval_split == "odom_10"
    # ), "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])
    opt.batch_size = 1

    filenames = readlines(
        os.path.join(
            os.path.dirname(__file__),
            "splits",
            "odom",
            "test_files_{:02d}.txt".format(sequence_id),
        )
    )

    # import pdb; pdb.set_trace()
    dataset = KITTIOdomDataset(
        opt.data_path,
        filenames,
        opt.height,
        opt.width,
        [0, 1],
        1,
        1,
        is_train=False,
        img_ext=".png",
    )
    dataloader = DataLoader(
        dataset,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    # print("Loading MaskRCNN")
    # config_file = "./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
    # cfg.merge_from_file(config_file)
    # cfg.freeze()
    # # maskrcnn_path = "./e2e_mask_rcnn_R_50_FPN_1x.pth"
    # maskrcnn_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    # pose_encoder = networks.ResnetEncoder(cfg, maskrcnn_path)
    # # pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    # # pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    print("Loading pose decoder")
    # pose_decoder = networks.PoseDecoder(len(opt.frame_ids))
    # pose_decoder = networks.PoseCNN(len(opt.frame_ids))
    # pose_decoder = networks.PoseResNet(len(opt.frame_ids))
    pose_decoder = networks.PoseViT(len(opt.frame_ids))
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    # pose_encoder.cuda()
    # pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input
    gt_local_poses = []
    mse = []
    ii = 1
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            # import pdb; pdb.set_trace()
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            # all_color_aug = torch.cat(
            #     [inputs[("color_aug", i, 0)] for i in opt.frame_ids], axis=0
            # )
            all_color_aug = torch.stack(
                [inputs[("color_aug", i, 0)] for i in opt.frame_ids], axis=1
            )

            # all_features = pose_encoder(all_color_aug)
            # all_features = [torch.split(f, opt.batch_size) for f in all_features]

            # features = {}
            # for i, k in enumerate(opt.frame_ids):
            #     features[k] = [f[i] for f in all_features]
            # pose_inputs = [features[i] for i in opt.frame_ids if i != "s"]

            # axisangle, translation = pose_decoder(pose_inputs)
            axisangle, translation = pose_decoder(all_color_aug)
            if ii == 0:
                pred_poses.append(
                    transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0],
                        True,
                    )
                    .cpu()
                    .numpy()
                )
            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                .cpu()
                .numpy()
            )
            mse.append(
                compute_pose_loss(
                    axisangle[:, 0, 0],
                    inputs[("gt_axisangle", 0, 1)],
                    translation[:, 0, 0],
                    inputs[("gt_translation", 0, 1)],
                )
                .cpu()
                .item()
            )
            gt_local_poses.append(
                transformation_from_parameters(
                    inputs[("gt_axisangle", 0, 1)][None],
                    inputs[("gt_translation", 0, 1)][None],
                )
                .cpu()
                .numpy()
            )
            # if ii % opt.log_frequency == 0:
            # print("{:04d}-th image processing".format(ii))
            ii += 1
            if ii == len(dataloader.dataset):
                break
        # pred_poses.append(
        #     transformation_from_parameters(axisangle[:, 1], translation[:, 1]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    gt_local_poses = np.concatenate(gt_local_poses)

    # gt_poses_path = os.path.join(
    #    opt.data_path, "poses", "{:02d}.txt".format(sequence_id)
    # )
    # gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    # gt_global_poses = np.concatenate(
    #    (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1
    # )
    # gt_global_poses[:, 3, 3] = 1
    # gt_xyzs = gt_global_poses[:, :3, 3]

    # gt_local_poses = []
    # for i in range(1, len(gt_global_poses)):
    #    gt_local_poses.append(
    #        np.linalg.inv(
    #            np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])
    #        )
    #    )

    # mse = compute_pose_loss(pred_poses, gt_local_poses)
    # mse = np.concatenate(mse)
    print("\n   MSE: {:0.3f}, std: {:0.3f}\n".format(np.mean(mse), np.std(mse)))

    ates = []
    num_frames = len(gt_local_poses) + 1
    track_length = 2

    # pred_x = []
    # pred_y = []
    # gt_x = []
    # gt_y = []

    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i : i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i : i + track_length - 1]))

        # pred_x.append(local_xyzs[0][0])
        # pred_y.append(local_xyzs[0][1])
        # gt_x.append(gt_local_xyzs[0][0])
        # gt_y.append(gt_local_xyzs[0][1])

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    """
    for i in range(0, num_frames - 2):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i + 1:i + track_length]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    """

    trajectory_pred = np.array(dump_xyz(pred_poses))
    trajectory_gt = np.array(dump_xyz(gt_local_poses))
    print(f"Trajectory ATE: {compute_ate(trajectory_gt, trajectory_pred)}")

    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # plt.plot(gt_x, gt_y, color="r", label="Ground Truth")
    # plt.plot(pred_x, pred_y, color="b", label="SimVODIS")
    # # plt.gca().set_aspect("equal", adjustable="datalim")
    # plt.savefig("comparision_sim_gt.png")

    print(
        "\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(
            np.mean(ates), np.std(ates)
        )
    )

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
    """
    opt = options.parse()
    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, -1, 1], 4, is_train=False, img_ext='.png')
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    # pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    config_file = "./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(config_file)
    cfg.freeze()
    maskrcnn_path = "./e2e_mask_rcnn_R_50_FPN_1x.pth"
    pose_encoder = networks.ResnetEncoder(cfg, maskrcnn_path)
    # pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    # pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(len(opt.frame_ids))
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    # opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids])

            all_features = pose_encoder(all_color_aug)
            all_features = [torch.split(f, opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(opt.frame_ids):
                features[k] = [f[i] for f in all_features]
            pose_inputs = [features[i] for i in opt.frame_ids if i != "s"]

            axisangle, translation = pose_decoder(pose_inputs)

            pred_poses.append(
                # transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
                transformation_from_parameters(axisangle[:, 1], translation[:, 1]).cpu().numpy())
            # break

    pred_poses = np.concatenate(pred_poses)

    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    """
    # for i in range(0, num_frames - 1):
    # local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
    # gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

    # ates.append(compute_ate(gt_local_xyzs, local_xyzs))
    """
    for i in range(0, num_frames - 2):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i + 1:i + track_length]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)
    """
