import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pdb
from options import MonodepthOptions
from utils import readlines
from datasets import KITTIOdomDataset
from torch.utils.data import DataLoader
import networks
import torch
from tqdm import tqdm

predicted_result_dir = "./result"


def plot_route(gt, out, c_gt="g", c_out="r"):
    x_idx = 0
    y_idx = 2
    x = [v for v in gt[:, x_idx]]
    y = [v for v in gt[:, y_idx]]
    plt.plot(x, y, color=c_gt, label="Ground Truth")
    # plt.scatter(x, y, color='b')

    x = [v for v in out[:, x_idx]]
    y = [v for v in out[:, y_idx]]
    plt.plot(x, y, color=c_out, label="Predicted")
    # plt.scatter(x, y, color='b')
    # plt.gca().set_aspect('equal', adjustable='datalim')


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()

    """Evaluate odometry on the KITTI dataset"""
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(
        opt.load_weights_folder
    )

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

    print("Loading pose decoder")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")
    # pose_decoder = networks.PoseDecoder(len(opt.frame_ids))
    # pose_decoder = networks.PoseCNN(len(opt.frame_ids))
    pose_decoder = networks.PoseResNet(len(opt.frame_ids))
    # pose_decoder = networks.PoseViT(len(opt.frame_ids))
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_decoder.cuda()
    pose_decoder.eval()

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input
    pred_trans = [np.array([0.0, 0.0, 0.0])]
    gt_trans = [np.array([0.0, 0.0, 0.0])]
    ii = 1
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            # import pdb; pdb.set_trace()
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.stack(
                [inputs[("color_aug", i, 0)] for i in opt.frame_ids], axis=1
            )

            axisangle, translation = pose_decoder(all_color_aug)
            pred_trans.append(translation.squeeze().cpu().numpy())
            gt_trans.append(inputs[("gt_translation", 0, 1)].squeeze().cpu().numpy())

            ii += 1
            if ii == len(dataloader.dataset):
                break

    pred_poses = np.stack(pred_trans)
    gt_local_poses = np.stack(gt_trans)

    gt = np.cumsum(gt_local_poses, 0)
    out = np.cumsum(pred_poses, 0)

    step = 10
    plt.clf()
    plt.scatter([gt[0][0]], [gt[0][2]], label="sequence start", marker="s", color="k")
    # for st in range(0, len(out), step):
    # end = st + step
    # g = max(0.2, st / len(out))
    # c_gt = (0, g, 0)
    # c_out = (1, g, 0)
    # plot_route(gt[st:end], out[st:end], c_gt, c_out)
    # if st == 0:
    #     plt.legend()
    # plt.title("Video {}".format(sequence_id))
    # save_name = "{}/route_{}_gradient.png".format(predicted_result_dir, sequence_id)

    plot_route(gt, out, "red", "blue")
    plt.legend()
    plt.title("Video {}".format(sequence_id))
    save_name = "{}/route_{}.png".format(predicted_result_dir, sequence_id)
    plt.savefig(save_name)
