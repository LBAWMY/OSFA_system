import torch
from torch import nn
import numpy as np
import time
import random
from sklearn import metrics
import mstcn
from transformer2_3_1 import Transformer2_3_1
import os
import cv2
import argparse
from torchvision import models, transforms
from PIL import Image

def read_labels_from_file(label_path):
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line_no == 1:
                continue
            frame_str, phase_str = line.split()
            labels.append(int(phase_str))
    if not labels:
        raise ValueError(f"No labels were read from {label_path}")
    return labels


def load_video_frames(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Can not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    framenum = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video fps: {fps:.3f}")
    print(f"video total frames (meta): {framenum}")

    video_frame = np.zeros((framenum, 250, 250, 3), dtype='uint8')
    cnt = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.resize(frame, (250, 250))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame[cnt] = frame
        cnt += 1
    capture.release()
    if cnt == 0:
        raise RuntimeError(f"No frame decoded from video: {video_path}")
    if cnt != framenum:
        video_frame = video_frame[:cnt]
    return video_frame, cnt, fps


def build_sampled_indices(decoded_num_frames, fps, predict_rate):
    if predict_rate == "1fps":
        step = max(1, int(round(fps)))
        return list(range(0, decoded_num_frames, step))
    return list(range(decoded_num_frames))


def build_labels_for_inference(raw_labels, decoded_num_frames, sampled_indices, predict_rate):
    if predict_rate == "original_fps":
        if len(raw_labels) != decoded_num_frames:
            raise ValueError(
                f"For original_fps mode, label count ({len(raw_labels)}) must match decoded frames ({decoded_num_frames})."
            )
        sampled_labels = [raw_labels[idx] for idx in sampled_indices]
    else:
        # 1fps mode supports either per-frame labels or already-downsampled labels.
        if len(raw_labels) == decoded_num_frames:
            sampled_labels = [raw_labels[idx] for idx in sampled_indices]
        elif len(raw_labels) == len(sampled_indices):
            sampled_labels = raw_labels
        else:
            raise ValueError(
                f"For 1fps mode, label count should be either decoded frames ({decoded_num_frames}) "
                f"or sampled 1fps frames ({len(sampled_indices)}), but got {len(raw_labels)}."
            )
    # Dataset labels are 1-based; convert to 0-based to match model predictions.
    sampled_labels = [int(x) - 1 for x in sampled_labels]
    return np.asarray(sampled_labels, dtype=np.int64).reshape(-1, 1)


class LFBResNetEmbedding(torch.nn.Module):
    def __init__(self):
        super(LFBResNetEmbedding, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        return x


def extract_lfb_features(video_frames, embedding_model_path, sequence_length_lfb=1, batch_size=100):
    fe_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_lfb = LFBResNetEmbedding()
    model_lfb.load_state_dict(torch.load(embedding_model_path), strict=False)
    model_lfb.to(fe_device)
    model_lfb.eval()

    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837]),
    ])

    frame_tensors = [transform(Image.fromarray(fr)) for fr in video_frames]
    if len(frame_tensors) < sequence_length_lfb:
        raise ValueError(
            f"video frame count ({len(frame_tensors)}) is smaller than LFB sequence_length ({sequence_length_lfb})."
        )

    # Same indexing idea as generate_LFB.get_useful_start_idx_LFB for one video.
    useful_start_idx = list(range(0, len(frame_tensors) + 1 - sequence_length_lfb))
    sample_idx = []
    for sidx in useful_start_idx:
        for k in range(sequence_length_lfb):
            sample_idx.append(sidx + k)

    g_lfb_test = np.zeros(shape=(0, 2048))
    with torch.no_grad():
        for s in range(0, len(sample_idx), batch_size):
            chunk_idx = sample_idx[s:s + batch_size]
            chunk = torch.stack([frame_tensors[idx] for idx in chunk_idx], dim=0).to(fe_device)
            chunk = chunk.view(-1, sequence_length_lfb, 3, 224, 224)
            outputs_feature = model_lfb.forward(chunk).data.cpu().numpy()
            g_lfb_test = np.concatenate((g_lfb_test, outputs_feature), axis=0)
    return g_lfb_test


def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature


class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q

        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                        d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q = sequence_length)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)

    def forward(self, x, long_feature):
        out_features = x.transpose(1,2)
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
            else:
                input = out_features[:, i-self.len_q+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0,1))
        output = self.transformer(inputs, feas)
        return output

test_label_save_path = './data/'
TeCNO_best_model_path = "./model/TeCNO.pth"
TeCNO_Trans_best_model_path = "./model/TeCNO_Trans.pth"
embedding_model_path_default = "./model/embedding.pth"
video_path = "./data/020.mp4"
label_path = "./data/020.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default=video_path, help="Path to local video file")
parser.add_argument("--label_path", type=str, default=label_path, help="Path to frame-level label txt/csv file")
parser.add_argument("--save_dir", type=str, default=test_label_save_path, help="Output directory")
parser.add_argument("--embedding_model_path", type=str, default=embedding_model_path_default,
                    help="Checkpoint used in generate_LFB-style embedding extraction")
parser.add_argument("--predict_rate", type=str, default="original_fps", choices=["original_fps", "1fps"],
                    help="Predict at original frame rate or 1fps")
args = parser.parse_args()

test_label_save_path = args.save_dir
os.makedirs(test_label_save_path, exist_ok=True)

video_frames, decoded_num_frames, video_fps = load_video_frames(args.video_path)
sampled_indices = build_sampled_indices(decoded_num_frames, video_fps, args.predict_rate)
sampled_video_frames = video_frames[sampled_indices]

g_LFB_test = extract_lfb_features(
    sampled_video_frames,
    embedding_model_path=args.embedding_model_path,
    sequence_length_lfb=1,
    batch_size=100
)

raw_test_labels = read_labels_from_file(args.label_path)
test_labels_80 = build_labels_for_inference(
    raw_test_labels, decoded_num_frames, sampled_indices, args.predict_rate
)
test_num_each_80 = [len(test_labels_80)]
test_start_vidx = [0]

out_features = 7
batch_size = 1
mstcn_causal_conv = True
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim = 2048
mstcn_stages = 2

sequence_length = 30

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model.load_state_dict(torch.load(TeCNO_best_model_path))
model.cuda()
model.eval()

model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
model1.load_state_dict(torch.load(TeCNO_Trans_best_model_path))
model1.cuda()

test_we_use_start_idx_80 = [x for x in range(len(test_num_each_80))]

for epoch in range(1):
    torch.cuda.empty_cache()
    model1.train()

    # Sets the module in evaluation mode.
    model.eval()
    model1.eval()

    test_progress = 0
    test_corrects_phase = 0
    test_all_preds_phase = []
    test_all_labels_phase = []
    test_acc_each_video = []
    test_start_time = time.time()

    with torch.no_grad():
        for i in test_we_use_start_idx_80:
            labels_phase = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                labels_phase.append(test_labels_80[j][0])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = get_long_feature(start_index=test_start_vidx[i],
                                            lfb=g_LFB_test, LFB_length=test_num_each_80[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            out_features = model.forward(video_fe)[-1]
            out_features = out_features.squeeze(1)
            p_classes1 = model1(out_features, long_feature)

            p_classes = p_classes1.squeeze()

            _, preds_phase = torch.max(p_classes.data, 1)

            test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / test_num_each_80[i])

            for j in range(len(preds_phase)):
                test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

            test_progress += 1

            label_results_all = []
            for k in range(test_num_each_80[i]):
                frame_id = int(sampled_indices[k] + 1)
                gt_v = int(labels_phase.data.cpu()[k])
                pred_v = int(preds_phase.data.cpu()[k])
                label_each = [frame_id, gt_v, pred_v]
                label_results_all.append(label_each)

            video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
            out_csv_path = os.path.join(test_label_save_path, video_stem + "_Phase_Label_Pred.csv")
            np.savetxt(out_csv_path, label_results_all, delimiter=",", fmt="%d")
            print("saved:", out_csv_path)

    test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
    test_acc_video = np.mean(test_acc_each_video)
    test_elapsed_time = time.time() - test_start_time

    test_recall_phase = metrics.recall_score(test_all_labels_phase, test_all_preds_phase, average='macro')
    test_precision_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average='macro')
    test_jaccard_phase = metrics.jaccard_score(test_all_labels_phase, test_all_preds_phase, average='macro')
    test_precision_each_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average=None)
    test_recall_each_phase = metrics.recall_score(test_all_labels_phase, test_all_preds_phase, average=None)
    print("test_precision_phase", test_precision_phase)
    print("test_recall_phase", test_recall_phase)
    print("test_jaccard_phase", test_jaccard_phase)

    print('test in: {:2.0f}m{:2.0f}s'
          ' test accu(phase): {:.4f}'
          ' test accu(video): {:.4f}'
          .format(test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_accuracy_phase,
                  test_acc_video))


