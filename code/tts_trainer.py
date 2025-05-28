import argparse
import gpytorch
import torch
import wandb
import pandas as pd
import os
from tqdm import tqdm
from models.GP import DirichletGPModel
from torch.utils.data import DataLoader
# from utils.tts_dataset import DeepFakeDetectionDataset, custom_collate
from utils.tts_personalized_dataset import DeepFakeDetectionDataset, custom_collate
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from utils.utils import str2bool, list_of_strings
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import random

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, path, epoch):
    """Save model state dictionary."""
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"xlsr4_model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def compute_ece(probs, labels, n_bins=20):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        bin_prob = probs[bin_mask]
        bin_labels = labels[bin_mask]
        if len(bin_prob) > 0:
            bin_accuracy = bin_labels.mean()
            bin_confidence = bin_prob.mean()
            ece += np.abs(bin_confidence - bin_accuracy) * len(bin_prob) / len(probs)
    return ece



def plot_calibration_curve_during_training(predicted_probs, true_labels, n_bins, id, val_model):
    # Convert lists to NumPy arrays
    predicted_probs = np.array(predicted_probs)
    true_labels = np.array(true_labels)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(true_labels, predicted_probs, n_bins=n_bins)

    # Compute bin counts with matching shape
    bin_counts, _ = np.histogram(predicted_probs, bins=n_bins, range=(0, 1))
    bin_counts = bin_counts[:len(prob_pred)]  # Ensure shape matches

    # Compute metrics
    ece = np.sum((np.abs(prob_pred - prob_true) * bin_counts) / np.sum(bin_counts))  # Weighted ECE
    mce = np.max(np.abs(prob_pred - prob_true))
    brier_score = np.mean((predicted_probs - true_labels) ** 2)

    # Plot calibration curve
    plt.figure(figsize=(5, 5))
    plt.bar(prob_pred, prob_true, width=0.8/n_bins, color='#191970', edgecolor='black', alpha=0.7)
    plt.plot([0, 1], [0, 1], linestyle='--', color='darkred', linewidth=2,)
    print('prob_pred, prob_true')
    print(prob_pred, prob_true)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()

    # Display metrics
    textstr = f"ECE: {ece:.3f}\nMCE: {mce:.3f}\nBrier: {brier_score:.3f}"
    plt.text(0.05, 0.95, textstr, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.6),
             transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='left')

    plt.grid(False)
    save_path = f"./figs/calibration_curve_{'-'.join(val_model)}.png"
    plt.savefig(save_path)
    print(f"Calibration curve saved at {save_path}")

    plt.close()

def plot_calibration_curve(predicted_probs, true_labels, n_bins, id, val_model):
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(true_labels, predicted_probs, n_bins=n_bins)

    # Plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration curve')
    # plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{id}_calibration_curve_{"-".join(args.val_tts)}.png')


def plot_roc_curve(true_labels, predicted_probs, id, val_model):
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'/figs/{id}_roc_curve_{"-".join(val_model)}.png')
    plt.show()


def calculate_eer(fpr, tpr):
    """Calculate EER by finding the threshold where FPR and FNR are equal."""
    fnr = 1 - tpr
    eer_threshold = np.argmin(np.abs(fnr - fpr))
    eer = fpr[eer_threshold]
    return eer


def evaluate(xlsr_model, gp_model, likelihood, dataloader, feature_extractor, is_likelihood = False, id=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xlsr_model.eval().to(device)
    gp_model.eval().to(device)
    likelihood.eval().to(device)
    total_correct = 0
    total_count = 0
    all_labels = []
    all_probs = []
    all_neg_probs = []
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for input_features, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = feature_extractor(input_features, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to(device)
            output_xlsr = xlsr_model(inputs).last_hidden_state.mean(dim=1).to(device)

            # Predict the posterior distribution
            with gpytorch.settings.fast_pred_var():
                latent_dist = gp_model(output_xlsr)
                if is_likelihood:
                    observed_pred = likelihood(latent_dist)
                    pred_means = observed_pred.mean  # Get the mean of the predictions
                    probs = torch.sigmoid(pred_means).squeeze()
                    preds = (probs > 0.5).long().cpu().numpy()[-1]
                else:
                    pred_samples = latent_dist.sample(torch.Size((1024,))).exp()
                    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)

                    confidence = torch.max(probabilities, dim=0)[0]
                    preds = (probabilities > 0.5).long().cpu().numpy()[-1]

            positive_probs = probabilities.cpu().numpy()[-1]
            negative_probs = probabilities.cpu().numpy()[0]

            all_neg_probs.extend(negative_probs)
            all_probs.extend(positive_probs)
            all_preds.extend(preds)
            # all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confidence = [float(c) for c in confidence]
            all_confidences.extend(confidence)
            labels = labels.numpy()
            total_correct += (preds == labels).sum()
            total_count += labels.size

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    # Generate the calibration plot
    plot_calibration_curve_during_training(all_probs, all_labels, 20, id, args.val_tts)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    eer = calculate_eer(fpr, tpr)


    # Find recall where precision is 95%
    target_precision = 0.95
    recall_at_target_precision = None
    for p, r in zip(precisions, recalls):
        if p >= target_precision:
            recall_at_target_precision = r
            break

    recall_95 = recall_at_target_precision
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = total_correct / total_count


    auc = roc_auc_score(all_labels, all_probs)
    avg_confidence = float(sum(all_confidences) / len(all_confidences))
    ece = compute_ece(np.array(all_probs), np.array(all_labels))

    # plot_calibration_curve(all_probs, all_labels, 10, id, args.val_tts)
    # plot_roc_curve(all_labels, all_probs, id, args.val_tts)
    print(auc)
    print(f'eer: {eer}')
    eers.append(eer)
    print(eers)
    log_file = "EER_10pos_10neg_no_11labs_new.txt"
    with open(log_file, "a") as f:
        f.write(f"EER: {eer}\n")

    return accuracy, avg_confidence, auc, precision, recall, ece, all_confidences, recall_95, eer



def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")

    train_tts_dataset = DeepFakeDetectionDataset(
        data_path=args.train_path,
        tts_models=args.train_tts,
        eleven_labs=args.eleven_labs,
        few_shot_samples=args.few_shot,
        id=args.id)


    gp_train_tts_dataset = DeepFakeDetectionDataset(
        data_path=args.train_path,
        tts_models=args.train_tts,
        eleven_labs=True,
        few_shot_samples=args.few_shot,
        type='eleven_labs',
        id=args.id)

    val_datasets = {}
    for tts in args.val_tts:
        val_datasets[tts] = DeepFakeDetectionDataset(
        data_path=args.val_path,
        tts_models=[tts],
        type='val',
        id=args.id)


    val_dataloaders = {}
    train_dataloader = DataLoader(train_tts_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=custom_collate)
    gp_train_dataloader = DataLoader(gp_train_tts_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=custom_collate)

    for tts in args.val_tts:
        val_dataloaders[tts] = DataLoader(val_datasets[tts], batch_size=args.val_batch_size, shuffle=True, collate_fn=custom_collate)

    local_xlsr_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-2b").to(
        device)

    global_step = 0
    if args.checkpoint_path is not None:
        local_xlsr_model.load_state_dict(torch.load(args.checkpoint_path))

    # Training Only Last block:
    c = 0
    for param in local_xlsr_model.parameters():
        param.requires_grad = False
        c += 1
        if c == 805:
            break

    xlsr_optimizer = torch.optim.Adam([
        {'params': local_xlsr_model.parameters(), 'lr': 0.000001},
    ])


    for epoch in range(args.n_epochs):
        if args.train:
            local_xlsr_model.train()
            for batch_idx, (input_features, labels) in enumerate(tqdm(train_dataloader, desc=f"Training {args.train_tts}")):
                device='cuda'
                inputs = feature_extractor(input_features, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
                output_xlsr = local_xlsr_model(inputs).last_hidden_state.mean(dim=1)
                likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(labels.to(device), learn_additional_noise=True).to(device)
                gp_model = DirichletGPModel(output_xlsr, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes).to(device)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model).to(device)

                gp_model.train()
                likelihood.train()

                optimizer = torch.optim.Adam([
                    {'params': gp_model.parameters(), 'lr': args.lr_gp}
                ])
                xlsr_optimizer.zero_grad()
                optimizer.zero_grad()
                output = gp_model(output_xlsr)
                loss = -mll(output, likelihood.transformed_targets).sum()
                loss.backward()  # retain_graph=True
                optimizer.step()
                optimizer.zero_grad()

                xlsr_optimizer.step()
                xlsr_optimizer.zero_grad()

            save_model(local_xlsr_model, "/dsi/fetaya-lab/DFD/ft_xlsr", epoch)

        if args.eval:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            local_xlsr_model.eval()

            all_output_xlsr = torch.tensor([]).to('cpu')
            all_labels = torch.tensor([]).to('cpu')
            with torch.no_grad():
                for batch_idx, (input_features, labels) in enumerate(
                        tqdm(gp_train_dataloader, desc=f"Evaluating XLSR for client {id}")):
                    inputs = feature_extractor(input_features, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to(
                        device)
                    output_xlsr = local_xlsr_model(inputs).last_hidden_state.mean(dim=1)
                    all_output_xlsr = torch.cat((all_output_xlsr, output_xlsr.cpu()), 0)
                    all_labels = torch.cat((all_labels, labels), 0).to(torch.int)

            device = 'cpu'
            likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(all_labels.to(device),
                                                                                learn_additional_noise=False).to(device)
            gp_model = DirichletGPModel(all_output_xlsr, likelihood.transformed_targets, likelihood,
                                        num_classes=likelihood.num_classes).to(device)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model).to(device)

            gp_model.train()
            likelihood.train()
            gp_optimizer = torch.optim.Adam([
                {'params': gp_model.parameters(), 'lr': args.lr_gp}
            ])
            gp_optimizer.zero_grad()
            gp_iterations = 300
            for i in range(gp_iterations):
                output = gp_model(all_output_xlsr)
                loss = -mll(output, likelihood.transformed_targets).sum()
                loss.backward()
                gp_optimizer.step()
                gp_optimizer.zero_grad()
                if i%100==0:
                    print('loss', loss.item())

            print('args.val_tts', args.val_tts)
            for tts in args.val_tts:
                print('*****', tts, 'evaluation:', '*****')
                val_accuracy, avg_confidence, auc, precision, recall, ece, all_confidence, recall_95, eer = evaluate(local_xlsr_model, gp_model, likelihood, val_dataloaders[tts], feature_extractor, False, args.id)
                print('eer', eer)
                print('ece', ece)
                print('---------')

            torch.cuda.empty_cache()


        if args.wandb:
            wandb.log({"train/epoch": epoch}, step=global_step)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="DeepFake Detection in Audio speech")

    parser.add_argument("--train-path",
                        type=str,
                        default="data",
                        help="directory path for dataset")

    parser.add_argument("--val-path",
                        type=str,
                        default="val",
                        help="directory path for dataset")

    parser.add_argument("--test-path",
                        type=str,
                        default="test",
                        help="directory path for dataset")

    parser.add_argument("--checkpoint-path",
                        type=str,
                        default=None,
                        help="directory path for dataset")

    parser.add_argument("--n-epochs",
                        type=int,
                        default=1,
                        help="number of training epochs")

    parser.add_argument("--lr-gp",
                        type=float,
                        default=0.05,
                        help="learning rate for GP model")

    parser.add_argument("--lr-xlsr",
                        type=float,
                        default=0.001,
                        help="learning rate for XLS-R model")

    parser.add_argument("--train-batch-size",
                        type=int,
                        default=64,
                        help="training batch size")

    parser.add_argument("--val-batch-size",
                        type=int,
                        default=8,
                        help="validation batch size")

    parser.add_argument("--test-batch-size",
                        type=int,
                        default=32,
                        help="test batch size")

    parser.add_argument("--few-shot",
                        type=int,
                        default=32,
                        help="test batch size")


    parser.add_argument("--wandb",
                        type=str2bool,
                        default=False,
                        help="use wandb for logging")

    parser.add_argument("--eval",
                        type=str2bool,
                        default=False,
                        help="use wandb for logging")
    parser.add_argument("--train",
                        type=str2bool,
                        default=False,
                        help="use wandb for logging")

    parser.add_argument("--eleven-labs",
                        type=str2bool,
                        default=False,
                        help="use wandb for logging")

    parser.add_argument('--tts_models',
                        type=list_of_strings,
                        help="pass tts model seperated with comma")

    parser.add_argument('--train-tts',
                        type=list_of_strings,
                        help="what will be the tts_model the gp will be trained on")

    parser.add_argument('--val-tts',
                        type=list_of_strings,
                        help="what will be the tts model the gp will be validated on")

    parser.add_argument('--id',
                        type=str)

    args = parser.parse_args()

    if args.wandb:
        wandb.init(project="DeepFakeDetection", entity="your_entity")

    ids = list(set(pd.read_csv('data/speakers_data_voxceleb.csv')['person_id']))
    ids.sort()
    eers = []

    for i in ids:
        print('ids', i)
        args.id = i
        train(args)
