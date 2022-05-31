import numpy as np
import torch
from tqdm import tqdm


def get_predictions(model, points, index):
    if not torch.is_tensor(points):
        points = torch.from_numpy(points)
    points = points.unsqueeze(0).to(model.device)
    index = torch.tensor(index).view(1).to(model.device)
    pred = model.forward_with_preprocessing([points, index])

    try:
        pred = pred.squeeze(1)
    except:
        pass
    pred = pred.squeeze(0).detach().cpu().numpy()
    points = points.squeeze(0).cpu().numpy()

    return points, pred


def get_batched_predictions(model, points, index, batch_size=1000000, verbose=0):
    # print(f"Predicting (shape {points.shape})")
    outputs = []
    for points_ in tqdm(points.split(batch_size), disable=verbose < 1):
        outputs.append(get_predictions(model, points_, index))

    points_out = np.concatenate([o[0] for o in outputs], axis=0)
    pred_out = np.concatenate([o[1] for o in outputs], axis=0)

    return points_out, pred_out


def get_predictions_w_latents(model, points, latent):
    if not torch.is_tensor(points):
        points = torch.from_numpy(points)
    points_preprocessed = points.view(1, -1, 3).to(model.device)

    # Add Fourier features
    if model.fourier:
        points_preprocessed = model.fourier_features(points_preprocessed)

    # Add latents
    if not torch.is_tensor(latent):
        latent = torch.from_numpy(latent)
    latent = latent.view(1, -1).float().to(model.device)
    points_preprocessed = model.add_latent(points_preprocessed, latent)

    # Forward
    pred = model.forward(points_preprocessed)

    # Return numpy
    pred = pred.flatten().detach().cpu().numpy()
    points = points.view(-1, 3).cpu().numpy()
    return points, pred


def get_batched_predictions_w_latents(model, points, latent, batch_size=1000000):
    # print(f"Predicting (shape {points.shape})")
    outputs = []
    for points_ in tqdm(points.split(batch_size)):
        outputs.append(get_predictions_w_latents(model, points_, latent))

    points_out = np.concatenate([o[0] for o in outputs], axis=0)
    pred_out = np.concatenate([o[1] for o in outputs], axis=0)

    return points_out, pred_out
