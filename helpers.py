import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import torch.optim as optim
import time

class Helpers:
  @staticmethod
  def visualize_predictions(model, loader, device, batch_size):
      model.eval()
      with torch.no_grad():
          images, labels = next(iter(loader))
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          preds = torch.sigmoid(outputs) > 0.5

          for i in range(min(batch_size, images.size(0))):
              img = images[i].cpu().permute(1, 2, 0).numpy()
              h, w, _ = img.shape
              b_channel = np.zeros((h, w, 1))
              img = np.concatenate((img, b_channel), axis=2)
              lbl = labels[i].cpu().squeeze().numpy()
              pred = preds[i].cpu().squeeze().numpy()
              tp = ((pred == 1) & (lbl == 1)).sum()
              fp = ((pred == 1) & (lbl == 0)).sum()
              fn = ((pred == 0) & (lbl == 1)).sum()
              dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

              fig, axes = plt.subplots(1, 3, figsize=(12, 4))
              axes[0].imshow(img)
              axes[0].set_title('Input')
              axes[0].axis('off')

              axes[1].imshow(lbl, cmap='gray')
              axes[1].set_title('Ground Truth')
              axes[1].axis('off')

              axes[2].imshow(pred, cmap='gray')
              axes[2].set_title(f'Prediction\nDice: {dice:.4f}')
              axes[2].axis('off')
              plt.tight_layout()
              plt.show()

  @staticmethod
  def imshowpair(model, loader, device, batch_size):
      model.eval()
      with torch.no_grad():
          images, labels = next(iter(loader))
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          preds = torch.sigmoid(outputs) > 0.5

          for i in range(min(batch_size, images.size(0))):
              image = images[i].cpu().permute(1, 2, 0).numpy()
              h, w, _ = image.shape
              b_channel = np.zeros((h, w, 1))
              image = np.concatenate((image, b_channel), axis=2)
              label = labels[i].cpu().squeeze().numpy()
              pred = preds[i].cpu().squeeze().numpy()

              overlay = np.zeros_like(image)
              overlay[...,1] = label
              overlay[...,0] = pred

              fig, axes = plt.subplots(1, 3)
              axes[0].imshow(label, cmap="gray")
              axes[0].set_title("ground truth")
              axes[0].axis('off')

              axes[1].imshow(pred, cmap="gray")
              axes[1].set_title("yhat")
              axes[1].axis('off')

              axes[2].imshow(overlay)
              axes[2].set_title("GT=G, Prediction=R")
              axes[2].axis('off')

              plt.tight_layout()
              plt.show()

  @staticmethod
  def training_curve_plot(title, train_costs, test_costs, train_dice, test_dice, batch_size, learning_rate, epochs, training_time):
    lg=18
    md=13
    sm=9
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, y=1.15, fontsize=lg)
    sub = f'| Batch size:{batch_size} | Learning rate:{learning_rate} | Number of Epochs:{epochs} | Training time:{training_time:.0f}s |'
    fig.text(0.5, 0.99, sub, ha='center', fontsize=md)
    x = range(1, len(train_costs)+1)
    axs[0].plot(x, train_costs, label=f'Final train cost: {train_costs[-1]:.4f}')
    axs[0].plot(x, test_costs, label=f'Final test cost: {test_costs[-1]:.4f}')
    axs[0].set_title('Costs', fontsize=md)
    axs[0].set_xlabel('Epochs', fontsize=md)
    axs[0].set_ylabel('Cost', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)
    axs[1].plot(x, [acc / 100 for acc in train_dice], label=f'Final train dice: {train_dice[-1]:.2f}')
    axs[1].plot(x, [acc / 100 for acc in test_dice], label=f'Final test dice: {test_dice[-1]:.2f}')
    axs[1].set_title('Dice Score', fontsize=md)
    axs[1].set_xlabel('Epochs', fontsize=md)
    axs[1].set_ylabel('Dice Score', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)
  
  @staticmethod
  def dice_components(outputs, labels, threshold):
    preds = (torch.sigmoid(outputs) > threshold).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    return tp, fp, fn

  @staticmethod
  def dice_score(tp, fp, fn):
    return (2. * tp) / (2. * tp + fp + fn + 1e-6)
    
  
  @staticmethod
  def training_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, EPOCHS):
    train_costs = []
    val_costs = []
    train_dices = []
    val_dices = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        # training
        model.train()
        train_loss = 0.0
        train_tp, train_fp, train_fn = 0.0, 0.0, 0.0  # tp, fp, fn
        val_loss = 0.0
        val_tp, val_fp, val_fn = 0.0, 0.0, 0.0  # tp, fp, fn

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #forward
            outputs = model(images)

            # loss
            loss = criterion(outputs, labels)
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            tp, fp, fn = Helpers.dice_components(outputs, labels, 0.5)
            train_tp += tp
            train_fp += fp
            train_fn += fn

        train_dice = Helpers.dice_score(train_tp, train_fp, train_fn)
        avg_train_loss = train_loss / len(train_loader)
        train_costs.append(avg_train_loss)
        train_dices.append(train_dice)

        print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Dice: {train_dice:.4f}')

        # validation
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                tp, fp, fn = Helpers.dice_components(outputs, labels, 0.5)
                val_tp += tp
                val_fp += fp
                val_fn += fn

        val_dice = Helpers.dice_score(val_tp, val_fp, val_fn)
        avg_val_loss = val_loss / len(val_loader)
        val_costs.append(avg_val_loss)
        val_dices.append(val_dice)

        if scheduler is not None:
            scheduler.step(avg_val_loss)
        print(f'Epoch {epoch+1}/{EPOCHS} - Validation Loss: {avg_val_loss:.4f}, Validation Dice: {val_dice:.4f}')

    training_time = (time.time() - start_time)
    print(f'Training time for {EPOCHS} epochs: {training_time} seconds')
    return train_costs, val_costs, train_dices, val_dices, training_time
