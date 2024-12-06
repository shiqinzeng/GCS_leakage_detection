import numpy as np
import h5py
import math
import torch
import torchvision.transforms as transforms
from dataset import form_dataset
from model import create_model
from train import train_model
from test import test_model
from torch import nn
import argparse
import random

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and Test a CNN model on JRM.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument('--data_length', type=int, default=1971, help="Length of the dataset. Default: 1971.")
    parser.add_argument('--model_name', type=str, default='resnet50', 
                        help="Name of the model to use (e.g., resnet50,vgg16). Default: resnet50.")
    args = parser.parse_args()

    # Get dataset path, data length, and model name from arguments
    dataset_path = args.dataset_path
    data_length = args.data_length
    model_name = args.model_name

    # Load data and create dataloaders
    trainloader, validationloader, testloader = form_dataset(dataset_path, data_length)

    # Create the CNN model
    model = create_model(model_name=model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Randomly assign weights [a, b] such that b > a and a + b = 1
    a = random.uniform(0, 0.5)  # Randomly choose a in [0, 0.5]
    b = 1 - a                   
    weight_loss = [a, b]

    print(f"Randomly assigned weights: a = {a:.4f}, b = {b:.4f}")

    # Training process
    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, 
        trainloader, 
        validationloader, 
        device, 
        num_epochs=30, 
        lr=0.00005, 
        step_size=1, 
        gamma_lr=0.9999, 
        weight_loss=weight_loss, 
        recall_loss_ratio=0.7
    )

    # Test the model
    testing_loss, testing_acc, label_list, pred_list, output_list = test_model(model, testloader)
    print('Testing loss:', testing_loss, ', Testing accuracy:', testing_acc)

# Run the main function
if __name__ == "__main__":
    main()
