# Import necessary libraries
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
#from torchvision import models
import time
from tqdm import tqdm
from sklearn.metrics import recall_score
import numpy as np
import copy
import timm

   

def train_model(model, trainloader, validationloader, device, num_epochs=100, lr=0.0005, step_size=1, gamma_lr=0.99,weight_loss = [0.3,0.7], recall_loss_ratio = 0.5):
    
    # Configure the Focal Loss function
    weights = torch.tensor(weight_loss,device=device)
    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma_lr)
    
    # Call the train() function and return the results
    trained_model, train_loss, val_loss, train_acc, val_acc = \
        train(model, trainloader, validationloader, device, criterion, optimizer, scheduler, num_epochs, recall_loss_ratio)
    
    return trained_model, train_loss, val_loss, train_acc, val_acc

# Function to train a vision transformer model
def train(model, trainloader, validationloader, device, criterion, optimizer, scheduler, num_epochs, recall_loss_ratio):
    # Initialize variables for timing and best accuracy
    since = time.time()
    best_acc = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    recall_val_loss = []
    recall_train_loss =[]
    best_model_wts = copy.deepcopy(model.state_dict())


    # Loop through each epoch
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        # Loop through training and validation phases
        for phase in ['train', 'val']:
            # Set the model to the appropriate mode and choose the data loader
            if phase == 'train':
                model.train()  # Set model to training mode
                loader = trainloader
            else:
                model.eval()   # Set model to evaluate mode
                loader = validationloader


            # Initialize variables for loss and accuracy
            
            input_size = len(loader.dataset)
            running_loss = 0.0
            running_corrects = 0.0
            recall_runningloss = 0.0
            all_labels = []
            all_probabilities = []

            # Loop through the data in the loader
            for inputs, labels in tqdm(loader):
                # Move inputs and labels to the device
    
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Compute the outputs and loss
                with torch.set_grad_enabled(phase == 'train'):
                    #print("inputs",inputs.size())
                    outputs = model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                    #print("outputs",outputs.size())
                    labels_indices = torch.argmax(labels, dim=1)
                    loss = criterion(outputs, labels_indices)
                    _, preds = torch.max(probabilities, 1)
                    true_positives = torch.sum((preds == 1) & (labels_indices == 1))
                    possible_positives = (labels_indices == 1).float().sum()
                    recall = true_positives / (possible_positives + 1e-9)
                    recall_loss = (1 - recall)
                    
                    # If in training phase, update the weights
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Update the running loss and accuracy
                recall_runningloss+= recall_loss * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == torch.argmax(labels.data, dim=1))
                
            # Update the learning rate scheduler in the training phase
            if phase == 'train':
                scheduler.step()
            
            # Calculate and print the epoch loss and accuracy
            epoch_loss = running_loss / input_size
            epoch_acc = running_corrects.double() / input_size
            epoch_recall_loss =  recall_runningloss/input_size
            
            # Append the results to the appropriate lists
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu().numpy())
                recall_train_loss.append(epoch_recall_loss.cpu().numpy().item())

            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.cpu().numpy())
                recall_val_loss.append(epoch_recall_loss.cpu().numpy().item())
                
                weight_score = epoch_acc.cpu().numpy() - epoch_recall_loss.cpu().numpy().item()*recall_loss_ratio 
                if weight_score > best_acc:
                    best_acc = weight_score
                    best_model_wts = copy.deepcopy(model.state_dict())
                   
            print("{} Loss: {:.4f} Acc: {:.4f} Recall_loss: {:.4f}".format(phase, epoch_loss, epoch_acc,epoch_recall_loss))
            
        print()
        print()
    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, 'best_model_test.pth')
    print("save the model successfully")
    # model.load_state_dict(best_model_wts), 
    return model, train_loss, val_loss, train_acc, val_acc
