#!/usr/bin/env python3
"""
EXACT replica of main_TMLCN_CIFAR10_VGG16.py adapted for ResNet50 with CIFAR-100
Non-IID data distribution functionality included
All learning methods, hyperparameters, and noise mechanisms remain similar
"""

import torch
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
import math
from collections import deque
import copy
import argparse
import os
import sys

# Add src directory to path for non-IID functionality
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_utils import create_non_iid_splits, create_iid_splits, analyze_data_distribution

#######################################
### PRE-TRAINED MODELS AVAILABLE HERE
## https://pytorch.org/docs/stable/torchvision/models.html
from torchvision import models
#######################################

now = datetime.datetime.now()

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

##########################
### SETTINGS (EXACTLY as in original, adapted for CIFAR-100)
##########################

def parse_args():
    """Parse command line arguments - ONLY for non-IID settings"""
    parser = argparse.ArgumentParser(description='FbFTL CIFAR-100 ResNet50 with Non-IID')
    
    # ONLY non-IID related arguments are added
    parser.add_argument('--data_split', type=str, default='iid', 
                       choices=['iid', 'non_iid'], help='Data distribution type')
    parser.add_argument('--num_clients', type=int, default=10, 
                       help='Number of clients for non-IID split')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha for non-IID (lower = more heterogeneous)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show data distribution analysis')
    
    # All original settings remain as constants (no command line override)
    return parser.parse_args()

args = parse_args()

# EXACT original settings (adapted for CIFAR-100)
FL_type = 'FbFTL'  # 'FL', 'FTLf', 'FTLc', 'FbFTL'
train_set_denominator = 'full'  # 'full', int <= 50000

# Hyperparameters (adapted for CIFAR-100)
NUM_CLASSES = 100  # Changed from 10 to 100 for CIFAR-100
U_clients = 6250  # number of clients, 50000/8
random_seed = 1
learning_rate = 1e-2  # 1e-3, 0.05, 1e-2
num_epochs = 200  # 10, 300, 200
batch_size = 64  # 128, 128, (out of memory:64)
momentum = 0.9  # None, 0.9
lr_decay = 5e-4  # 1e-6, 5e-4

# FL type settings (EXACT from original)
if FL_type == 'FL':
    transfer, full = False, True
    sigma = 0.
elif FL_type == 'FTLf':
    transfer, full = True, True  
    sigma = 0.
elif FL_type == 'FTLc':
    transfer, full = True, False 
    sigma = 0.
elif FL_type == 'FbFTL':
    transfer, full = True, False 
    sigma = 0  # 0.8? relative std for additive gaussian noise on features
    saved_noise = True  # save noise at beginning
else:
    raise ValueError('Unknown FL_type: ' + FL_type)

relative_noise_type = 'all_std'  # 'individual', 'all_std'
packet_loss_rate = 0.  # 0, 0.05, 0.1, 0.15
quan_digit = 32  # digits kept after feature quantization: None or int
sparse_rate = 0.9  # ratio of uplink elements kept after sparsification: None or (0,1]

class ErrorFeedback(object):
    queue = deque(maxlen=U_clients)
    temp = deque()

if (quan_digit or sparse_rate) and FL_type != 'FbFTL':
    errfdbk = ErrorFeedback()

write_hist = False  # Disable file writing for this version

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = learning_rate * (0.5 ** ((epoch * 10) // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# EXACT seed setting from original
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed)
torch.manual_seed(random_seed)

##########################
### CIFAR100 DATASET (changed from CIFAR10)
##########################

custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# Changed to CIFAR100
train_dataset = datasets.CIFAR100(root='data', train=True, transform=custom_transform, download=True)
test_dataset = datasets.CIFAR100(root='data', train=False, transform=custom_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)

# MODIFIED PART: Add non-IID data distribution
if args.data_split == 'non_iid':
    print(f"\n=== Creating Non-IID Data Distribution ===")
    print(f"Number of clients: {args.num_clients}")
    print(f"Dirichlet alpha: {args.alpha}")
    
    # Create non-IID splits for training data
    client_train_indices = create_non_iid_splits(
        train_dataset, 
        num_clients=args.num_clients,
        alpha=args.alpha
    )
    
    # Create non-IID splits for test data (using same alpha)
    client_test_indices = create_non_iid_splits(
        test_dataset, 
        num_clients=args.num_clients,
        alpha=args.alpha
    )
    
    if args.verbose:
        print("\n=== Training Data Distribution Analysis ===")
        analyze_data_distribution(train_dataset, client_train_indices, NUM_CLASSES)
        print("\n=== Test Data Distribution Analysis ===")
        analyze_data_distribution(test_dataset, client_test_indices, NUM_CLASSES)
    
    # For this implementation, we simulate by using a subset that represents
    # the aggregated data from all clients (maintaining original single-loader structure)
    all_train_indices = []
    for client_indices in client_train_indices:
        all_train_indices.extend(client_indices)
    
    all_test_indices = []
    for client_indices in client_test_indices:
        all_test_indices.extend(client_indices)
    
    # Create subsets with aggregated indices
    train_subset = torch.utils.data.Subset(train_dataset, all_train_indices)
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, 
                             num_workers=8, shuffle=True, pin_memory=True)
    train_set_len = len(all_train_indices)
    
    test_subset = torch.utils.data.Subset(test_dataset, all_test_indices)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, 
                            num_workers=8, shuffle=False, pin_memory=True)
    
    print(f"Total training samples after non-IID simulation: {train_set_len}")
    print(f"Total test samples after non-IID simulation: {len(all_test_indices)}")

else:
    # Original IID case
    if train_set_denominator == 'full':
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                 num_workers=8, shuffle=True, pin_memory=True)
        train_set_len = len(train_dataset)
    else:
        selected_list = list(range(0, len(train_dataset), train_set_denominator))
        trainset_1 = torch.utils.data.Subset(train_dataset, selected_list)
        train_loader = torch.utils.data.DataLoader(dataset=trainset_1, batch_size=batch_size, 
                                                  num_workers=8, shuffle=True, pin_memory=True)
        train_set_len = len(selected_list)
    
    # Keep original test loader for IID case
    # test_loader is already defined above

##########################
### LOAD MODEL - ResNet50 instead of VGG16
##########################

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer - adapted for ResNet features"""
    def __init__(self, sigma=0, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        if saved_noise:
            # ResNet50 feature dimension is 2048 instead of VGG's 4096
            self.register_buffer('noise', torch.empty(train_set_len*2048).normal_(mean=0,std=1))
            self.i = 0
        else:
            self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and quan_digit:
            x = torch.round((2**quan_digit-1) / torch.max(x) * x) * torch.max(x) / (2**quan_digit-1)
        
        if self.training and self.sigma != 0:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(random_seed)
            torch.manual_seed(random_seed)
            
            if relative_noise_type == 'individual':
                scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
                sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            elif relative_noise_type == 'all_std':
                x_std = torch.std(x.detach()) if self.is_relative_detach else torch.std(x)
                if saved_noise:
                    sampled_noise = torch.reshape(
                        self.noise[self.i*batch_size*2048 : (self.i+1)*batch_size*2048],
                        (-1, 2048)
                    ).detach().float() * x_std * self.sigma
                    self.i = self.i + 1 if (self.i+1)*batch_size*2048<train_set_len*2048 else 0
                else:
                    sampled_noise = self.noise.expand(*x.size()).float().normal_(std=x_std*self.sigma)
            x = x + sampled_noise
        return x 
        
    def set_sigma(self, sigma):
        self.sigma = sigma

# Load ResNet50 instead of VGG16
if not full:
    model = models.resnet50(pretrained=True)
    
    # Freeze all parameters except final layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the number of features in the final layer
    num_features = model.fc.in_features  # 2048 for ResNet50
    
    if FL_type == 'FbFTL':
        # Replace the final layer with our custom head including GaussianNoise
        model.fc = nn.Sequential(
            GaussianNoise(sigma=sigma),
            nn.Linear(num_features, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
    else:
        # Replace the final layer without GaussianNoise
        model.fc = nn.Sequential(
            nn.Linear(num_features, 2048), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
    
    # Make the new layers trainable
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Also make the last residual block trainable for better adaptation
    for param in model.layer4.parameters():
        param.requires_grad = True

else:  # full training
    if transfer:
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)
    
    # Get the number of features in the final layer
    num_features = model.fc.in_features  # 2048 for ResNet50
    
    # Replace the final layer for CIFAR-100
    model.fc = nn.Sequential(
        nn.Linear(num_features, 2048), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(512, NUM_CLASSES)
    )
    
    # Make all parameters trainable for full training
    for param in model.parameters():
        param.requires_grad = True

def Gaussian_noise_to_weights(m):
    """EXACT from original"""
    if sigma != 0:
        with torch.no_grad():
            for param in m.parameters():
                if param.requires_grad:
                    if relative_noise_type == 'individual':
                        scale = sigma * param.grad.detach()
                        noise = torch.tensor(0).to(DEVICE)
                        sampled_noise = noise.expand(*param.size()).float().normal_() * scale
                    elif relative_noise_type == 'all_std':
                        param_grad_std = torch.std(param.grad.detach()) 
                        noise = torch.tensor(0).to(DEVICE)
                        sampled_noise = noise.expand(*param.size()).float().normal_(std=param_grad_std*sigma)
                    param.add_(sampled_noise)

def Errfdbk_to_weights(m):
    """EXACT from original"""
    print("inner model.apply")
    with torch.no_grad():
        for param in m.parameters():
            print('len(param)', len(param))
            if param.requires_grad:
                print('len(errfdbk.queue)', len(errfdbk.queue))
                print('len(errfdbk.temp)', len(errfdbk.temp))
                p_grad = param.grad.detach()
                if err_flag:
                    p_grad += errfdbk.temp.popleft()
                p_grad_qs = copy.deepcopy(p_grad)
                if sparse_rate:
                    pass
                if quan_digit:
                    pass
                err = p_grad_qs - p_grad
                param.add_(err)
                errfdbk.temp.append(-err)
    print('seems good')

# EXACT packet loss simulation from original
if FL_type == 'FbFTL':
    received_batches_FbFTL = np.ones(len(train_loader))
    received_batches_FbFTL[:int(len(train_loader)*packet_loss_rate)] = 0
    np.random.shuffle(received_batches_FbFTL)
    
def Packet_Received(batch_idx):
    if FL_type == 'FbFTL':
        return received_batches_FbFTL[batch_idx]
    else:
        return np.random.choice(2, p=[packet_loss_rate, 1-packet_loss_rate])

##########################
### TRAIN MODEL (EXACT from original)
##########################

model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lr_decay, momentum=momentum)

def compute_accuracy(model, data_loader):
    """EXACT from original"""
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100

def compute_client_accuracies(model, test_dataset, client_test_indices, batch_size=64):
    """
    Compute accuracy for each individual client's test data
    """
    model.eval()
    client_accuracies = []
    
    for client_id, indices in enumerate(client_test_indices):
        if len(indices) == 0:
            client_accuracies.append(0.0)
            continue
            
        # Create dataloader for this client's test data
        client_test_subset = torch.utils.data.Subset(test_dataset, indices)
        client_test_loader = DataLoader(
            client_test_subset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Compute accuracy for this client
        correct_pred, num_examples = 0, 0
        with torch.no_grad():
            for features, targets in client_test_loader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                
                logits = model(features)
                _, predicted_labels = torch.max(logits, 1)
                num_examples += targets.size(0)
                correct_pred += (predicted_labels == targets).sum()
        
        client_accuracy = correct_pred.float() / num_examples * 100
        client_accuracies.append(client_accuracy.item())
    
    return client_accuracies

def compute_epoch_loss(model, data_loader):
    """EXACT from original"""
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(data_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

# Store client indices globally for evaluation
global_client_test_indices = None
if args.data_split == 'non_iid':
    global_client_test_indices = client_test_indices

print(f"\n{'='*80}")
print(f"Training Configuration")
print(f"{'='*80}")
print(f"Model: ResNet50")
print(f"Dataset: CIFAR-100")
print(f"FL Type: {FL_type}")
print(f"Data Distribution: {args.data_split}")
if args.data_split == 'non_iid':
    print(f"Number of Clients: {args.num_clients}")
    print(f"Dirichlet Alpha: {args.alpha}")
print(f"Training Samples: {train_set_len}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Epochs: {num_epochs}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Momentum: {momentum}")
print(f"Weight Decay: {lr_decay}")
print(f"Noise Sigma: {sigma}")
print(f"Packet Loss Rate: {packet_loss_rate}")
print(f"{'='*80}")

# EXACT training loop from original
start_time = time.time()

for epoch in range(num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        if Packet_Received(batch_idx):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### PRIVACY NOISE
            if FL_type != 'FbFTL':
                model.apply(Gaussian_noise_to_weights)

            ### Sparsification/Quantization with Error Feedback
            if (quan_digit or sparse_rate) and FL_type != 'FbFTL':
                print('main loop: len(errfdbk.queue)', len(errfdbk.queue))
                if len(errfdbk.queue) < U_clients:
                    errfdbk.temp = deque()
                    err_flag = False
                else:
                    errfdbk.temp = errfdbk.queue.popleft()
                    err_flag = True
                model.apply(Errfdbk_to_weights)
                print('completed one cycle!')
                errfdbk.queue.append(errfdbk.temp)
        
            ### LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' % 
                      (epoch+1, num_epochs, batch_idx, len(train_loader), cost))

    # EXACT evaluation from original
    model.eval()
    accuracy = compute_accuracy(model, test_loader)
    loss = compute_epoch_loss(model, test_loader)
    
    # Additional evaluation for non-IID: individual client accuracies
    if args.data_split == 'non_iid' and (epoch + 1) % 20 == 0:  # Every 20 epochs
        client_accuracies = compute_client_accuracies(model, test_dataset, global_client_test_indices, batch_size)
        avg_client_accuracy = sum(client_accuracies) / len(client_accuracies)
        std_client_accuracy = (sum([(acc - avg_client_accuracy)**2 for acc in client_accuracies]) / len(client_accuracies)) ** 0.5
        
        print('Epoch: %03d/%03d | Test: %.3f%% | Loss: %.3f' % 
              (epoch+1, num_epochs, accuracy, loss))
        print('  Individual Client Accuracies: [%s]' % 
              ', '.join(['%.1f' % acc for acc in client_accuracies]))
        print('  Client Accuracy Stats: Avg=%.3f%%, Std=%.3f%%' % 
              (avg_client_accuracy, std_client_accuracy))
    else:
        with torch.set_grad_enabled(False):
            print('Epoch: %03d/%03d | Test: %.3f%% | Loss: %.3f' % 
                  (epoch+1, num_epochs, accuracy, loss))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

# EXACT final evaluation from original
with torch.set_grad_enabled(False):
    accuracy = compute_accuracy(model, test_loader)
    print('Test accuracy: %.2f%%' % (accuracy))

print(f"\n{'='*80}")
print("Training Complete!")
print(f"{'='*80}")
print(f"Model: ResNet50")
print(f"Dataset: CIFAR-100 ({NUM_CLASSES} classes)")
print(f"Final Test Accuracy: {accuracy:.2f}%")
print(f"Data Distribution: {args.data_split}")
if args.data_split == 'non_iid':
    print(f"Clients: {args.num_clients}, Alpha: {args.alpha}")
print(f"Total Training Time: {(time.time() - start_time)/60:.2f} minutes")
print(f"{'='*80}")