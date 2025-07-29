import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import datetime
import matplotlib.pyplot as plt
import os

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
### SETTINGS
##########################

FL_type = 'FbFTL_Rotation_Experiment' 
train_set_denominator = 'full'  # 'full', int <= 50000  # pick a subset with 50000/int training samples

# Hyperparameters
NUM_CLASSES = 10
random_seed = 1 
learning_rate = 1e-2  
num_epochs = 50  # Reduced for rotation experiment
batch_size = 64  
momentum = 0.9  
lr_decay = 5e-4  

# Rotation angles to test (degrees)
rotation_angles = [0, 15, 30, 45, 60, 75, 90, 135, 180]

write_hist = True

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = learning_rate * (0.5 ** ((epoch * 10) // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed) # Sets the seed for generating random numbers for the current GPU. 
torch.manual_seed(random_seed) # sets the seed for generating random numbers.

if write_hist:
    file1 = open('rotation_experiment_history.txt', 'w')  # Create new file for rotation experiment
    file1.write('Rotation Experiment Results\n')
    file1.write('Time: ' + str(now.year) + ' ' + str(now.month) + ' ' + str(now.day) + ' ' 
                + str(now.hour) + ' ' + str(now.minute) + ' ' + str(now.second) + '\n')
    file1.write('FL_type: ' + FL_type + ', train_deno: ' + str(train_set_denominator) + '\n')
    file1.write('Rotation angles tested: ' + str(rotation_angles) + '\n\n')
    file1.close()

##########################
### CIFAR10 DATASET
##########################

# Standard transform for training (no rotation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# Function to create rotated test transform
def create_rotated_transform(angle):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(angle, angle)),  # Fixed rotation angle
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

# Load datasets
train_dataset = datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True)
original_test_dataset = datasets.CIFAR10(root='data', train=False, transform=train_transform)

# Create training data loader
if train_set_denominator == 'full':
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
else:
    selected_list = list(range(0, len(train_dataset), train_set_denominator))
    trainset_1 = torch.utils.data.Subset(train_dataset, selected_list)
    train_loader = torch.utils.data.DataLoader(dataset=trainset_1, batch_size=batch_size, num_workers=8, shuffle=True)

##########################
### LOAD MODEL
##########################

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier[3].requires_grad = True
model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 4096), 
    nn.ReLU(), 
    nn.Dropout(0.5), 
    nn.Linear(4096, 4096), 
    nn.ReLU(), 
    nn.Dropout(0.5), 
    nn.Linear(4096, 512), 
    nn.ReLU(), 
    nn.Dropout(0.5), 
    nn.Linear(512, NUM_CLASSES)
)

##########################
### TRAIN MODEL
##########################

model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lr_decay, momentum=momentum)

def compute_accuracy(model, data_loader):
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

def compute_epoch_loss(model, data_loader):
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

print("Starting training on original (non-rotated) data...")
print("="*60)

start_time = time.time()
original_test_loader = DataLoader(dataset=original_test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    # Test on original (non-rotated) data
    model.eval()
    accuracy = compute_accuracy(model, original_test_loader)
    loss = compute_epoch_loss(model, original_test_loader)
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d | Original Test: %.3f%% | Loss: %.3f' % (epoch+1, num_epochs, accuracy, loss))
        if write_hist:
            file1 = open('rotation_experiment_history.txt', 'a')
            file1.write('Epoch: %03d/%03d | Original Test: %.3f%% | Loss: %.3f\n' % (epoch+1, num_epochs, accuracy, loss))
            file1.close()

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

##########################
### ROTATION EXPERIMENT
##########################

print("\n" + "="*60)
print("Starting rotation experiment...")
print("Testing model performance on rotated images")
print("="*60)

# Store results for analysis
rotation_results = {}

# Test on original data first
with torch.set_grad_enabled(False):
    original_accuracy = compute_accuracy(model, original_test_loader)
    rotation_results[0] = original_accuracy.item()
    print('Original (0°) Test accuracy: %.2f%%' % original_accuracy)

# Test on rotated data
for angle in rotation_angles[1:]:  # Skip 0 degrees as we already tested it
    print(f"\nTesting with {angle}° rotation...")
    
    # Create rotated test dataset
    rotated_transform = create_rotated_transform(angle)
    rotated_test_dataset = datasets.CIFAR10(root='data', train=False, transform=rotated_transform)
    rotated_test_loader = DataLoader(dataset=rotated_test_dataset, batch_size=batch_size, 
                                   num_workers=8, shuffle=False)
    
    # Compute accuracy on rotated data
    with torch.set_grad_enabled(False):
        rotated_accuracy = compute_accuracy(model, rotated_test_loader)
        rotation_results[angle] = rotated_accuracy.item()
        accuracy_drop = original_accuracy - rotated_accuracy
        
        print(f'Rotated ({angle}°) Test accuracy: {rotated_accuracy:.2f}%')
        print(f'Accuracy drop: {accuracy_drop:.2f}%')
        
        if write_hist:
            file1 = open('rotation_experiment_history.txt', 'a')
            file1.write(f'Rotated ({angle}°) Test accuracy: {rotated_accuracy:.2f}%\n')
            file1.write(f'Accuracy drop: {accuracy_drop:.2f}%\n')
            file1.close()

##########################
### ANALYSIS AND VISUALIZATION
##########################

print("\n" + "="*60)
print("ROTATION EXPERIMENT RESULTS SUMMARY")
print("="*60)

# Print summary table
print("Rotation Angle | Test Accuracy | Accuracy Drop")
print("-" * 45)
for angle in rotation_angles:
    if angle == 0:
        print(f"{angle:12d}° | {rotation_results[angle]:11.2f}% | {0:11.2f}%")
    else:
        accuracy_drop = rotation_results[0] - rotation_results[angle]
        print(f"{angle:12d}° | {rotation_results[angle]:11.2f}% | {accuracy_drop:11.2f}%")

# Save detailed results
if write_hist:
    file1 = open('rotation_experiment_history.txt', 'a')
    file1.write('\n\nROTATION EXPERIMENT RESULTS SUMMARY\n')
    file1.write('='*60 + '\n')
    file1.write("Rotation Angle | Test Accuracy | Accuracy Drop\n")
    file1.write('-' * 45 + '\n')
    for angle in rotation_angles:
        if angle == 0:
            file1.write(f"{angle:12d}° | {rotation_results[angle]:11.2f}% | {0:11.2f}%\n")
        else:
            accuracy_drop = rotation_results[0] - rotation_results[angle]
            file1.write(f"{angle:12d}° | {rotation_results[angle]:11.2f}% | {accuracy_drop:11.2f}%\n")
    file1.close()

# Create visualization
plt.figure(figsize=(12, 8))

# Plot 1: Accuracy vs Rotation Angle
plt.subplot(2, 1, 1)
angles = list(rotation_results.keys())
accuracies = list(rotation_results.values())
plt.plot(angles, accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Test Accuracy (%)')
plt.title('Model Performance vs Image Rotation Angle\n(Feature Extractor Obsolescence Experiment)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation_angles)

# Plot 2: Accuracy Drop vs Rotation Angle
plt.subplot(2, 1, 2)
accuracy_drops = [rotation_results[0] - rotation_results[angle] for angle in angles]
plt.plot(angles, accuracy_drops, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Accuracy Drop (%)')
plt.title('Accuracy Degradation due to Image Rotation')
plt.grid(True, alpha=0.3)
plt.xticks(rotation_angles)

plt.tight_layout()
plt.savefig('rotation_experiment_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nExperiment completed! Results saved to:")
print(f"- rotation_experiment_history.txt")
print(f"- rotation_experiment_results.png")

print(f"\nKey Findings:")
print(f"- Original accuracy: {rotation_results[0]:.2f}%")
worst_angle = max(rotation_angles[1:], key=lambda x: rotation_results[0] - rotation_results[x])
worst_drop = rotation_results[0] - rotation_results[worst_angle]
print(f"- Worst performance at {worst_angle}° rotation: {rotation_results[worst_angle]:.2f}% (drop: {worst_drop:.2f}%)")

# Calculate average accuracy drop
avg_drop = np.mean([rotation_results[0] - rotation_results[angle] for angle in rotation_angles[1:]])
print(f"- Average accuracy drop across all rotations: {avg_drop:.2f}%")
