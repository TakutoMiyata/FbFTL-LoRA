"""
Debug script to identify the issue with test accuracy
"""

import torch
import torch.nn as nn
from src.fedsa_ftl_model import create_model
from src.fedsa_ftl_server import FedSAFTLServer
from src.fedsa_ftl_client import FedSAFTLClient
from src.data_utils import load_cifar10_data, create_non_iid_splits, get_client_dataloader
from torch.utils.data import DataLoader
import numpy as np

def debug_federated_learning():
    """Debug the federated learning process"""
    print("=" * 80)
    print("Debugging FedSA-FTL")
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    trainset, testset = load_cifar10_data('./data')
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # Create simple splits for 2 clients
    client_indices = create_non_iid_splits(trainset, num_clients=2, alpha=0.5)
    
    # Model configuration
    config = {
        'num_classes': 10,
        'lora_r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.1,
        'freeze_backbone': True
    }
    
    # Create global model and server
    print("1. Creating global model and server...")
    global_model = create_model(config)
    server = FedSAFTLServer(global_model, device)
    
    # Initial test accuracy
    print("\n2. Testing initial global model...")
    initial_metrics = server.evaluate(test_loader)
    print(f"   Initial test accuracy: {initial_metrics['accuracy']:.2f}%")
    print(f"   Initial test loss: {initial_metrics['loss']:.4f}")
    
    # Create clients
    print("\n3. Creating 2 clients...")
    clients = []
    for i in range(2):
        client_model = create_model(config)
        client = FedSAFTLClient(i, client_model, device)
        clients.append(client)
    
    # First round of training
    print("\n4. Running first federated round...")
    
    # Get initial A parameters
    initial_global_A = server.get_global_A_params()
    print(f"   Initial global A params keys: {list(initial_global_A.keys())}")
    
    # Client training
    client_updates = []
    for i in range(2):
        print(f"\n   Training client {i}...")
        client_dataloader = get_client_dataloader(trainset, client_indices[i], batch_size=32)
        
        # Update client with global A (in first round, should be same as initial)
        clients[i].update_model(initial_global_A)
        
        # Train
        result = clients[i].train(client_dataloader, {'local_epochs': 1, 'learning_rate': 1e-3})
        print(f"   Client {i} accuracy: {result['accuracy']:.2f}%")
        
        # Check if client has different A parameters after training
        client_A = result['lora_A_params']
        for key in initial_global_A.keys():
            if key in client_A:
                diff = (client_A[key] - initial_global_A[key]).abs().mean().item()
                print(f"   Client {i} {key} diff from initial: {diff:.6f}")
        
        client_updates.append(result)
    
    # Server aggregation
    print("\n5. Server aggregation...")
    round_stats = server.federated_round(client_updates, test_loader)
    
    print(f"   Round 1 test accuracy: {round_stats['test_accuracy']:.2f}%")
    print(f"   Round 1 test loss: {round_stats['test_loss']:.4f}")
    
    # Check if global A parameters changed
    new_global_A = server.get_global_A_params()
    print("\n6. Checking if global model was updated...")
    for key in initial_global_A.keys():
        if key in new_global_A:
            diff = (new_global_A[key] - initial_global_A[key]).abs().mean().item()
            print(f"   {key} diff: {diff:.6f}")
            if diff < 1e-6:
                print(f"   WARNING: {key} was not updated!")
    
    # Direct test: manually update global model and test
    print("\n7. Direct test: manually updating global model...")
    
    # Create a simple model and train it directly
    direct_model = create_model(config)
    direct_model.to(device)
    
    optimizer = torch.optim.Adam([p for p in direct_model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few batches
    direct_model.train()
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= 50:
            break
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = direct_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Test directly trained model
    direct_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 30:  # Test on subset for speed
                break
            images, labels = images.to(device), labels.to(device)
            outputs = direct_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    direct_acc = 100. * correct / total
    print(f"   Direct model accuracy (after 50 batches): {direct_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("Debug Complete")
    print("=" * 80)

if __name__ == "__main__":
    debug_federated_learning()