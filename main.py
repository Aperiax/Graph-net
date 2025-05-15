import torch
from dataset import TSPDataset
from graph_net import GraphResNet
from REINFORCE import REINFORCE

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.set_num_interop_threads(8)
    
    # Parameters
    num_samples = 100
    min_nodes = 10      
    max_nodes = 10  
    
    batch_size = 64 
    num_epochs = 1      
    steps_per_epoch = 1000
    lr = 1e-5
    
    # Create dataset
    print("Creating dataset...")
    dataset = TSPDataset(num_samples, min_nodes=min_nodes, max_nodes=max_nodes)
    
    # Create model
    print("Creating model...")

    model = GraphResNet(
        hidden_dims=[64,128,256,512], 
        init_channels=8, 
        heads=16, 
        dropout=0., 
        )
    print(model)


    # Create trainer
    print("Setting up trainer...")
    trainer = REINFORCE(
        model=model,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        baseline_update_freq=1  
    )
    
    # Train model
    print("Starting training...")
    trainer.train(dataset)
    
    print("Training completed!")
    print(f"Final training loss: {trainer.train_losses[-1]}")
    print(f"Baseline model updates: {trainer.baseline_updates}")
    

    print("Saving final model...")
    torch.save(model, f"Model_trained_{max_nodes}.pt")
    




if __name__ == "__main__":
    main()