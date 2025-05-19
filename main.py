import torch
from dataset import TSPDataset
from graph_net import GraphResNet
from REINFORCE import REINFORCE
from last_ditch_effort import TSPTransformer

def main():
    torch.manual_seed(42)
    torch.set_num_interop_threads(8)
    
    num_samples = 1000
    min_nodes = 20     
    max_nodes = 20 # This is a leftover from when I was trying to make it train on variable graph sizes, 
                   # the assert in EncodeArray is going to crash if this is not 20 
    
    batch_size = 32
    num_epochs = 20      
    steps_per_epoch = 1000
    lr = 1e-4
    
    print("Creating dataset...")
    dataset = TSPDataset(num_samples, min_nodes=min_nodes, max_nodes=max_nodes)
    
    print("Creating model...")

    model = GraphResNet(
        hidden_dims=[64],
        init_channels=8, 
        heads=16,
        dropout=0.1
    )

    print(model)


    print("Setting up trainer...")
    trainer = REINFORCE(
        model=model,
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        baseline_update_freq=1  
    )
    
    print("Starting training...")
    trainer.train(dataset)
    
    print("Training completed!")
    print(f"Final training loss: {trainer.train_losses[-1]}")
    print(f"Baseline model updates: {trainer.baseline_updates}")
    

    print("Saving final model...")
    torch.save(model, f"Model_trained_{max_nodes}.pt")
    




if __name__ == "__main__":
    main()