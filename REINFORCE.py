import torch
from copy import deepcopy
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm


class REINFORCE:
    """
    Implementation of REINFORCE with greedy rollout baseline as in Kool et al. 2019
    "Attention, learn to solve routing problems!"
    """
    def __init__(
            self, 
            model, 
            lr=1e-5, 
            weight_decay=5e-4,
            batch_size=512,
            num_epochs=100,
            steps_per_epoch=2500,
            significance_level=0.1,
            baseline_update_freq=1,
            revisit_bonus_factor=1.0,  
            device='cuda' if torch.cuda.is_available() else 'cpu'
        ):
        """
        Args:
            model: GraphNet model to train
            lr: Learning rate for Adam optimizer
            beta: Beta parameter for Adam
            batch_size: Batch size for training
            num_epochs: Number of epochs to train
            steps_per_epoch: Number of steps per epoch
            significance_level: Significance level for paired t-test when updating baseline
            baseline_update_freq: Frequency (in epochs) to check for baseline updates
            revisit_bonus_factor: Scaling factor for revisit bonuses
            device: Device to train on
        """
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # just like in the original paper, the baseline model is deepcopied, since it won't get that frequent updates
        self.baseline_model = deepcopy(model).to(device)
        self.baseline_model.eval()
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.significance_level = significance_level
        self.baseline_update_freq = baseline_update_freq
        self.revisit_bonus_factor = revisit_bonus_factor
        self.device = device
        
        # Statistics for tracking
        self.baseline_updates = 0
        self.train_losses = []
        self.eval_stats = []
        self.revisit_stats = []  

    def _custom_collate_fn(self, batch): 
        # to sidestep DataLoader autostacking tensors
        return batch 

    def _compute_tour_length(self, tour, batch_data):
        """
        Compute the length of tours
        """
        batch_size = tour.size(0)
        tour_lengths = torch.zeros(batch_size, device=self.device)
        
        for b in range(batch_size):
            edge_lookup = batch_data[b][3]
            tour_b = tour[b]
            total_distance = 0.0
            
            unique_nodes = torch.unique(tour_b).size(0)
            all_nodes = edge_lookup.size(0)
            
            for i in range(tour_b.size(0) - 1): 
                src = tour_b[i]
                tgt = tour_b[i+1]
                total_distance += edge_lookup[src, tgt]
            
            # Complete the tour by returning to start
            total_distance += edge_lookup[tour_b[-1], tour_b[0]]
            
            # check if all were visited
            if unique_nodes < all_nodes:
                # heavily penalize incomplete tours
                total_distance += 100000 * (all_nodes - unique_nodes)
            
            tour_lengths[b] = total_distance
    
        return tour_lengths
    
    def _compute_clean_tour_length(self, tour, batch_data_item):
        """
        Compute the length of a 'clean' tour that visits each node exactly once
        """
        edge_lookup = batch_data_item[3]
        
        # clean tour with no revisits
        seen = set()
        clean_tour = []
        
        for node in tour:
            node_idx = node.item()
            if node_idx not in seen:
                clean_tour.append(node)
                seen.add(node_idx)
                
        # Ensure we have at least one node
        if not clean_tour:
            return torch.tensor(float('inf'), device=self.device)
            
        clean_tour = torch.stack(clean_tour)
        
        # Add return to start

        clean_tour = torch.cat([clean_tour, clean_tour[0:1]])
        
        # Compute clean tour length
        total_distance = 0.0
        for i in range(clean_tour.size(0) - 1):
            src = clean_tour[i]
            tgt = clean_tour[i+1]
            total_distance += edge_lookup[src, tgt]
            
        return total_distance

    def _sample_solution(self, model, batch_data, greedy=False):
        """
        Sample a solution from the model
        
        """
        if greedy:
            return model(batch_data, greedy=greedy)    
            
        else:
            # Regular stochastic sampling
            return model(batch_data, greedy=greedy)
    
    def _paired_t_test(self, current_model_lengths, baseline_model_lengths):
        """
        Perform a one-sided paired t-test to check if current model is significantly better
        """
        _, p_value = stats.ttest_rel(
            baseline_model_lengths.cpu().numpy(),
            current_model_lengths.cpu().numpy(),
            alternative='greater'  # H1: baseline_lengths > current_lengths
        )
        
        return p_value < self.significance_level
    
    def train(self, dataset):
        """
        Train the model using REINFORCE with rollout baseline
        
        """
        train_dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self._custom_collate_fn, 
        )
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_revisit_bonus_count = 0
            epoch_revisit_count = 0
            epoch_beneficial_revisit_count = 0
            tours = None
            
            for batch_data in tqdm(train_dataloader, f"Training epoch: {epoch}", total=len(train_dataloader), dynamic_ncols=True, leave=True):
                batch_data = [
                    (coords.to(self.device), 
                    edge_index.to(self.device), 
                    edge_attr.to(self.device), 
                    edge_lookup.to(self.device))
                    for coords, edge_index, edge_attr, edge_lookup in batch_data
                ]
                
                tours, log_probs = self._sample_solution(self.model, batch_data, greedy=False)
                baseline_tours, _ = self._sample_solution(self.baseline_model, batch_data, greedy=False)
                
                tour_lengths = self._compute_tour_length(tours, batch_data)
                baseline_lengths = self._compute_tour_length(baseline_tours, batch_data)
                
                # Calculate base advantage
                advantage = baseline_lengths - tour_lengths
                
                batch_revisit_count = 0
                batch_beneficial_revisit_count = 0
                
                for b in range(len(batch_data)):
                    # original node coutns
                    edge_lookup = batch_data[b][3]
                    all_nodes = edge_lookup.size(0)
                    
                    # count unique nodes in the tour
                    tour_b = tours[b]
                    unique_nodes = torch.unique(tour_b).size(0)
                    
                    # Are there revisits beyond the closing node?
                    has_revisits = unique_nodes < all_nodes
                    
                    if has_revisits:
                        batch_revisit_count += 1
                        
                        # wath the tour length would be without revisits
                        clean_tour_length = self._compute_clean_tour_length(tour_b, batch_data[b])
                        
                        # if tour with revisits is better than clean tour
                        if tour_lengths[b] < clean_tour_length:
                            # how much better and apply bonus
                            improvement = (clean_tour_length - tour_lengths[b]) / clean_tour_length
                            bonus = improvement * self.revisit_bonus_factor
                            
                            # adding the revisit bonus
                            advantage[b] = advantage[b] * (1.0 + bonus)
                            
                            epoch_revisit_bonus_count += 1
                            batch_beneficial_revisit_count += 1
                
                epoch_revisit_count += batch_revisit_count
                epoch_beneficial_revisit_count += batch_beneficial_revisit_count
                
                loss = torch.mean(-advantage * log_probs.sum(dim=1))

                self.optimizer.zero_grad()
                loss.backward()
                # this doesn't really influence the learning apart for making the gradients more stable
                # or at least I didn't notice any change with or without it
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / self.steps_per_epoch
            self.train_losses.append(avg_epoch_loss)
            
            revisit_rate = epoch_revisit_count / self.steps_per_epoch if self.steps_per_epoch > 0 else 0
            beneficial_rate = epoch_beneficial_revisit_count / epoch_revisit_count if epoch_revisit_count > 0 else 0
            
            self.revisit_stats.append({
                'epoch': epoch,
                'revisit_rate': revisit_rate,
                'beneficial_rate': beneficial_rate,
                'bonus_applied_count': epoch_revisit_bonus_count
            })
            
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_epoch_loss:.4f}")
            print(f"Revisit rate: {revisit_rate:.2%}, Beneficial: {beneficial_rate:.2%}, Bonuses: {epoch_revisit_bonus_count}")
            print("Training complete") 

            # trigger the update every update_freq epochs 
            if (epoch + 1) % self.baseline_update_freq == 0:
                self._evaluate_and_update_baseline(dataset)
            if epoch % 5 == 0: 
                torch.save(self.model, f"cpt_{epoch}.pt")
        
        # once we're done, just save the baseline to be sure
        print("Saving the baseline model") 
        torch.save(self.baseline_model, "baseline.pt")   

    def _analyze_tour(self, tour, edge_lookup):
        """
        Analyzes a tour for optimality and handling of revisits
        """
        unique_nodes = torch.unique(tour).size(0)
        total_nodes = tour.size(0)
        revisit_count = total_nodes - unique_nodes
        
        total_distance = 0.0
        for i in range(len(tour) - 1):
            src = tour[i]
            tgt = tour[i+1]
            total_distance += edge_lookup[src, tgt]
                
        total_distance += edge_lookup[tour[-1], tour[0]]

        unique_tour = []
        for node in tour:
            if node.item() not in [n.item() for n in unique_tour]:
                unique_tour.append(node)
                
        unique_tour = torch.stack(unique_tour)
        unique_distance = 0.0

        for i in range(len(unique_tour) - 1):
            src = unique_tour[i]
            tgt = unique_tour[i+1]
            unique_distance += edge_lookup[src, tgt]
                
        unique_distance += edge_lookup[unique_tour[-1], unique_tour[0]]

        # Check if revisits improved the tour
        improved_by_revisit = total_distance < unique_distance if revisit_count > 0 else False
        
        return {
            'total_length': total_distance.item(),
            'unique_nodes': unique_nodes,
            'revisit_count': revisit_count,
            'improved_by_revisit': improved_by_revisit,
            'improvement_amount': (unique_distance - total_distance).item() if improved_by_revisit else 0.0
        }
    
    def _evaluate_and_update_baseline(self, dataset, num_eval_instances=100):
        """
        Evaluate current model and update baseline if significantly (alpha = 0.05) better
        """
        self.model.eval()
        self.baseline_model.eval()
        
        eval_dataset = torch.utils.data.Subset(
            dataset, 
            indices=torch.randperm(len(dataset))[:num_eval_instances]
        )
        print(len(eval_dataset))
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=self._custom_collate_fn
        )
        
        current_lengths = []
        baseline_lengths = []
        
        with torch.no_grad():
            for batch_data in eval_dataloader:
                # Move data to device
                device_batch_data = [
                    (coords.to(self.device), 
                     edge_index.to(self.device), 
                     edge_attr.to(self.device), 
                     edge_lookup.to(self.device))
                    for coords, edge_index, edge_attr, edge_lookup in batch_data
                ]

                # Get greedy solutions from both models
                current_tours, _ = self._sample_solution(self.model, device_batch_data, greedy=True)
                baseline_tours, _ = self._sample_solution(self.baseline_model, device_batch_data, greedy=True)
                
                c_lengths = self._compute_tour_length(current_tours, device_batch_data)
                b_lengths = self._compute_tour_length(baseline_tours, device_batch_data)
                
                current_lengths.append(c_lengths)
                baseline_lengths.append(b_lengths)
        
        current_lengths = torch.cat(current_lengths)
        baseline_lengths = torch.cat(baseline_lengths)
        
        current_mean = current_lengths.mean().item()
        baseline_mean = baseline_lengths.mean().item()
        improvement = (baseline_mean - current_mean) / baseline_mean * 100

        sample_batch = next(iter(eval_dataloader))

        device_sample_batch = [
            (coords.to(self.device), 
             edge_index.to(self.device), 
             edge_attr.to(self.device), 
             edge_lookup.to(self.device)) 
            for coords, edge_index, edge_attr, edge_lookup in sample_batch
        ]
        
        current_tours, _ = self._sample_solution(self.model, device_sample_batch, greedy=True)
        
        # Use the first tour and first graph's edge lookup for analysis
        edge_lookup = device_sample_batch[0][3]
        tour_analysis = self._analyze_tour(current_tours[0], edge_lookup)
        
        self.eval_stats.append({
            'current_mean': current_mean,
            'baseline_mean': baseline_mean,
            'improvement': improvement,
            'revisit_stats': tour_analysis
        })
        
        print(f"Sample tour analysis:")
        print(f"  - Unique nodes visited: {tour_analysis['unique_nodes']}")
        print(f"  - Revisits: {tour_analysis['revisit_count']}")
        if tour_analysis['revisit_count'] > 0:
            print(f"  - Revisits improved tour: {'Yes' if tour_analysis['improved_by_revisit'] else 'No'}")
            if tour_analysis['improved_by_revisit']:
                print(f"  - Improvement amount: {tour_analysis['improvement_amount']:.4f}")
        
        print(f"Evaluation: Current model mean tour length: {current_mean:.4f}")
        print(f"Evaluation: Baseline model mean tour length: {baseline_mean:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # t-test and update if better
        if self._paired_t_test(current_lengths, baseline_lengths):
            print("Current model is significantly better. Updating baseline model.")
            self.baseline_model = deepcopy(self.model).to(self.device)
            self.baseline_model.eval()
            self.baseline_updates += 1
        else:
            print("Current model is not significantly better than the baseline, no update")