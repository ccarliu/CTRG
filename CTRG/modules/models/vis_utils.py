import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 假设你有一个包含所有attention map的列表
# self_attention_maps 和 cross_attention_maps
# self_attention_maps: List of shape [1, 12, 380, 380]
# cross_attention_maps: List of shape [1, 12, 380, 110]

def compute_last_token_attention_proportion(self_attention_maps, cross_attention_maps):
    num_layers = len(self_attention_maps)
    seq_len = self_attention_maps[0].shape[2]
    img_feat_len = cross_attention_maps[0].shape[3]
    
    # Initialize rollout attention as identity matrix for self attention
    rollout_self_attention = np.eye(seq_len)
    
    # Initialize cross attention accumulation
    accumulated_cross_attention = np.zeros((seq_len, img_feat_len))
    
    for layer_idx in range(num_layers):
        # Fuse self attention maps by averaging over the heads
        self_attention_map = np.mean(self_attention_maps[layer_idx][0], axis=0)
        
        # Add identity matrix to self attention map to account for residual connections
        self_attention_map += np.eye(seq_len)
        
        # Normalize the self attention map
        self_attention_map /= self_attention_map.sum(axis=-1, keepdims=True)
        
        # Update rollout self attention
        rollout_self_attention = np.matmul(rollout_self_attention, self_attention_map)
        
        # Fuse cross attention maps by averaging over the heads
        cross_attention_map = np.mean(cross_attention_maps[layer_idx][0], axis=0)
        
        # Accumulate cross attention
        accumulated_cross_attention += np.matmul(rollout_self_attention, cross_attention_map)
    
    # Extract the attention weights for the last token
    # print(rollout_self_attention.shape)
    # print(accumulated_cross_attention.shape)
    last_token_self_attention = rollout_self_attention[-1, :]
    last_token_cross_attention = accumulated_cross_attention[-1, :]
    
    # Calculate the proportion of attention for the last token
    total_attention = last_token_self_attention.sum() + last_token_cross_attention.sum()
    self_attention_proportion = last_token_self_attention.sum() / total_attention
    cross_attention_proportion = last_token_cross_attention.sum() / total_attention
    
    return self_attention_proportion, last_token_cross_attention.sum()

def compute_combined_attention(self_attention_maps, cross_attention_maps):
    num_layers = len(self_attention_maps)
    num_heads = self_attention_maps[0].shape[1]
    seq_len = self_attention_maps[0].shape[2]
    img_feat_len = cross_attention_maps[0].shape[3]
    
    # Initialize rollout attention as identity matrix for self attention
    rollout_self_attention = np.eye(seq_len)
    
    # Initialize cross attention accumulation
    accumulated_cross_attention = np.zeros((seq_len, img_feat_len))
    
    for layer_idx in range(num_layers):
        # Fuse self attention maps by averaging over the heads
        self_attention_map = np.mean(self_attention_maps[layer_idx][0], axis=0)
        
        # Add identity matrix to self attention map to account for residual connections
        self_attention_map += np.eye(seq_len)
        
        # Normalize the self attention map
        self_attention_map /= self_attention_map.sum(axis=-1, keepdims=True)
        
        # Update rollout self attention
        rollout_self_attention = np.matmul(rollout_self_attention, self_attention_map)
        
        # Fuse cross attention maps by averaging over the heads
        cross_attention_map = np.mean(cross_attention_maps[layer_idx][0], axis=0)
        
        # Accumulate cross attention
        accumulated_cross_attention += np.matmul(rollout_self_attention, cross_attention_map)
    
    return rollout_self_attention, accumulated_cross_attention

def save_attention_maps(rollout_self_attention, accumulated_cross_attention, save_dir):

    rollout_self_attention = (rollout_self_attention - rollout_self_attention.min()) / (rollout_self_attention.max() - rollout_self_attention.min())
    # Save self attention map
    plt.figure(figsize=(15, 10))
    sns.heatmap(rollout_self_attention, cmap='viridis')
    plt.title('Rollout Self Attention Map')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/rollout_self_attention_map.png")
    plt.close()
    
    # Save cross attention map
    plt.figure(figsize=(15, 10))
    sns.heatmap(accumulated_cross_attention, cmap='viridis')
    plt.title('Accumulated Cross Attention Map')
    plt.xlabel('Image Features')
    plt.ylabel('Tokens')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accumulated_cross_attention_map.png")
    plt.close()
    
    print(f"Saved attention maps to {save_dir}")

def compute_ema(data, alpha=0.3):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def save_average_cross_attention(accumulated_cross_attention, save_path):
    # Compute the average cross attention for each output token
    average_cross_attention = np.max(accumulated_cross_attention, axis=1)
    
    # Compute the EMA trend line
    ema_trend = compute_ema(average_cross_attention, alpha=0.05)
    
    # Plot the average cross attention and the EMA trend line
    plt.figure(figsize=(15, 5))
    plt.plot(average_cross_attention, marker='o', label='Average Attention')
    plt.plot(ema_trend, color='red', linewidth=2, label='EMA Trend')
    plt.title('Average Cross Attention with EMA Trend')
    plt.xlabel('Image Features')
    plt.ylabel('Average Attention')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved average cross attention plot with trend to {save_path}")

if __name__ == "__main__":
    # 计算combined attention maps
    rollout_self_attention, accumulated_cross_attention = compute_combined_attention(self_attention_maps, cross_attention_maps)

    # 保存attention maps
    save_attention_maps(rollout_self_attention, accumulated_cross_attention, save_dir='./attention_maps')