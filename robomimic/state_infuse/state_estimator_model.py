import torch
import torch.nn as nn


class CompletionEstimationWithStateDescription(nn.Module):
    def __init__(self, task_str_emb_size, hidden_mapping_size, transformer_encoding_size, state_descp_size,
                 attention_heads=1,
                 dropout_rate=0.1):
        super(CompletionEstimationWithStateDescription, self).__init__()
        self.hidden_mapping_size = hidden_mapping_size

        # Linear layer to map the completion rate sto a vector
        # self.map_completion_to_vector = nn.Linear(1, hidden_mapping_size)

        # Linear layer to map the state description to a vector
        self.map_state_descp_to_vector = nn.Linear(state_descp_size, hidden_mapping_size)

        # Linear layer to merge task embedding and state description embedding
        self.merge_info = nn.Linear(hidden_mapping_size + task_str_emb_size, hidden_mapping_size)

        # Additional linear layers for added depth
        self.hidden_layer_1 = nn.Linear(hidden_mapping_size, hidden_mapping_size)
        self.hidden_layer_2 = nn.Linear(hidden_mapping_size, hidden_mapping_size)

        # Batch normalization layers
        self.batch_norm_1 = nn.BatchNorm1d(hidden_mapping_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_mapping_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Self-attention mechanism to enhance information fusion
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_mapping_size, num_heads=attention_heads,
                                                    batch_first=True)

        # Nonlinear activation function
        self.sigmoid = nn.Sigmoid()
        self.nonlinear = nn.ReLU(inplace=False)

        self.action_size = 12
        self.step_size = 10
        self.hidden_to_ac = nn.Linear(hidden_mapping_size, self.action_size * self.step_size)

        # Final mapping to the transformer encoding size
        self.output_size = transformer_encoding_size
        self.map_to_encoding_input_size = nn.Linear(self.action_size * self.step_size, self.output_size * self.step_size)

    def forward(self, task_str_emb, state_descp):
        # Map the completion rate to a vector
        # completion_emb = self.map_completion_to_vector(completion_rate)

        # Map the state description to a vector
        state_descp_emb = self.map_state_descp_to_vector(state_descp)
        # task_str_emb = self.linear_task(task_str_emb)

        # Concatenate task string embedding with completion embedding and state description embedding
        concatenated = torch.cat((task_str_emb, state_descp_emb), dim=1)

        # Pass through the merging layer
        merged = self.merge_info(concatenated)
        merged = self.batch_norm_1(merged)
        merged = self.nonlinear(merged)

        # Additional hidden layers
        hidden_output = self.hidden_layer_1(merged)
        hidden_output = self.batch_norm_2(hidden_output)
        hidden_output = self.nonlinear(hidden_output)

        hidden_output = self.hidden_layer_2(hidden_output)
        hidden_output = self.nonlinear(hidden_output)

        # Apply dropout
        hidden_output = self.dropout(hidden_output)

        # Reshape for self-attention
        hidden_output = hidden_output.contiguous().unsqueeze(1)  # Add a sequence dimension for attention
        attention_output, _ = self.self_attention(hidden_output, hidden_output, hidden_output)
        attention_output = attention_output.squeeze(1)  # Remove the sequence dimension

        # Final transformation to match the transformer encoding size
        # output = self.map_to_encoding_input_size(attention_output)
        attention_output = attention_output.contiguous()
        _output = self.sigmoid(self.hidden_to_ac(attention_output))
        _output = self.map_to_encoding_input_size(_output.clone().contiguous())

        return _output


class CompletionEstimationModelComplicationVersion(nn.Module):
    def __init__(self, task_str_emb_size, hidden_mapping_size, transformer_encoding_size, attention_heads=4,
                 dropout_rate=0.1):
        super(CompletionEstimationModelComplicationVersion, self).__init__()
        self.hidden_mapping_size = hidden_mapping_size

        # Linear layer to map the completion rate to a vector
        self.map_completion_to_vector = nn.Linear(1, hidden_mapping_size)

        # Linear layer to merge task embedding with completion rate embedding
        self.merge_info = nn.Linear(hidden_mapping_size + task_str_emb_size, hidden_mapping_size)

        # Additional linear layers for added depth
        self.hidden_layer_1 = nn.Linear(hidden_mapping_size, hidden_mapping_size)
        self.hidden_layer_2 = nn.Linear(hidden_mapping_size, hidden_mapping_size)

        # Batch normalization layers
        self.batch_norm_1 = nn.BatchNorm1d(hidden_mapping_size)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_mapping_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Self-attention mechanism to enhance information fusion
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_mapping_size, num_heads=attention_heads,
                                                    batch_first=True)

        # Nonlinear activation function
        self.nonlinear = nn.ReLU()

        # Final mapping to the transformer encoding size
        self.map_to_encoding_input_size = nn.Linear(hidden_mapping_size, transformer_encoding_size)

    def forward(self, completion_rate, task_str_emb):
        # Map the completion rate to a vector
        completion_emb = self.map_completion_to_vector(completion_rate)

        # Concatenate task string embedding with completion embedding
        concatenated = torch.cat((completion_emb, task_str_emb), dim=1)

        # Pass through the merging layer
        merged = self.merge_info(concatenated)
        merged = self.batch_norm_1(merged)
        merged = self.nonlinear(merged)

        # Additional hidden layers
        hidden_output = self.hidden_layer_1(merged)
        hidden_output = self.batch_norm_2(hidden_output)
        hidden_output = self.nonlinear(hidden_output)

        hidden_output = self.hidden_layer_2(hidden_output)
        hidden_output = self.nonlinear(hidden_output)

        # Apply dropout
        hidden_output = self.dropout(hidden_output)

        # Reshape for self-attention
        hidden_output = hidden_output.unsqueeze(1)  # Add a sequence dimension for attention
        attention_output, _ = self.self_attention(hidden_output, hidden_output, hidden_output)
        attention_output = attention_output.squeeze(1)  # Remove the sequence dimension

        # Final transformation to match the transformer encoding size
        output = self.map_to_encoding_input_size(attention_output)
        return output

class CompletionTaskEmbeddingModel(nn.Module):
    def __init__(self, task_str_emb_size, hidden_mapping_size, transformer_encoding_size):
        super(CompletionTaskEmbeddingModel, self).__init__()
        self.hidden_mapping_size = hidden_mapping_size
        self.map_completion_to_vector = nn.Linear(1, hidden_mapping_size)  # Maps scalar 'p' to dimension d
        # self.map_s_to_scalar = nn.Linear(input_dim_s, 1)  # Maps sentence embedding 's' to scalar
        self.merge_info = nn.Linear(hidden_mapping_size + task_str_emb_size, hidden_mapping_size)  # Concatenates 'p' and scalar s, then maps to dimension d
        self.nonlinear = nn.ReLU()  # Nonlinear activation function
        self.map_to_encoding_input_size = nn.Linear(hidden_mapping_size, transformer_encoding_size)  # Maps to dimension V

    def forward(self, completion_rate, task_str_emb):
        # Map sentence embedding s to a scalar
        completion_emb = self.map_completion_to_vector(completion_rate)

        # s_scalar = self.map_s_to_scalar(task_str_emb)
        # Concatenate scalar s and scalar p
        concatenated = torch.cat((completion_emb, task_str_emb), dim=1)
        # Map the concatenated result to dimension d
        mapped_to_d = self.merge_info(concatenated)
        # Apply nonlinear function
        nonlinear_output = self.nonlinear(mapped_to_d)
        # Map to dimension V
        output = self.map_to_encoding_input_size(nonlinear_output)
        return output


if __name__ == '__main__':

    # Example usage:
    input_dim_s = 784  # Dimension of the sentence embedding 's'
    d = 128  # Dimension for the intermediate mapping
    V = 512  # Final output dimension

    model = CompletionEstimationWithStateDescription(input_dim_s, d, V, state_descp_size=1536)

    # Example input data
    batch_size = 16
    time_steps = 10
    p = torch.randn(batch_size, 1)  # Example scalar 'p' for each time step
    s = torch.randn(batch_size, input_dim_s)  # Example sentence embeddings 's'
    sd = torch.randn(batch_size, 1536)  # Example state descriptions 'sd'

    print(p.size())
    print(s.size())
    # Forward pass
    output = model(p, s, sd)
    print(output.shape)  # Output shape: [batch_size, time_steps, V]