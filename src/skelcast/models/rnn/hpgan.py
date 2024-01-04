import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest


class HPGANAttention(nn.Module):
    def __init__(self, hidden_dim, attention_len=None) -> None:
        super().__init__()
        if attention_len is None:
            attention_len = hidden_dim

        self.linear_v = nn.Linear(hidden_dim, attention_len, bias=True)
        self.linear_u = nn.Linear(attention_len, 1, bias=False)

    def forward(self, inputs, activation=None):
        v = self.linear_v(inputs)
        if activation is not None:
            v = activation(v)

        vu = self.linear_u(v).squeeze(-1)
        alpha = F.softmax(vu, dim=-1)

        return torch.sum(inputs * alpha.unsqueeze(-1), dim=1)


def generate_skeletons_with_prob(g, d_real_prob, d_fake_prob, data_preprocessing, d_inputs, g_inputs, g_z):
    '''
    Generate multiple future skeleton poses for each `z` in `g_z` using PyTorch models.
    '''
    def generate(input_data, input_past_data, z_data_p, batch_index=0):
        skeleton_data = []
        d_is_sequence_probs = []

        # Move input data to the same device as the model
        input_data_tensor = torch.tensor(input_data).to(d_real_prob.device)
        input_past_data_tensor = torch.tensor(input_past_data).to(g.device)

        # Get real data probability
        with torch.no_grad():
            prob = d_real_prob(input_data_tensor).squeeze().cpu().numpy()
        
        skeleton_data.append(data_preprocessing.unnormalize(input_data[batch_index, :, :, :]))
        d_is_sequence_probs.append(prob[batch_index])

        for z_value_p in z_data_p:
            z_tensor = torch.tensor(z_value_p).to(g.device)
            
            with torch.no_grad():
                # Generate prediction and fake data probability
                pred = g(input_past_data_tensor, z_tensor)
                inout_pred = torch.cat((input_past_data_tensor, pred), dim=1)
                fake_prob = d_fake_prob(inout_pred).squeeze().cpu().numpy()

            skeleton_data.append(data_preprocessing.unnormalize(inout_pred[batch_index, :, :, :].numpy()))
            d_is_sequence_probs.append(fake_prob[batch_index])
        
        return skeleton_data, d_is_sequence_probs
    
    return generate


class RNNDiscriminator(nn.Module):
    def __init__(self, inputs_depth, sequence_length, use_attention=False, use_residual=False, 
                 cell_type='gru', output_category_dims=None):
        super(RNNDiscriminator, self).__init__()
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.num_neurons = 1024
        self.num_layers = 2
        self.sequence_length = sequence_length
        self.output_dims = 1
        self.output_category_dims = output_category_dims

        # Define the RNN layer
        if cell_type == 'gru':
            self.rnn = nn.GRU(input_size=inputs_depth, hidden_size=self.num_neurons, 
                              num_layers=self.num_layers, batch_first=True)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size=inputs_depth, hidden_size=self.num_neurons, 
                               num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError("Unsupported cell type")

        # Attention layer
        if self.use_attention:
            self.attention = HPGANAttention(self.num_neurons)

        # Fully connected layers
        self.fc1 = nn.Linear(self.num_neurons, self.num_neurons)
        self.output_layer = nn.Linear(self.num_neurons, self.output_dims)
        if self.output_category_dims is not None:
            self.output_category_layer = nn.Linear(self.num_neurons, self.output_category_dims)

    def forward(self, inputs):
        # RNN forward pass
        outputs, _ = self.rnn(inputs)

        if self.use_attention:
            last = self.attention(outputs)
        else:
            last = outputs[:, -1, :]

        base = F.relu(self.fc1(last))

        output = self.output_layer(base)
        prob = torch.sigmoid(output)

        if self.output_category_dims is not None:
            output_category = self.output_category_layer(base)
            return output, output_category, prob
        else:
            return output, prob


class ResidualBlock(nn.Module):
    def __init__(self, num_neurons, activation_fn):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(num_neurons, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_neurons)
        self.activation_fn = activation_fn

    def forward(self, x):
        residual = x
        out = self.activation_fn(self.fc1(x))
        out = self.fc2(out)
        out += residual
        return self.activation_fn(out)


class NNDiscriminator(nn.Module):
    def __init__(self, inputs_depth, num_layers=3):
        super(NNDiscriminator, self).__init__()
        self.num_neurons = 512
        self.num_layers = num_layers
        self.output_dims = 1
        self.stddev = 0.001

        # Create a sequential model for the fully connected layers
        fc_layers = []
        for _ in range(num_layers):
            fc_layers.append(nn.Linear(inputs_depth, self.num_neurons))
            fc_layers.append(nn.ReLU())
            inputs_depth = self.num_neurons  # Set input depth for next layer

        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(self.num_neurons, self.output_dims)

    def forward(self, inputs):
        # Reshape and flatten inputs if necessary
        inputs = inputs.view(inputs.size(0), -1)

        net = self.fc_layers(inputs)
        output = self.output_layer(net)
        prob = torch.sigmoid(output)

        return output, prob
    

class NNResidualDiscriminator(nn.Module):
    def __init__(self, inputs_depth, num_residual_blocks=3, activation_fn=F.relu):
        super(NNResidualDiscriminator, self).__init__()
        self.num_neurons = 512
        self.num_residual_blocks = num_residual_blocks
        self.activation_fn = activation_fn

        # Initial fully connected layer
        self.fc1 = nn.Linear(inputs_depth, self.num_neurons)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(self.num_neurons, self.activation_fn) for _ in range(num_residual_blocks)]
        )

        # Output layer
        self.fc2 = nn.Linear(self.num_neurons, 1)

    def forward(self, inputs):
        # Reshape and flatten inputs if necessary
        inputs = inputs.view(inputs.size(0), -1)

        net = self.activation_fn(self.fc1(inputs))

        # Pass through residual blocks
        net = self.residual_blocks(net)

        # Final output layer
        output = self.fc2(net)
        prob = torch.sigmoid(output)

        return output, prob


class RNNGenerator(nn.Module):
    def __init__(self, inputs_depth, batch_size):
        super(RNNGenerator, self).__init__()
        self.batch_size = batch_size
        self.inputs_depth = inputs_depth
        self.num_neurons = 256
        self.num_layers = 2
        self.stddev = 0.001

        # LSTM layer
        self.rnn = nn.LSTM(input_size=inputs_depth, hidden_size=self.num_neurons, 
                           num_layers=self.num_layers, batch_first=True)

        # Output layer
        self.output_layer = nn.Linear(self.num_neurons, inputs_depth)

    def forward(self, inputs):
        # Reshape and flatten inputs if necessary
        inputs = inputs.view(inputs.size(0), -1, self.inputs_depth)

        # LSTM forward pass
        outputs, _ = self.rnn(inputs)

        # Getting the last output
        last_output = outputs[:, -1, :]

        # Passing through the output layer
        output = self.output_layer(last_output)

        # Reshaping to match the desired output shape
        output = output.view(self.batch_size, 1, -1)

        return torch.tanh(output)


class NNGenerator(nn.Module):
    def __init__(self, inputs_depth, batch_size):
        super(NNGenerator, self).__init__()
        self.batch_size = batch_size
        self.inputs_depth = inputs_depth
        self.num_neurons = 1024
        self.num_layers = 3
        self.stddev = 0.001

        # Sequential model for the fully connected layers
        self.fc_layers = nn.Sequential()
        for _ in range(self.num_layers):
            self.fc_layers.append(nn.Linear(inputs_depth, self.num_neurons))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(p=0.5))
            inputs_depth = self.num_neurons  # Set input depth for next layer

        # Output layer
        self.output_layer = nn.Linear(self.num_neurons, inputs_depth)

    def forward(self, inputs):
        # Reshape and flatten inputs if necessary
        inputs = inputs.view(inputs.size(0), -1)

        net = self.fc_layers(inputs)
        output = self.output_layer(net)

        # Reshaping to match the desired output shape
        output = output.view(self.batch_size, 1, -1)

        return torch.tanh(output)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cell_type='gru', use_residual=False):
        super(EncoderRNN, self).__init__()
        self.cell_type = cell_type
        self.use_residual = use_residual

        if cell_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Unsupported cell type")

    def forward(self, inputs):
        outputs, hidden = self.rnn(inputs)
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, cell_type='gru'):
        super(DecoderRNN, self).__init__()
        self.cell_type = cell_type

        if cell_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Unsupported cell type")

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        outputs, hidden = self.rnn(inputs, hidden)
        outputs = self.out(outputs)
        return outputs, hidden


class SequenceToSequenceGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_sequence_length, output_sequence_length, num_layers=2, cell_type='gru'):
        super(SequenceToSequenceGenerator, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, cell_type)
        self.decoder = DecoderRNN(hidden_size, hidden_size, output_size, num_layers, cell_type)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

    def forward(self, inputs, z=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)

        # Prepare initial input for decoder
        decoder_input = torch.zeros(inputs.shape[0], 1, inputs.shape[2], device=inputs.device)
        decoder_hidden = encoder_hidden
        outputs = []

        for _ in range(self.output_sequence_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs.append(decoder_output)
            decoder_input = decoder_output

        outputs = torch.cat(outputs, dim=1)
        return outputs

@pytest.fixture
def setup_discriminator():
    # Example setup parameters
    inputs_depth = 128
    sequence_length = 10
    batch_size = 5
    use_attention = True
    output_category_dims = 3

    # Create an instance of the RNNDiscriminator class
    discriminator = RNNDiscriminator(inputs_depth, sequence_length, use_attention, 
                                     output_category_dims=output_category_dims)

    # Create dummy input data
    dummy_inputs = torch.randn(batch_size, sequence_length, inputs_depth)

    return discriminator, dummy_inputs


def test_rnndiscriminator_output_shape(setup_discriminator):
    discriminator, dummy_inputs = setup_discriminator

    # Forward pass
    outputs = discriminator(dummy_inputs)

    # Check output shapes
    assert len(outputs) == 3, "Expected 3 outputs (output, output_category, prob)"
    assert outputs[0].shape == (dummy_inputs.shape[0], discriminator.output_dims), \
        f"Output shape is incorrect: {outputs[0].shape}"
    assert outputs[1].shape == (dummy_inputs.shape[0], discriminator.output_category_dims), \
        f"Output category shape is incorrect: {outputs[1].shape}"
    assert outputs[2].shape == (dummy_inputs.shape[0],), \
        f"Probability output shape is incorrect: {outputs[2].shape}"


def test_rnndiscriminator_output_type(setup_discriminator):
    discriminator, dummy_inputs = setup_discriminator

    # Forward pass
    outputs = discriminator(dummy_inputs)

    # Check output types
    assert isinstance(outputs[0], torch.Tensor), "Output is not a torch.Tensor"
    assert isinstance(outputs[1], torch.Tensor), "Output category is not a torch.Tensor"
    assert isinstance(outputs[2], torch.Tensor), "Probability output is not a torch.Tensor"


def test_attention():
    # TODO: To be included in tests
    batch_size = 10
    seq_length = 20
    num_neurons = 30
    attention_len = 15

    # Create an instance of the Attention class
    attention_layer = HPGANAttention(num_neurons, attention_len)

    # Create dummy input data
    dummy_inputs = torch.randn(batch_size, seq_length, num_neurons)

    # Get the output from the attention layer
    output = attention_layer(dummy_inputs)

    # Assert the output shape
    assert output.shape == (batch_size, num_neurons), f"Output shape is incorrect: {output.shape}"

    # Assert the output type
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor"


if __name__ == '__main__':
    test_attention()