import argparse
import torch


def list_of_strings(arg):
    return arg.split(',')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def group_data_by_person_id(embeddings, input_features, labels, person_ids):
    grouped_data = {}
    for emb, inp_feat, label, p_id in zip(embeddings, input_features, labels, person_ids):
        if p_id not in grouped_data:
            grouped_data[p_id] = {'embeddings': [], 'input_features': [], 'labels': []}
        grouped_data[p_id]['embeddings'].append(emb)
        grouped_data[p_id]['input_features'].append(inp_feat)
        grouped_data[p_id]['labels'].append(label)
        # grouped_data[p_id]['mel'].append(mel)
    # Convert lists to tensors
    for p_id, data in grouped_data.items():
        data['embeddings'] = torch.stack(data['embeddings'])
        data['input_features'] = torch.stack(data['input_features'])
        data['labels'] = torch.tensor(data['labels'])
        # data['mel'] = torch.tensor(data['mel'])
    return grouped_data

def compute_loss(output, labels):
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(output, labels)


def accumulate_gradients(model, gradient_accumulator, person_id):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name not in gradient_accumulator:
                gradient_accumulator[name] = []
            gradient_accumulator[name].append(param.grad.clone())




def standardize_embeddings(embeddings):
    """
    Standardize embeddings to have mean 0 and standard deviation 1 along the feature dimension.
    """
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True) + 1e-6  # Adding a small constant to prevent division by zero
    standardized_embeddings = (embeddings - mean) / std
    return standardized_embeddings



def process_in_slices(model, input_features, decoder_input_ids, slice_size):
    num_slices = (input_features.size(0) + slice_size - 1) // slice_size
    last_hidden_states_aggregated = []
    for i in range(num_slices):
        start_idx = i * slice_size
        end_idx = min((i + 1) * slice_size, input_features.size(0))
        input_features_slice = input_features[start_idx:end_idx]
        decoder_input_ids_slice = decoder_input_ids[start_idx:end_idx]
        with torch.no_grad():
            last_hidden_state_slice = model(input_features=input_features_slice,
                                            decoder_input_ids=decoder_input_ids_slice).last_hidden_state
        last_hidden_states_aggregated.append(last_hidden_state_slice)
    # Concatenate all slices to form the full batch again
    last_hidden_states = torch.cat(last_hidden_states_aggregated, dim=0)
    return last_hidden_states