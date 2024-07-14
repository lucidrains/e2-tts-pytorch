import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    mel_specs = [item['mel_spec'].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    
    max_mel_length = mel_lengths.max().item()
    padded_mel_specs = []
    for spec in mel_specs:
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = torch.nn.functional.pad(spec, padding, mode='constant', value=0)
        padded_mel_specs.append(padded_spec)
    
    mel_specs = torch.stack(padded_mel_specs)

    text = [item['text'] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])
    batch_dict = {
        'mel': mel_specs,
        'mel_lengths': mel_lengths,
        'text': text,
        'text_lengths': text_lengths,
    }
    return batch_dict