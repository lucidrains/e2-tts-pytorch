import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    mel_spec = [item['mel_spec'].squeeze(0) for item in batch]
    mel_lengths = [item['mel_spec'].shape[-1] for item in batch]
    text = [item['text'] for item in batch]
    max_mel_length = max(mel_lengths)
    padded_audio = []
    for item in mel_spec:
        padding = (0, max_mel_length - item.size(-1))
        padded_item = torch.nn.functional.pad(item, padding, mode='constant', value=0)
        padded_audio.append(padded_item)
    audio = torch.stack(padded_audio)

    text_lengths = torch.LongTensor([len(item) for item in text])
    text = pad_sequence([torch.LongTensor(item) for item in text], batch_first=True)
    batch_dict = {
        'mel': mel_spec,
        'mel_lengths': mel_lengths,
        'text': text,
        'text_lengths': text_lengths,
    }
    return batch_dict