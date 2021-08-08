import torch


def _compute_output_mask(gen_seq, eos_token):
    # to be used when the decoder conditions on its own output
    eos_idx = gen_seq.eq(eos_token).to(torch.int64)
    value_range = torch.arange(gen_seq.shape[0], 0, -1)
    output_multiplied = (eos_idx.transpose(0, 2) * value_range).transpose(0, 2)
    first_eos_idx = torch.argmax(output_multiplied, 0, keepdim=True).view(-1)

    print(first_eos_idx)

    gen_seq_mask = torch.ones(gen_seq.shape[0], gen_seq.shape[1], gen_seq.shape[2])
    policy_mask = torch.ones(gen_seq.shape[0] - 1, gen_seq.shape[1], gen_seq.shape[2])

    for row in range(0, gen_seq.shape[1]):
        print(first_eos_idx[row])
        gen_seq_mask[first_eos_idx[row] + 1:, row] = 0
        policy_mask[first_eos_idx[row]:, row] = 0

    return gen_seq_mask.to(torch.bool), policy_mask.to(torch.bool)


input_tensor = torch.tensor([[[1],[4],[7],[10],[13]],[[2],[5],[8],[0],[14]],[[3],[6],[9],[12],[0]]])

print(input_tensor)
print(input_tensor.shape)

seq_mask, pol_mask = _compute_output_mask(input_tensor, 0)

print(seq_mask)
print(seq_mask.shape)