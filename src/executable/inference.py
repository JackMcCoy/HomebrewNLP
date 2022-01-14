import typing

import torch

from src.dataclass import Context
from src.model import LinearAttention
from src.utils.setup import encode, decode, get_model


def complete_batch(ctx: Context, model: LinearAttention, str_prompt: str, prompt: torch.Tensor, temperature: float,
                   generated_tokens: int) -> typing.List[str]:
    batch, prompt_size = prompt.size()
    C, d = prompt.shape
    out = torch.zeros(C, ctx.model.features, device = ctx.model.device)
    out[:,ctx.model.features - d :] = prompt
    prompt = out
    for _ in range(prompt_size + 1, generated_tokens-1):
        tmp = model(prompt.type(torch.long))
        tmp = tmp[:, :, -1]
        tmp += torch.rand_like(tmp).clamp(min=1e-9).log().neg().log() * (-temperature)
        new_item = torch.argmax(tmp, -1).view(batch, -1)
        out = prompt = torch.cat([out[:,1:], new_item], -1)
        str_prompt = str_prompt + chr(int(out[0,-1]))
        if ctx.eval.cache:
            prompt = new_item
    model.reset_cache()
    return str_prompt


def complete(ctx: Context, model: LinearAttention, prompt: str, temperature: float, generated_tokens: int) -> str:
    return complete_batch(ctx, model, prompt, encode(prompt).to(dtype=torch.long, device=ctx.model.device).view(1, -1),
                          temperature, generated_tokens)


@torch.no_grad()
def inference_cli(ctx: Context, temperature: float, generated_tokens: int):
    mod = get_model(ctx, True).model
    mod.eval()
    while True:
        try:
            prompt = input("Prompt: ")
        except KeyboardInterrupt:
            break
        print(complete(ctx, mod, prompt, temperature, generated_tokens))
