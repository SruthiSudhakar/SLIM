"""
Change C — fix the temporal cross-attention degeneracy in diffusers'
`TransformerSpatioTemporalModel`.

Stock `forward` builds the temporal branch's key/value by taking ONLY frame 0's action token
(`[:, 0]`) and broadcasting it to every frame, so the temporal cross-attention is effectively
blind to the action trajectory (see the base-model architecture analysis). This patch instead
feeds ALL F per-frame action tokens, so each video token can attend over the whole action
trajectory in the temporal branch (trajectory-level motion reasoning).

Mechanics: only the `time_context` construction changes. NO new parameters — it reuses the
pretrained temporal `attn2` with a longer KV sequence (length 1 -> F). Best paired with Change A
(temporal action encoder), which gives the F tokens positional/trajectory structure so the
temporal attention can align frame t <-> action t±k.

Deployment: monkeypatch `TransformerSpatioTemporalModel.forward` (one world model per process, so
a global patch is safe and simple). IMPORTANT CAVEATS vs A/B:
  * Adds no weights -> NOT auto-detectable from a checkpoint -> the flag MUST be set explicitly at
    BOTH train and eval (env USE_TEMPORAL_ACTION_COND=1 / config use_temporal_action_cond).
  * NOT a zero-init no-op -> it changes the temporal branch's input distribution at step 0
    (frame-0 -> full-sequence), so expect a transient when warm-starting.
"""
import torch
from diffusers.models.transformers.transformer_temporal import (
    TransformerSpatioTemporalModel,
    TransformerTemporalModelOutput,
)

# handle to the original, in case we ever want to restore it
_STOCK_FORWARD = TransformerSpatioTemporalModel.forward


def _forward_full_temporal_context(
    self,
    hidden_states,
    encoder_hidden_states=None,
    image_only_indicator=None,
    return_dict=True,
):
    # 1. Input
    batch_frames, _, height, width = hidden_states.shape
    num_frames = image_only_indicator.shape[-1]
    batch_size = batch_frames // num_frames

    # ---- Change C: keep ALL F per-frame action tokens for the temporal cross-attention ----
    # (stock code did `[:, 0]` here, keeping only frame 0 and broadcasting it to every frame).
    # encoder_hidden_states: [B*F, S, C] -> [B, F, S, C] -> [B, F*S, C]; broadcast over spatial only.
    C = encoder_hidden_states.shape[-1]
    tc = encoder_hidden_states.reshape(batch_size, num_frames, -1, C).reshape(batch_size, -1, C)  # [B, F*S, C]
    time_context = tc[:, None].broadcast_to(batch_size, height * width, tc.shape[1], C)
    time_context = time_context.reshape(batch_size * height * width, tc.shape[1], C)  # [B*HW, F*S, C]

    residual = hidden_states

    hidden_states = self.norm(hidden_states)
    inner_dim = hidden_states.shape[1]
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
    hidden_states = self.proj_in(hidden_states)

    num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
    num_frames_emb = num_frames_emb.repeat(batch_size, 1)
    num_frames_emb = num_frames_emb.reshape(-1)
    t_emb = self.time_proj(num_frames_emb)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_pos_embed(t_emb)
    emb = emb[:, None, :]

    # 2. Blocks
    for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, None, encoder_hidden_states, None
            )
        else:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states_mix = hidden_states
        hidden_states_mix = hidden_states_mix + emb

        hidden_states_mix = temporal_block(
            hidden_states_mix,
            num_frames=num_frames,
            encoder_hidden_states=time_context,
        )
        hidden_states = self.time_mixer(
            x_spatial=hidden_states,
            x_temporal=hidden_states_mix,
            image_only_indicator=image_only_indicator,
        )

    # 3. Output
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

    output = hidden_states + residual

    if not return_dict:
        return (output,)

    return TransformerTemporalModelOutput(sample=output)


_PATCHED = False


def apply_full_temporal_action_context():
    """Monkeypatch the temporal transformer to feed ALL per-frame action tokens to the temporal
    cross-attention (Change C). Idempotent; global for the process."""
    global _PATCHED
    if not _PATCHED:
        TransformerSpatioTemporalModel.forward = _forward_full_temporal_context
        _PATCHED = True
        print("[Change C] temporal cross-attention now receives ALL per-frame action tokens "
              "(patched TransformerSpatioTemporalModel.forward). NOTE: not checkpoint-detectable "
              "-> ensure USE_TEMPORAL_ACTION_COND=1 is set at eval too.")
