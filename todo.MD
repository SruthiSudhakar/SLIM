Training a Minimal, Decision-Equivalent World Model
Background: what we have
A diffusion world model: takes (current image, candidate action) → generates a video of the robot executing that action.
A critic that ranks the generated videos to pick the best action. Concretely, our critic is a Qwen2.5-VL tournament: for 5 candidate videos we run all 10 pairwise comparisons, feeding two images plus the prompt "say which one shows more progress," and the VLM outputs ±1 per pair. The action with the most wins is chosen.
This pipeline works. The question is how to make the world model as small/cheap as possible without hurting decisions.
The core idea
We don't need an accurate video model — we need a decision-equivalent one. Some generated details matter (gripper pose to ~1mm); others don't (leaves in the background). We do not want to hand-code which is which.
The solution is to train the diffusion model with a single objective: the decision loss itself. Generate videos for K candidate actions, score them with the critic, and penalize the model whenever the critic's ranking disagrees with ground truth (real rollout outcomes, or our current full-size model as teacher).
Why this automatically learns what matters: the gradient reaching the diffusion weights must pass through the critic. The critic's sensitivity to pixels (its Jacobian, ∂V/∂x) is large on task-relevant content (gripper, object) and near-zero on irrelevant content (background leaves). Wherever that sensitivity is zero, no training signal flows — so the model gets zero pressure to render leaves well, and under capacity pressure (small network, weight decay), it reallocates those weights to gripper precision on its own. The critic's Jacobian is the importance mask, recomputed every step, never hand-coded.This connects to known literature: value-equivalence / value-aware model learning ("objective mismatch" in model-based RL), and diffusion reward-finetuning (DRaFT, AlignProp, DDPO).
The objective
Because our critic is pairwise, use a Bradley–Terry style pairwise loss. Define a differentiable pairwise score from Qwen's logits (not its sampled ±1 token, which is non-differentiable):
m(x_i, x_j) = logit("+1") − logit("−1") # If m large+positive, Qwen strongly believes video j shows more progress; large+negative, strongly video i.
m̃(i,j) = [m(x_i,x_j) − m(x_j,x_i)] / 2      # symmetrize to kill position bias

Loss over pairs of generated videos x̂:
L(θ) = E over pairs (i,j) [ −log σ( m̃(x̂_i, x̂_j) · y_ij ) ] # σ(m) = "probability that the pair-winner is j." y_ij ∈ {+1, −1} is the ground-truth answer for that pair.; −log σ(m · y_ij) is just binary cross-entropy. 
If you have a full ranking of the N actions (e.g., from real rollout outcomes, or the teacher model's tournament), then every pair has a correct answer. Action ranked 3rd genuinely beats action ranked 5th, and that comparison carries real training signal even though neither is the overall winner. All choose(N,2) pairs are labeled; sampling 2–4 random ones per step is just stochastic minibatching over them — over many steps you cover everything.
If you only know which of the N was best (a single winner label, nothing about the ordering of the losers), then only the N-1 pairs containing the winner have known labels. In that case, subsample from those N-1 and skip loser-vs-loser pairs entirely — you have no ground truth for them, so training on them would inject noise.
How the gradient gets through the diffusion sampler
The chain is: θ → denoising sampler → video frames → Qwen (frozen) → logits → loss. Options:
Backprop through the sampler (preferred): fix the noise, use a deterministic DDIM/ODE chain, differentiate end-to-end with gradient checkpointing — or truncate to the last few (low-noise) steps, which is where fine detail like gripper precision is formed anyway. Distilling to a few-step model first (consistency distillation, 2–8 steps) makes this cheap and is the biggest practical simplification.
OR we can Stay at 50 steps, truncated backprop: run the full 50-step sampler but only differentiate through the last k steps (say 5–10), treating everything earlier as a constant input. This is exactly what DRaFT does. Justifiable here because low-noise (late) steps are where fine, decision-relevant detail like gripper geometry gets formed. Cost: no gradient reaches the model's high-noise behavior.
REINFORCE fallback (DDPO-style): reward = tournament win count; no backprop through Qwen or the sampler. Noisier, but no VLM memory cost — worth having wired up as a baseline.
Non-negotiable training mechanics
Warm-start from the current world model. The ranking loss carries only a few bits per rollout; it can re-sculpt a good model but cannot train one from scratch.
Freeze Qwen. We only need gradients w.r.t. its input pixels. Use checkpointing/bf16; consider a small Qwen variant for training gradients, big one for eval.
Shared noise seeds across the K action rollouts so score comparisons are paired and low-variance.
Keep the pipeline tensor-to-tensor. Resize/normalize must be differentiable — no image-file round trips or uint8 quantization inside the graph.
Choose critic input frames deliberately. Only frames Qwen sees get direct supervision. If mid-trajectory progress matters, feed 2–4 keyframes per video, not just the last frame.
Getting the "minimal" model
The loss does capacity reallocation by itself. For explicit shrinking: apply weight decay / pruning / quantization, reduce width, and cut denoising steps (compress high-noise steps aggressively; preserve late low-noise steps). Compress until the real metric breaks:
Stopping criterion = tournament-winner agreement on held-out decision problems (does the small model's best action match ground truth), and ultimately real task success. Not FVD or visual quality — those are diagnostics only.
The main risk: critic hacking (take this seriously)
Pixel gradients through a frozen VLM are literally the recipe for adversarial examples against it. Our training loop is, by construction, an adversarial-example generator pointed at Qwen. Guardrails:
Anchor to the teacher: a small KL / score-matching term to the pretrained world model, keeping videos on the natural manifold where Qwen's judgments mean something. With a VLM critic, include this from day one.
Held-out judge the gradient never touched: a different VLM family and/or real-outcome data. A growing gap between training-Qwen scores and the held-out judge = hacking alarm.
Refresh the critic periodically on the student's own samples with real outcome labels, if possible.
Look at the videos. Background turning to mush is expected and fine. Subtle texture noise in task-relevant regions that still scores well is the hacking signature.
First experiment (before building anything big)
Feed Qwen a real pair, compute ∂(logit-margin)/∂pixels, and inspect the saliency map. Verify (a) sensitivity concentrates on gripper/object, not background, and (b) it isn't dominated by high-frequency junk everywhere (which predicts fast adversarial hacking).
Check the logit margin correlates with ground-truth preferences on a held-out set. This is the signal the whole method amplifies — measure its quality first.
If both checks pass: distill to few-step → attach the pairwise loss → verify winner-agreement holds → compress until it breaks.
One conceptual limit worth knowing
Taken to its logical end, the minimal decision-equivalent model stops emitting video at all and just predicts critic features from (image, action) — a latent (TD-MPC-style) model. We keep video for interpretability, critic generality, and reuse; that tells you where the compression floor is.
