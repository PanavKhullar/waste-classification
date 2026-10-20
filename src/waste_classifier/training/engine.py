"""
engine.py — Training and validation loop logic.

Responsibilities:
- train_one_epoch(): Forward pass, loss, backward, optimizer step.
- validate(): Forward pass with torch.no_grad(), compute metrics.
- Early stopping logic.
- Checkpoint saving (best model by validation metric).

Key concepts (will be explained when implemented):
- model.train() vs model.eval()
- loss.backward() and autograd
- optimizer.step() and zero_grad()
- torch.no_grad() / inference_mode for evaluation

Will be implemented in Phase 2.
"""

