"""Log-linear policy over the finite response set."""
import torch
import torch.nn.functional as F


def policy_logits(theta, psi, ref_logits):
    """logits_i(x) = ref_logits_i(x) + theta^T psi(x, y_i).

    theta: [d]; psi: [B, K, d]; ref_logits: [B, K]. Returns [B, K].
    """
    return ref_logits + torch.einsum("bkd,d->bk", psi, theta)


def policy_probs(theta, psi, ref_logits):
    return F.softmax(policy_logits(theta, psi, ref_logits), dim=-1)


def policy_log_probs(theta, psi, ref_logits):
    return F.log_softmax(policy_logits(theta, psi, ref_logits), dim=-1)


def policy_entropy(probs, eps=1e-12):
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)


def top1_prob(probs):
    return probs.max(dim=-1).values


def diagonal_mass(probs):
    return (probs * probs).sum(dim=-1)


def kl_to_ref(theta, psi, ref_logits, eps=1e-12):
    """E_x sum_i pi_theta(i|x) * (log pi_theta(i|x) - log pi_ref(i|x)).

    pi_theta = softmax(ref_logits + theta @ psi); pi_ref = softmax(ref_logits).
    Returns a [B] tensor; caller takes .mean() to reduce over prompts.
    """
    log_pi_th = policy_log_probs(theta, psi, ref_logits)
    log_pi_ref = F.log_softmax(ref_logits, dim=-1)
    pi_th = log_pi_th.exp()
    return (pi_th * (log_pi_th - log_pi_ref)).sum(dim=-1)


def project_l2(theta, radius):
    """Return a tensor with ||theta||_2 <= radius, differentiable."""
    norm = theta.norm(p=2)
    scale = torch.clamp(radius / norm.clamp_min(1e-12), max=1.0)
    return theta * scale


@torch.no_grad()
def project_l2_(param, radius):
    """In-place L2 projection on a parameter tensor."""
    n = param.data.norm(p=2).item()
    if n > radius:
        param.data.mul_(radius / n)
