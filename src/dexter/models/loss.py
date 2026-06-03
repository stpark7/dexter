import torch
from chamferdist import ChamferDistance
from torch import Tensor

from dexter.utils.shadowhand import contact_map_of_m_to_n


def get_hand_chamfer_loss(prediction, target, reduce: bool = True) -> dict[str, Tensor]:
    # chamfer loss between predict-hand point cloud and target-hand point cloud
    chamfer_distance = ChamferDistance()
    pred_hand_pc = prediction["surface_points"]
    target_hand_pc = target["surface_points"]
    chamfer_loss = chamfer_distance(
        pred_hand_pc, target_hand_pc, bidirectional=True, batch_reduction="mean" if reduce else None
    )
    return chamfer_loss


def get_cmap_loss(prediction, target, reduce: bool = True) -> dict[str, Tensor]:
    # chamfer loss between predict-hand point cloud and target-hand point cloud
    pred_hand_pc = prediction["surface_points"]
    target_hand_pc = target["surface_points"]
    pc = target["obj_pc"]
    pred_cmap = contact_map_of_m_to_n(pc, pred_hand_pc)
    gt_cmap = contact_map_of_m_to_n(pc, target_hand_pc)
    cmap_loss = torch.nn.functional.mse_loss(pred_cmap, gt_cmap, reduction="none")
    cmap_loss = cmap_loss.mean() if reduce else cmap_loss.mean(dim=-1)
    return cmap_loss


def get_obj_penetration_loss(
    prediction, target, training: bool = True, reduce: bool = True
) -> dict[str, Tensor]:
    batch_size = prediction["penetration_keypoints"].size(0)
    # signed squared distances from object_pc to hand, inside positive, outside negative
    distances = prediction["penetration"]
    # loss_pen
    if training:
        loss_pen = distances[distances > 0].sum() / batch_size
    else:
        loss_pen = distances.max(dim=-1).values
        loss_pen = loss_pen.mean() if reduce else loss_pen
    return loss_pen


def get_self_penetration_loss(
    prediction, target, training: bool = True, reduce: bool = True
) -> dict[str, Tensor]:
    batch_size = prediction["penetration_keypoints"].size(0)
    # loss_spen
    penetration_keypoints = prediction["penetration_keypoints"]
    dis_spen = (
        (penetration_keypoints.unsqueeze(1) - penetration_keypoints.unsqueeze(2) + 1e-13)
        .square()
        .sum(3)
        .sqrt()
    )
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    if training:
        loss_spen = dis_spen.sum() / batch_size
    else:
        loss_spen = dis_spen.reshape(batch_size, -1).max(dim=-1).values
        loss_spen = loss_spen.mean() if reduce else loss_spen
    return loss_spen


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    """
    Compute token-level accuracy for tokens specified by mask.

    Args:
        predicted_token_ids: [B, L] predicted token IDs
        ground_truth_token_ids: [B, L] ground truth token IDs
        mask: [B, L] boolean mask for tokens to evaluate

    Returns:
        Accuracy scalar (fraction of correct predictions among masked tokens)
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predicted_token_ids.device)
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_category_token_accuracies(
    action_tokenizer, predicted_token_ids, ground_truth_token_ids, overall_mask=None
):
    """
    Compute token accuracies grouped by token category (action, joint, position, overall).

    This function uses ground truth token types to categorize tokens, then measures
    if predictions match GT at those positions.

    Args:
        action_tokenizer: GraspTokenizerQwen3 instance with token type checking methods
        predicted_token_ids: [B, L] predicted token IDs
        ground_truth_token_ids: [B, L] ground truth token IDs
        overall_mask: [B, L] optional mask for valid tokens (e.g., non-padding).
                     If None, all tokens are considered valid.

    Returns:
        Dictionary with accuracies:
        {
            'overall': accuracy across all valid tokens,
            'action': accuracy for action bin tokens,
            'joint': accuracy for joint name tokens,
            'position': accuracy for position bin tokens
        }
    """
    # If no overall mask provided, evaluate all tokens
    if overall_mask is None:
        overall_mask = torch.ones_like(ground_truth_token_ids, dtype=torch.bool)

    # Create category masks based on ground truth token types
    action_mask = action_tokenizer.is_action_token(ground_truth_token_ids) & overall_mask
    joint_mask = action_tokenizer.is_joint_token(ground_truth_token_ids) & overall_mask
    position_mask = action_tokenizer.is_position_token(ground_truth_token_ids) & overall_mask

    # Compute accuracies for each category
    accuracies = {
        "overall": compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, overall_mask
        ),
        "action": compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, action_mask),
        "joint": compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, joint_mask),
        "position": compute_token_accuracy(
            predicted_token_ids, ground_truth_token_ids, position_mask
        ),
    }

    return accuracies


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids):
    """
    Compute L1 loss between predicted and ground truth actions.

    This function identifies action tokens from ground truth, decodes both predicted
    and ground truth action bins to continuous action values, and computes L1 loss.
    Samples without action tokens (due to position dropout or no-ECoT training) are
    excluded from the average.

    Args:
        action_tokenizer: GraspTokenizerQwen3 instance with action decoding methods
        predicted_token_ids: [B, L] predicted token IDs
        ground_truth_token_ids: [B, L] ground truth token IDs

    Returns:
        L1 loss averaged over samples with action tokens, or 0 if no actions found
    """
    device = predicted_token_ids.device

    # Find all action tokens in ground truth
    action_mask = action_tokenizer.is_action_token(ground_truth_token_ids)

    # Count action tokens per sample
    num_action_tokens = action_mask.sum(dim=-1)  # [B]

    # Identify samples that have action tokens
    valid_samples_mask = num_action_tokens > 0
    num_valid_samples = valid_samples_mask.sum().item()

    if num_valid_samples == 0:
        return torch.tensor(0.0, device=device)

    # Get the action dimension (should be consistent across valid samples)
    action_dim = num_action_tokens[valid_samples_mask][0].item()

    # Extract action tokens only from valid samples
    valid_pred_tokens = predicted_token_ids[valid_samples_mask]
    valid_true_tokens = ground_truth_token_ids[valid_samples_mask]
    valid_action_mask = action_mask[valid_samples_mask]

    pred_action_tokens = valid_pred_tokens[valid_action_mask].cpu().numpy()
    true_action_tokens = valid_true_tokens[valid_action_mask].cpu().numpy()

    # Reshape to [num_valid_samples, action_dim]
    pred_action_tokens = pred_action_tokens.reshape(num_valid_samples, action_dim)
    true_action_tokens = true_action_tokens.reshape(num_valid_samples, action_dim)

    # Decode to continuous actions
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(pred_action_tokens), device=device
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(true_action_tokens), device=device
    )

    # Compute L1 loss averaged over valid samples only
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss


def compute_positions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids):
    """
    Compute L1 loss between predicted and ground truth contact positions.

    This function identifies position tokens from ground truth, decodes both predicted
    and ground truth position bins to continuous 3D coordinates, and computes L1 loss.
    Samples without position tokens (due to position dropout or no-ECoT training) are
    excluded from the average.

    Args:
        action_tokenizer: GraspTokenizerQwen3 instance with position decoding methods
        predicted_token_ids: [B, L] predicted token IDs
        ground_truth_token_ids: [B, L] ground truth token IDs

    Returns:
        L1 loss averaged over samples with position tokens, or 0 if no positions found
    """
    device = predicted_token_ids.device

    # Find all position tokens in ground truth
    position_mask = action_tokenizer.is_position_token(ground_truth_token_ids)

    # Count position tokens per sample
    num_position_tokens = position_mask.sum(dim=-1)  # [B]

    # Identify samples that have position tokens
    valid_samples_mask = num_position_tokens > 0
    num_valid_samples = valid_samples_mask.sum().item()

    if num_valid_samples == 0:
        return torch.tensor(0.0, device=device)

    # Get the position token count per sample (should be consistent and divisible by 3)
    position_dim = num_position_tokens[valid_samples_mask][0].item()

    # Extract position tokens only from valid samples
    valid_pred_tokens = predicted_token_ids[valid_samples_mask]
    valid_true_tokens = ground_truth_token_ids[valid_samples_mask]
    valid_position_mask = position_mask[valid_samples_mask]

    pred_position_tokens = valid_pred_tokens[valid_position_mask].cpu().numpy()
    true_position_tokens = valid_true_tokens[valid_position_mask].cpu().numpy()

    # Reshape to [num_valid_samples * num_positions_per_sample, 3]
    # where num_positions_per_sample = position_dim // 3
    pred_position_tokens = pred_position_tokens.reshape(-1, 3)
    true_position_tokens = true_position_tokens.reshape(-1, 3)

    # Decode to continuous 3D positions
    pred_positions = torch.tensor(
        action_tokenizer.decode_token_ids_to_position(pred_position_tokens), device=device
    )
    true_positions = torch.tensor(
        action_tokenizer.decode_token_ids_to_position(true_position_tokens), device=device
    )

    # Compute L1 loss averaged over valid samples only
    l1_loss = torch.nn.functional.l1_loss(pred_positions, true_positions)
    return l1_loss
