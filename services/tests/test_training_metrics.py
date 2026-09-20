"""ROC-AUC and loss contracts, including a pairwise tie oracle (#17)."""
import pytest
import torch
from test_training_rpc_validation import loaded_service, invoke, pb
from training.server import _auc, _bce_link_loss


@pytest.mark.parametrize("positive,negative,expected", [
    ([0.], [0.], .5), ([0., 0.], [0., 0., 0.], .5),
    ([2., 3.], [0., 1.], 1.), ([0., 1.], [2., 3.], 0.),
    ([0., 1.], [0., 1.], .5), ([1., 2.], [0., 1.], .875),
])
def test_auc_known_results(positive, negative, expected):
    assert _auc(torch.tensor(positive), torch.tensor(negative)) == pytest.approx(expected)


def test_auc_matches_pairwise_oracle_and_is_order_invariant():
    generator = torch.Generator().manual_seed(29)
    for _ in range(30):
        positive = torch.randint(-2, 3, (7,), generator=generator).float()
        negative = torch.randint(-2, 3, (11,), generator=generator).float()
        pairs = positive[:, None] - negative
        expected = ((pairs > 0).float() + .5 * (pairs == 0).float()).mean().item()
        assert _auc(positive, negative) == pytest.approx(expected)
        assert _auc(positive.flip(0), negative.flip(0)) == pytest.approx(expected)


@pytest.mark.parametrize("function", [_auc, _bce_link_loss])
@pytest.mark.parametrize("positive,negative", [([], []), ([0.], []), ([], [0.]),
    ([float('nan')], [0.]), ([0.], [float('inf')])])
def test_unavailable_metrics_are_rejected(function, positive, negative):
    with pytest.raises(ValueError, match="nonempty finite"):
        function(torch.tensor(positive), torch.tensor(negative))


def test_valid_loss_retains_gradients():
    positive = torch.tensor([0., 1.], requires_grad=True)
    negative = torch.tensor([-1., 0.], requires_grad=True)
    loss, accuracy = _bce_link_loss(positive, negative)
    assert torch.isfinite(loss) and 0 <= accuracy <= 1
    loss.backward()
    assert torch.isfinite(positive.grad).all() and torch.isfinite(negative.grad).all()


def test_finite_logits_that_overflow_combined_loss_are_rejected():
    with pytest.raises(ValueError, match="not finite"):
        _bce_link_loss(torch.tensor([-3e38]), torch.tensor([3e38]))


def test_checkpoint_restores_best_weights_after_later_training(tmp_path):
    service = loaded_service()
    request = pb.TrainStepRequest(train_edge_indices=[0], neg_src=[3], neg_dst=[0])
    invoke(service, "TrainStep", request)
    best = {key: value.clone() for key, value in service.model.state_dict().items()}
    path = str(tmp_path / "best.pt")
    invoke(service, "Checkpoint", pb.CheckpointRequest(path=path))
    for _ in range(3):
        invoke(service, "TrainStep", request)
    assert any(not torch.equal(best[k], v) for k, v in service.model.state_dict().items())
    invoke(service, "LoadCheckpoint", pb.LoadCheckpointRequest(path=path, device="cpu"))
    for key, value in service.model.state_dict().items():
        torch.testing.assert_close(value, best[key])
