from finsight.config import Settings
from finsight.data.synthetic import generate_synthetic_panel
from finsight.eventstudy.study import event_study


def test_event_study_detects_signal_direction():
    panel = generate_synthetic_panel(Settings(n_days=500))
    events = [(n.date, n.asset, n.true_sentiment or 0.0) for n in panel.news]
    res = event_study(panel, events, threshold=0.5)
    assert res.n_pos > 10 and res.n_neg > 10
    # Positive news should drift up relative to negative news, significantly.
    assert res.car_long_short_final > 0
    assert res.t_long_short > 2.0
    assert len(res.taus) == len(res.car_pos)
