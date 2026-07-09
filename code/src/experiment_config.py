from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """
    Configuration for one FL trilemma experiment run.

    Has stored default values that may be overriden as needed.

    Args:
        config_id (int):         which of the 8 combos this is.
        num_clients (int):       total number of simulated clients.
        num_rounds (int):        number of FL rounds.
        num_byzantine (int):     number of malicious clients.
        use_dp (bool):           whether to enable DP-SGD.
        epsilon (float):         privacy budget. Only used when use_dp=True.
        delta (float):           privacy failure probability.
        use_fltrust (bool):      whether to enable FLTrust aggregation.
        use_topk (bool):         whether to enable Top-k compression.
        topk_ratio (float):      fraction of parameters to keep. Only used when use_topk=True.
        root_dataset_size (int): number of clean samples the server holds.
        rescale_to_ref_norm (bool): whether to rescale client updates to reference norm.
        seed (int | None):      random seed for the root/client data split and each client's
                                model init + per-round training randomness. None keeps the
                                unseeded (different every run) behavior. Configs sharing the
                                same seed get the same data split and initial global model,
                                isolating whatever parameter differs between them.
    """
    config_id:          int
    num_clients:        int   = 4
    num_rounds:         int   = 10
    num_byzantine:      int   = 1
    use_dp:             bool  = False
    epsilon:            float = 10.0
    delta:              float = 1e-5
    use_fltrust:        bool  = False
    use_topk:           bool  = False
    topk_ratio:         float = 0.1
    root_dataset_size:  int   = 2000
    rescale_to_ref_norm: bool = False
    seed:               int | None = None