import json
import logging
import multiprocessing
import os
import time
import traceback
from glob import glob

import hydra
from omegaconf import DictConfig, open_dict

from .eval_func import *
from .utils.grasp_data_preparation import prepare_prediction_grasp_data
from .utils.timer import _fmt


def safe_eval_one(params):
    index, (prediction, configs) = params
    grasp_data = prepare_prediction_grasp_data(prediction, configs)
    if grasp_data is None:
        return

    try:
        identifier = f"{grasp_data['obj_id']}/{index}"
        eval_npy_path = os.path.join(configs.task.eval_dir, f"{identifier}.npy")
        skip = configs.task.get("skip_existing", True)
        if skip and not (configs.task.debug_viewer or configs.task.debug_render):
            if os.path.exists(eval_npy_path):
                return
        if configs.hand.mocap:
            eval_func_name = f"{configs.setting}MocapEval"
        else:
            eval_func_name = f"{configs.setting}ArmEval"
        eval(eval_func_name)(grasp_data, configs, identifier).run()
        # copy grasp into annotated path
        return
    except Exception:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
    return


def task_eval_dexter(configs):
    assert (
        configs.task.simulation_metrics is not None
        or configs.task.analytic_fc_metrics is not None
        or configs.task.pene_contact_metrics is not None
    ), "You should at least evaluate one kind of metrics"

    prediction_path = os.path.join(configs.task.pred_path)
    with open(prediction_path, "r") as f:
        input_list = json.load(f)
    init_num = len(input_list)

    logging.info(f"Find {init_num} grasp data.")

    if configs.task.get("filter_ids_path", None):
        with open(configs.task.filter_ids_path) as f:
            filter_ids = set(line.strip() for line in f if line.strip())
        filter_indices = set(int(id.rsplit("/", 1)[-1]) for id in filter_ids)
        input_list = [pred for i, pred in enumerate(input_list) if i in filter_indices]
        logging.info(f"Filtering to {len(input_list)} items from {configs.task.filter_ids_path}")

    if len(input_list) == 0:
        return

    iterable_params = enumerate(zip(input_list, [configs] * len(input_list)))
    total = len(input_list)
    report_every = 100
    start = time.perf_counter()
    done = 0

    if configs.task.debug_viewer or configs.task.debug_render:
        for ip in iterable_params:
            safe_eval_one(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            for _ in pool.imap_unordered(safe_eval_one, iterable_params):
                done += 1
                if (done % report_every == 0) or (done == total):
                    elapsed = time.perf_counter() - start
                    eta = elapsed * (total / done - 1.0)
                    est_total = elapsed + eta
                    logging.info(
                        f"[{done}/{total}] "
                        f"elapsed={_fmt(elapsed)}  "
                        f"ETA≈{_fmt(eta)}  "
                        f"est_total≈{_fmt(est_total)}"
                    )

    succ_lst = glob(os.path.join(configs.task.succ_dir, "**/*.npy"), recursive=True)
    eval_lst = glob(os.path.join(configs.task.eval_dir, "**/*.npy"), recursive=True)
    logging.info(
        f"{len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.task.succ_dir}"
    )
    logging.info("Finish evaluation")

    return


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # Derive output dirs from the directory containing predictions.json, if not set
    prediction_dir = os.path.dirname(os.path.abspath(cfg.task.pred_path))
    with open_dict(cfg):
        if cfg.task.eval_dir is None:
            cfg.task.eval_dir = os.path.join(prediction_dir, "eval")
        if cfg.task.succ_dir is None:
            cfg.task.succ_dir = os.path.join(prediction_dir, "succ")
        if cfg.task.log_dir is None:
            cfg.task.log_dir = os.path.join(prediction_dir, "log")
        if cfg.task.debug_dir is None:
            cfg.task.debug_dir = os.path.join(prediction_dir, "debug")

    os.makedirs(cfg.task.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(cfg.task.log_dir, "eval.log")),
            logging.StreamHandler(),
        ],
    )

    task_eval_dexter(cfg)


if __name__ == "__main__":
    main()
