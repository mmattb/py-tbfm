#!/usr/bin/env python
# coding: utf-8

import os
import random
import shutil
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
import tqdm
import torch
import matplotlib.pyplot as plt

from tbfm import dataset
from tbfm import meta
from tbfm import multisession
from tbfm import utils

DATA_DIR = "/var/data/opto-coproc/"

OUT_DIR = "test"  # Local data cache; i.e. not reading from the opto-coproc folder.
EMBEDDING_REST_SUBDIR = "embedding_rest"
DEVICE = "cuda"  # cfg.device


def main(num_bases, num_sessions, gpu, coadapt=False, basis_residual_rank_in=None, train_size=5000, shuffle=False):

    my_out_dir = os.path.join(OUT_DIR, f"{num_bases}_{num_sessions}")
    if basis_residual_rank_in is not None:
        my_out_dir += f"_rr{basis_residual_rank_in}" + f"_{ 'coadapt' if coadapt else 'inner' }"

    # Add train_size and shuffle to folder name
    my_out_dir += f"_ts{train_size}"
    if shuffle:
        my_out_dir += "_shuffle"

    try:
        shutil.rmtree(my_out_dir)
    except OSError:
        pass
    os.makedirs(my_out_dir)

    print(f"----------- Device: {gpu}, out: {my_out_dir} ------------")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    meta = dataset.load_meta(DATA_DIR)
    conf_dir = Path("./conf").resolve()

    # Initialize Hydra with the configuration directory
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        # Compose the configuration
        cfg = compose(config_name="config")  # i.e. conf/config.yaml

    WINDOW_SIZE = cfg.data.trial_len
    NUM_HELD_OUT_SESSIONS = cfg.training.num_held_out_sessions

    if num_sessions == 40:
        held_in_session_ids = None
    elif num_sessions < 40:
        held_in_session_ids = [
            "MonkeyJ_20160426_Session2_S1",
            "MonkeyG_20150914_Session1_S1",
            "MonkeyG_20150915_Session3_S1",
            "MonkeyG_20150915_Session5_S1",
            "MonkeyG_20150916_Session4_S1",
            "MonkeyG_20150917_Session1_M1",
            "MonkeyG_20150917_Session1_S1",
            "MonkeyG_20150917_Session2_M1",
            "MonkeyG_20150917_Session2_S1",
            "MonkeyG_20150921_Session3_S1",
            "MonkeyG_20150921_Session5_S1",
            "MonkeyG_20150922_Session1_S1",
            "MonkeyG_20150922_Session2_S1",
            "MonkeyG_20150925_Session1_S1",
            "MonkeyG_20150925_Session2_S1",
            "MonkeyJ_20160426_Session3_S1",
            "MonkeyJ_20160428_Session3_S1",
            "MonkeyJ_20160429_Session1_S1",
            "MonkeyJ_20160502_Session1_S1",
            "MonkeyJ_20160624_Session3_S1",
            "MonkeyJ_20160625_Session4_S1",
            "MonkeyJ_20160625_Session5_S1",
            "MonkeyJ_20160627_Session1_S1",
            "MonkeyJ_20160630_Session3_S1",
            "MonkeyJ_20160702_Session2_S1",
        ][:num_sessions]

        MAX_BATCH_SIZE = 62500 // 8
        batch_size = (MAX_BATCH_SIZE // num_sessions) * num_sessions
    else:
        raise ValueError("blah")

    d, held_out_session_ids = multisession.load_stim_batched(
        window_size=WINDOW_SIZE,
        session_subdir="torchraw",
        data_dir=DATA_DIR,
        unpack_stiminds=True,
        held_in_session_ids=held_in_session_ids,
        batch_size=batch_size,
        num_held_out_sessions=NUM_HELD_OUT_SESSIONS,
    )
    data_train, data_test = d.train_test_split(train_size, test_cut=2500)

    held_in_session_ids = data_train.session_ids

    # Gather cached rest embeddings...
    embeddings_rest = multisession.load_rest_embeddings(
        held_in_session_ids, device=DEVICE
    )

    # Batch sizes will be:
    print("Batch shapes:")
    print("Train")
    b = next(iter(data_train))
    k = list(b.keys())
    k0 = k[0]

    for batch in iter(data_train):
        print(batch[k0][0].shape)

    print("Test")
    b = next(iter(data_test))
    k = list(b.keys())
    k0 = k[0]

    for batch in iter(data_test):
        print(batch[k0][0].shape)

    def cfg_identity(cfg, dim):
        cfg.ae.training.coadapt = False
        cfg.ae.warm_start_is_identity = True
        cfg.latent_dim = dim

    def cfg_base(cfg, dim):
        cfg_identity(cfg, dim)
        # cfg.training.grad_clip = 2.0
        # cfg.tbfm.training.lambda_ortho = 0.05
        cfg.tbfm.module.use_film_bases = False
        cfg.tbfm.module.num_bases = 12
        cfg.tbfm.module.latent_dim = 2
        cfg.training.epochs = 12001
        cfg.normalizers.module._target_ = "tbfm.normalizers.ScalerZscore"

    def cfg_big_bases(cfg):
        # cfg.training.grad_clip = 2.0
        # cfg.tbfm.training.lambda_ortho = 0.05
        cfg.tbfm.module.use_film_bases = False
        cfg.tbfm.module.num_bases = 100
        cfg.tbfm.module.latent_dim = 3
        cfg.training.epochs = 12001
        cfg.latent_dim = 74
        cfg.ae.use_two_stage = False
        cfg.ae.training.lambda_ae_recon = 0.03
        cfg.tbfm.training.lambda_fro = 60.0

    cfg.training.epochs = 12001
    cfg.latent_dim = 85
    cfg.tbfm.module.num_bases = num_bases
    cfg.ae.training.lambda_ae_recon = 0.03
    cfg.ae.use_two_stage = False
    cfg.ae.two_stage.freeze_only_shared = False
    cfg.ae.two_stage.lambda_mu = 0.01
    cfg.ae.two_stage.lambda_cov = 0.01
    cfg.tbfm.training.lambda_fro = 75.0

    if basis_residual_rank_in == 0:
        cfg.meta.is_basis_residual = False
    else:
        cfg.meta.is_basis_residual = True
        cfg.meta.basis_residual_rank = basis_residual_rank_in or 16
        cfg.meta.training.lambda_l2 = 1e-2
    cfg.meta.training.coadapt = coadapt  # Enable co-adaptation of embeddings

    ms = multisession.build_from_cfg(cfg, data_train, device=DEVICE)

    # Initialize trainable stim embeddings for co-adaptation if enabled
    if cfg.meta.training.coadapt:
        embed_dim_stim = ms.model.bases.embed_dim_stim
        embeddings_stim_init = {}
        for session_id in held_in_session_ids:
            emb = torch.randn(embed_dim_stim, device=DEVICE) * 0.1
            emb.requires_grad = True
            embeddings_stim_init[session_id] = emb
    else:
        embeddings_stim_init = None

    model_optims = multisession.get_optims(cfg, ms, embeddings_stim=embeddings_stim_init)

    embeddings_stim, results = multisession.train_from_cfg(
        cfg,
        ms,
        data_train,
        model_optims,
        embeddings_rest,
        embeddings_stim=embeddings_stim_init,
        data_test=data_test,
        test_interval=1000,
        epochs=cfg.training.epochs,
        random_sample_support=shuffle,
    )

    torch.save(embeddings_stim, os.path.join(my_out_dir, "es.torch"))
    torch.save(results, os.path.join(my_out_dir, "r.torch"))
    torch.save(held_in_session_ids, os.path.join(my_out_dir, "hisi.torch"))
    multisession.save_model(ms, os.path.join(my_out_dir, "model.torch"))

    txt = [t[0] for t in results["train_losses"]]
    tlt = [t[1] for t in results["train_losses"]]
    plt.plot(txt[-500:], tlt[-500:], label="train")
    plt.savefig(os.path.join(my_out_dir, "losses_last.png"))
    plt.clf()

    plt.plot(txt, tlt, label="train")
    tx = [t[0] for t in results["test_losses"]]
    tl = [t[1] for t in results["test_losses"]]
    plt.plot(tx, tl, label="test")
    plt.legend()
    plt.savefig(os.path.join(my_out_dir, "losses.png"))
    plt.clf()

    tx = [t[0] for t in results["train_r2s"]]
    tr = [t[1] for t in results["train_r2s"]]
    plt.plot(tx, tr, label="train")
    te = [t[1] for t in results["test_r2s"]]
    plt.plot(tx, te, label="test")
    plt.legend()
    plt.savefig(os.path.join(my_out_dir, "r2s.png"))
    plt.clf()

    def graph_for_sid(sid, results, cidx=30):
        from tbfm import test

        y_hats = results["y_hat"][sid].detach().cpu()
        y_hats_test = results["y_hat_test"][sid].detach().cpu()

        y = results["y"][sid].detach().cpu()
        y_test = results["y_test"][sid][2].detach().cpu()

        y_hat_mean = torch.mean(y_hats, dim=0)
        y_hat_test_mean = torch.mean(y_hats_test, dim=0)
        y_mean = torch.mean(y, dim=0)
        y_test_mean = torch.mean(y_test, dim=0)

        plt.plot(y_hat_mean[20:, cidx], label="hat")
        plt.plot(y_mean[20:, cidx], label="y")
        plt.legend()
        plt.savefig(os.path.join(my_out_dir, "ymean.png"))
        plt.clf()

        plt.plot(y_hat_test_mean[20:, cidx], label="hat")
        plt.plot(y_test_mean[20:, cidx], label="y")
        plt.legend()
        plt.savefig(os.path.join(my_out_dir, "ytestmean.png"))
        plt.clf()

        test.graph_state_dependency(y, y_hats, title="Train", runway_length=0, ch=cidx)
        plt.savefig(os.path.join(my_out_dir, "statedep.png"))
        plt.clf()

        test.graph_state_dependency(
            y_test, y_hats_test, title="Test", runway_length=0, ch=cidx
        )
        plt.savefig(os.path.join(my_out_dir, "statedeptest.png"))
        plt.clf()

    graph_for_sid("MonkeyJ_20160426_Session2_S1", results, cidx=30)


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    # Usage: python tma_standalone.py <num_bases> <num_sessions> <gpu> [coadapt] [basis_residual_rank] [train_size] [shuffle]
    num_bases = int(sys.argv[1])
    num_sessions = int(sys.argv[2])
    gpu = sys.argv[3]
    coadapt = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
    basis_residual_rank = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else None
    train_size = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    shuffle = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False

    main(
        num_bases,
        num_sessions,
        gpu,
        coadapt=coadapt,
        basis_residual_rank_in=basis_residual_rank,
        train_size=train_size,
        shuffle=shuffle,
    )
