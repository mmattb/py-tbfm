# Okay; this is a bit complicated... We want to be able to build the computational graph depending on provided options:
#  Session specific normalizers
#  Session-shared but TTA'd AEs versus session-specific AEs
#  Session-shared and TTA'd TBFMs versus session-specific TBFMs.
# We need a protocol whereby we input batches which have potentially mixed sessions, runs through the graph, and gives outputs, regardless
#  of the topology options above...
# In general: x and y values will be packaged as a collection: (session_id, *args, **kwargs)
# For the protocol let's assume the collection is unsorted:
#  we will be less likely to break then I think.

from utils import ScalerQuant
from ae import LinearChannelAE


def build(
    session_data,
    masks,
    latent_dim,
    normalizer_type=ScalerQuant,
    ae_type=LinearChannelAE,
    ae_is_shared=False,
    num_chan_max=96,
    device=None,
):
    """
    These are *held in* sets only:
        session_data: [(session_id, tensor(batch, time, ch)), ...]
        masks: [(session_id, mask), ...]
    """

    # Build normalizers
    # For now: no session-shared normalizer
    normalizers = {}
    for session_id, d in session_data:
        normalizer = normalizer_type()
        normalizer.fit(d)
        normalizers[session_id] = normalizer

    aes = {}
    lora_alphas = {}
    if ae_is_shared:
        ae = ae_type(
            num_chan_max=num_chan_max,
            latent_dim=latent_dim,
            use_bias=True,
            use_lora=True,
            device=device,
        )

        # TODO: PCA warm start using highest dimension session

        # TODO: set up LoRAs
        # TODO: LinearAE needs a LoRA dimension plumbed in; it assumes dim 1 right now.
        for session_id, d in session_data:
            lora_alphas[session_id] = ...

        ae = AEDispatcher(ae, masks, lora_alphas=lora_alphas)

        # TODO: we probably want a separate LR for AEs, and possible another for LoRAs.
        # TODO: an options to freeze one or the other. It's time for hydra.

        for session_id, d in session_data:
            aes[session_id] = ae
    else:
        for session_id, d in session_data:
            ae = ae_type(
                num_chan_max=num_chan_max,
                latent_dim=latent_dim,
                use_bias=True,
                use_lora=False,
                device=device,
            )
            ae = AEDispatcher(ae, {session_id: masks[session_id]})
            aes[session_id] = ae

    # TODO: build TBFMs
    # TODO: a module which flows them together


# TODO: TTA
