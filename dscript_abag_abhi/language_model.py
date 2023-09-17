import os
import sys
import subprocess as sp
import random
import torch
import h5py
import pickle

from tqdm import tqdm
from .fasta import parse, parse_directory, write
from .pretrained import get_pretrained
from .alphabets import Uniprot21
from .models.embedding import SkipLSTM
from .utils import log
from datetime import datetime

from abmap.model import AbMAPAttn
from abmap.plm_embed import reload_models_to_device


def lm_embed(sequence, use_cuda=False):
    """
    Embed a single sequence using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param sequence: Input sequence to be embedded
    :type sequence: str
    :param use_cuda: Whether to generate embeddings using GPU device [default: False]
    :type use_cuda: bool
    :return: Embedded sequence
    :rtype: torch.Tensor
    """

    model = get_pretrained("lm_v1")
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()
    model.eval()

    with torch.no_grad():
        alphabet = Uniprot21()
        es = torch.from_numpy(alphabet.encode(sequence.encode("utf-8")))
        x = es.long().unsqueeze(0)
        if use_cuda:
            x = x.cuda()
        z = model.transform(x)
        return z.cpu()


def embed_from_fasta(fastaPath, outputPath, device=0, verbose=False, use_abmap=False, both_abs=True):
    """
    Embed sequences using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param fastaPath: Input sequence file (``.fasta`` format)
    :type fastaPath: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    """

    log(f"use_abmap is: {use_abmap}")

    # LOAD ABMAP MODEL HERE!
    if use_abmap:
        reload_models_to_device(device)

        dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        pretrained_H = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                          proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)
        pretrained_L = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                          proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)

        model_path_H = '/net/scratch3/scratch3-3/chihoim/ablm/pretrained_models/AbMAP_beplerberger_H_epoch50.pt'
        model_path_L = '/net/scratch3/scratch3-3/chihoim/ablm/pretrained_models/AbMAP_beplerberger_L_epoch50.pt'

        checkpoint_H = torch.load(model_path_H, map_location=dev)
        pretrained_H.load_state_dict(checkpoint_H['model_state_dict'])
        pretrained_H.eval()

        checkpoint_L = torch.load(model_path_L, map_location=dev)
        pretrained_L.load_state_dict(checkpoint_L['model_state_dict'])
        pretrained_L.eval()



    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        if verbose:
            log(
                f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
            )
    else:
        if verbose:
            log("# Using CPU")

    if verbose:
        log("# Loading Model...")
    model = get_pretrained("lm_v1")
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()

    model.eval()
    if verbose:
        log("# Loading Sequences...")
    names, seqs = parse(fastaPath, verbose=True)
    alphabet = Uniprot21()
    encoded_seqs = []
    for s in tqdm(seqs):
        es = torch.from_numpy(alphabet.encode(s.encode("utf-8")))
        if use_cuda:
            es = es.cuda()
        encoded_seqs.append(es)
    
    # DELETE LATER!!
    from abmap.plm_embed import reload_models_to_device, embed_sequence
    reload_models_to_device(device)
    encoded_seqs = seqs

    # DELETE LATER!!
    # for s in tqdm(seqs):
    #     s1, s2 = s.split("|")
    #     es1 = torch.from_numpy(alphabet.encode(s1.encode("utf-8")))
    #     es2 = torch.from_numpy(alphabet.encode(s2.encode("utf-8")))
    #     if use_cuda:
    #         es1, es2 = es1.cuda(), es2.cuda()
    #     encoded_seqs.append((es1, es2))

    if verbose:
        num_seqs = len(encoded_seqs)
        log("# {} Sequences Loaded".format(num_seqs))
        log(
            "# Approximate Storage Required (varies by average sequence length): ~{}GB".format(
                num_seqs * (1 / 125)
            )
        )

    log("# Storing to {}...".format(outputPath))
    with torch.no_grad(), h5py.File(outputPath, "a") as h5fi:
        try:
            for (name, x) in tqdm(zip(names, encoded_seqs), total=len(names)):
                if name not in h5fi:
                    # print("name", name, "x", x)
                    # if use_abmap:

                    if both_abs and ('type: H' in name or 'type: L' in name):

                        # DELETE LATER!!!
                        x1, x2 = x
                        x1 = x1.long().unsqueeze(0)
                        z1 = model.transform(x1)
                        x2 = x2.long().unsqueeze(0)
                        z2 = model.transform(x2)
                        z = torch.cat((z1, z2), dim=1)
                        # --------------------------


                        # names_list = name.split('|')
                        # pdb_id1, chain_letter1 = names_list[0][:4], names_list[0][5]
                        # pdb_id2, chain_letter2 = names_list[2][:4], names_list[2][5]
                        # root_dir = '/net/scratch3/scratch3-3/chihoim/ab_ag_interaction/data/cdrembed_maskaug4/beplerberger'
                        # file_name1 = os.path.join(root_dir, f'{pdb_id1}_{chain_letter1}_abH.p')
                        # file_name2 = os.path.join(root_dir, f'{pdb_id2}_{chain_letter2}_abL.p')
                        # with open(file_name1, 'rb') as f:
                        #     x1 = pickle.load(f)
                        # with open(file_name2, 'rb') as f:
                        #     x2 = pickle.load(f)

                        # x1 = x1.unsqueeze(0).to(device)
                        # x2 = x2.unsqueeze(0).to(device)
                        # with torch.no_grad():
                        #     z1 = pretrained_H.embed(x1, task='structure', embed_type='variable')
                        #     z2 = pretrained_L.embed(x2, task='structure', embed_type='variable')

                        # z = torch.cat((z1, torch.zeros(1, 1, z1.shape[-1]).to(device), z2), 1)


                    elif 'type: H' in name or 'type: L' in name:

                        pdb_id, chain_letter = name[:4], name[5]
                        root_dir = '/net/scratch3/scratch3-3/chihoim/ab_ag_interaction/data/cdrembed_maskaug4/beplerberger'
                        file_name = os.path.join(root_dir, f'{pdb_id}_{chain_letter}_abH.p')
                        with open(file_name, 'rb') as f:
                            x = pickle.load(f)
                        x = x.unsqueeze(0).to(device)

                        with torch.no_grad():
                            z, _ = pretrained_H(x, x, None, None, task=0, return3=True)
                    else:
                        x = x.long().unsqueeze(0)
                        z = model.transform(x)

                    dset = h5fi.require_dataset(
                        name,
                        shape=z.shape,
                        dtype="float32",
                        compression="lzf",
                    )

                    dset[:] = z.cpu().numpy()

        except KeyboardInterrupt:
            sys.exit(1)


def embed_from_fasta_esm(fastaPath, outputPath, device=0, verbose=False, use_abmap=False, both_abs=True):
    """
    Embed sequences using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param fastaPath: Input sequence file (``.fasta`` format)
    :type fastaPath: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    """

    log(f"use_abmap is: {use_abmap}")

    # LOAD ABMAP MODEL HERE!
    if use_abmap:
        reload_models_to_device(device)

        dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        pretrained_H = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                          proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)
        pretrained_L = AbMAPAttn(embed_dim=2200, mid_dim2=1024, mid_dim3=512,
                                          proj_dim=252, num_enc_layers=1, num_heads=16).to(dev)

        model_path_H = '/net/scratch3/scratch3-3/chihoim/ablm/pretrained_models/AbMAP_beplerberger_H_epoch50.pt'
        model_path_L = '/net/scratch3/scratch3-3/chihoim/ablm/pretrained_models/AbMAP_beplerberger_L_epoch50.pt'

        checkpoint_H = torch.load(model_path_H, map_location=dev)
        pretrained_H.load_state_dict(checkpoint_H['model_state_dict'])
        pretrained_H.eval()

        checkpoint_L = torch.load(model_path_L, map_location=dev)
        pretrained_L.load_state_dict(checkpoint_L['model_state_dict'])
        pretrained_L.eval()



    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        if verbose:
            log(
                f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}"
            )
    else:
        if verbose:
            log("# Using CPU")

    if verbose:
        log("# Loading Model...")
    model = get_pretrained("lm_v1")
    torch.nn.init.normal_(model.proj.weight)
    model.proj.bias = torch.nn.Parameter(torch.zeros(100))
    if use_cuda:
        model = model.cuda()

    model.eval()
    if verbose:
        log("# Loading Sequences...")
    names, seqs = parse(fastaPath, verbose=True)
    alphabet = Uniprot21()
    encoded_seqs = []
    for s in tqdm(seqs):
        es = torch.from_numpy(alphabet.encode(s.encode("utf-8")))
        if use_cuda:
            es = es.cuda()
        encoded_seqs.append(es)
    
    # DELETE LATER!!
    # for s in tqdm(seqs):
    #     s1, s2 = s.split("|")
    #     es1 = torch.from_numpy(alphabet.encode(s1.encode("utf-8")))
    #     es2 = torch.from_numpy(alphabet.encode(s2.encode("utf-8")))
    #     if use_cuda:
    #         es1, es2 = es1.cuda(), es2.cuda()
    #     encoded_seqs.append((es1, es2))

    if verbose:
        num_seqs = len(encoded_seqs)
        log("# {} Sequences Loaded".format(num_seqs))
        log(
            "# Approximate Storage Required (varies by average sequence length): ~{}GB".format(
                num_seqs * (1 / 125)
            )
        )

    log("# Storing to {}...".format(outputPath))
    with torch.no_grad(), h5py.File(outputPath, "a") as h5fi:
        try:
            for (name, x) in tqdm(zip(names, encoded_seqs), total=len(names)):
                if name not in h5fi:
                    # print("name", name, "x", x)
                    # if use_abmap:

                    if both_abs and ('type: H' in name or 'type: L' in name):

                        # DELETE LATER!!!
                        # x1, x2 = x
                        # x1 = x1.long().unsqueeze(0)
                        # z1 = model.transform(x1)
                        # x2 = x2.long().unsqueeze(0)
                        # z2 = model.transform(x2)
                        # z = torch.cat((z1, z2), dim=1)
                        # --------------------------


                        names_list = name.split('|')
                        pdb_id1, chain_letter1 = names_list[0][:4], names_list[0][5]
                        pdb_id2, chain_letter2 = names_list[2][:4], names_list[2][5]
                        # root_dir = '/net/scratch3/scratch3-3/chihoim/ab_ag_interaction/data/cdrembed_maskaug4/beplerberger'
                        root_dir = '/net/scratch3/scratch3-3/chihoim/ab_ag_interaction/data/cdrembed_maskaug4/esm2'
                        file_name1 = os.path.join(root_dir, f'{pdb_id1}_{chain_letter1}_abH.p')
                        file_name2 = os.path.join(root_dir, f'{pdb_id2}_{chain_letter2}_abL.p')
                        with open(file_name1, 'rb') as f:
                            x1 = pickle.load(f)
                        with open(file_name2, 'rb') as f:
                            x2 = pickle.load(f)

                        x1 = x1.unsqueeze(0).to(device)
                        x2 = x2.unsqueeze(0).to(device)
                        with torch.no_grad():
                            z1 = pretrained_H.embed(x1, task='structure', embed_type='variable')
                            z2 = pretrained_L.embed(x2, task='structure', embed_type='variable')

                        z = torch.cat((z1, torch.zeros(1, 1, z1.shape[-1]).to(device), z2), 1)


                    elif 'type: H' in name or 'type: L' in name:

                        pdb_id, chain_letter = name[:4], name[5]
                        root_dir = '/net/scratch3/scratch3-3/chihoim/ab_ag_interaction/data/cdrembed_maskaug4/beplerberger'
                        file_name = os.path.join(root_dir, f'{pdb_id}_{chain_letter}_abH.p')
                        with open(file_name, 'rb') as f:
                            x = pickle.load(f)
                        x = x.unsqueeze(0).to(device)

                        with torch.no_grad():
                            z, _ = pretrained_H(x, x, None, None, task=0, return3=True)
                    else:
                        # x = x.long().unsqueeze(0)
                        # z = model.transform(x)

                        # DELETE LATER!!
                        emb = embed_sequence(x, 'esm2', device)
                        z = emb.long().unsqueeze(0)

                    dset = h5fi.require_dataset(
                        name,
                        shape=z.shape,
                        dtype="float32",
                        compression="lzf",
                    )

                    dset[:] = z.cpu().numpy()

        except KeyboardInterrupt:
            sys.exit(1)


def embed_from_directory(
    directory, outputPath, device=0, verbose=False, extension=".seq"
):
    """
    Embed all files in a directory in ``.fasta`` format using pre-trained language model from `Bepler & Berger <https://github.com/tbepler/protein-sequence-embedding-iclr2019>`_.

    :param directory: Input directory (``.fasta`` format)
    :type directory: str
    :param outputPath: Output embedding file (``.h5`` format)
    :type outputPath: str
    :param device: Compute device to use for embeddings [default: 0]
    :type device: int
    :param verbose: Print embedding progress
    :type verbose: bool
    :param extension: Extension of all files to read in
    :type extension: str
    """
    nam, seq = parse_directory(directory, extension=extension)
    fastaPath = f"{directory}/allSeqs.fa"
    if os.path.exists(fastaPath):
        fastaPath = f"{fastaPath}.{int(datetime.utcnow().timestamp())}"
    write(nam, seq, open(fastaPath, "w"))
    embed_from_fasta(fastaPath, outputPath, device, verbose)
