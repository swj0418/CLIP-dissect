import os
import argparse
import datetime
import json
import pandas as pd
import numpy as np
import torch

import utils
import similarity


parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-L/14",
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                   help="Which CLIP-model to use")
parser.add_argument("--target_model", type=str, default="imagenet256",
                   help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_layers", type=str, default="L0_36_1024,L1_36_1024",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="gan_probe",
                    choices = ["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden", "gan_probe"])
parser.add_argument("--concept_set", type=str, default="data/8k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="results", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder",
                                                                               "cos_similarity", "cos_similarity_cubed"])

parser.parse_args()

if __name__ == "__main__":
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")

    # similarity_fn = eval("similarity.{}".format(args.similarity_fn))
    similarity_fn = getattr(similarity, args.similarity_fn)

    # NOTE save activations
    print("save_activations for generative model")
    utils.save_gan_activations(
        clip_name=args.clip_model,
        target_name=args.target_model,
        target_layers=args.target_layers,
        d_probe=args.d_probe,
        concept_set=args.concept_set,
        batch_size=args.batch_size,
        device=args.device,
        pool_mode=args.pool_mode,
        save_dir=args.activation_dir,
    )

    outputs = {
        "layer": [],
        "unit": [],
        "description": [],
        "similarity": [],
    }

    # Read concept set
    with open(args.concept_set, "r") as f:
        words = (f.read()).split("\n")

    print("getting neuron-concept similarities...")
    all_layer_similarities = []
    for target_layer in args.target_layers:
        target_save_name, clip_save_name, text_save_name = utils.get_save_names(
            clip_name=args.clip_model,
            target_name=args.target_model,
            target_layer=target_layer,
            d_probe=args.d_probe,
            concept_set=args.concept_set,
            pool_mode=args.pool_mode,
            save_dir=args.activation_dir,
        )

        similarities = utils.get_similarity_from_activations(
            target_save_name,
            clip_save_name,
            text_save_name,
            similarity_fn,
            return_target_feats=False,
            device=args.device,
        )  # neuron x vocabulary_size
        print(target_layer, similarities.shape)
        all_layer_similarities.append(
            dict(
                similarities=similarities.detach().cpu(),
                layer=target_layer,
            )
        )

        # sys.exit(0)
        # take argmax
        # vals, ids = torch.max(similarities, dim=1)
        del similarities
        torch.cuda.empty_cache()
        # descriptions = [words[int(idx)] for idx in ids]
        # outputs["unit"].extend([i for i in range(len(vals))])
        # outputs["layer"].extend([target_layer] * len(vals))
        # outputs["description"].extend(descriptions)
        # outputs["similarity"].extend(vals.cpu().numpy())

    dir_out = f"my_data/{args.target_model}"
    os.makedirs(dir_out, exist_ok=True)
    fn_out = f"{dir_out}/all_layer_similarities.pt"
    print(f"saving neuron-concept similarities to {fn_out}")
    torch.save(all_layer_similarities, fn_out)

    # Save neuron-concept similarities per layer as numpy
    for sim in all_layer_similarities:
        layer_name = sim["layer"]

        sim = sim["similarities"]
        fn = f"{dir_out}/neuron_concept_similarities_{layer_name}.npy"
        np.save(fn, sim.numpy())

        # get top 100 concepts per neuron
        top = 100
        sim = sim.argsort(descending=True)
        sim = sim[:, :top]  # get top
        sim = sim.type(torch.int32)
        fn = f"{dir_out}/concepts_top{top}_{layer_name}.npy"
        np.save(fn, sim.numpy())

    # # save as csv
    # df = pd.DataFrame(outputs)
    # if not os.path.exists(args.result_dir):
    # os.mkdir(args.result_dir)
    # timestamp = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
    # save_path = f"{args.result_dir}/{args.target_model}_{timestamp}"
    # os.mkdir(save_path)
    # df.to_csv(os.path.join(save_path, "descriptions.csv"), index=False)

    # with open(os.path.join(save_path, "args.txt"), "w") as f:
    # json.dump(args.__dict__, f, indent=2)
