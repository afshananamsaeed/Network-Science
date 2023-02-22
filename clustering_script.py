import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score, rand_score, pair_confusion_matrix
from igraph import Graph
import numpy as np
from utilities.initial_and_fixed import get_initial_and_fixed
from utilities.labels import get_labels
from tqdm import tqdm


def clustering(df, df_gt, file_name, h, output_dir):
    if h == "h0": df["sum"] = df[["h0"]]           
    else: df["sum"] = df[['h0', 'h1']].sum(axis=1)     

    df_gt = df_gt.dropna(axis=0, how='any', subset=["address", "entity"])
    df_gt = df_gt.drop_duplicates(subset="address", keep=False)

    entity_counts = df_gt["entity"].value_counts()
    rare_entities = entity_counts[entity_counts < 10].index
    df_gt = df_gt.loc[~df_gt["entity"].isin(rare_entities), :]

    gt_addr = set(df_gt["address"])
    sample_addr = set(df["node1"]).union(set(df["node2"]))
    sample_known_addr = sample_addr.intersection(gt_addr)

    df_gt_known = df_gt.loc[df_gt["address"].isin(sample_addr), ['address', 'entity']]
    known_entities = set(df_gt_known['entity'])

    known_addr_entity_dict_list = [{a: idx for a in set(df_gt_known.loc[df_gt_known["entity"] == e, "address"])} for idx, e in enumerate(known_entities)]
    known_addr_entity_dict = {}
    [known_addr_entity_dict.update(d) for d in known_addr_entity_dict_list]
    known_entity_counts = df_gt_known['entity'].value_counts().rename("count").to_frame()
    known_entity_counts["file"] = file_name
    known_entity_counts["n_nodes_graph"] = len(sample_addr)
    known_entity_counts["n_edges_graph"] = len(df)

    file_name_output = f"known_entity_counts_{file_name}_res.csv"
    output_path = output_dir + file_name_output

    print('output_path: ', output_path)

    known_entity_counts.to_csv(output_path, index=True)

    edge_tuples = df[["node1", "node2", "sum"]].itertuples(index=False)
    g = Graph.TupleList(edge_tuples, directed=False, weights=True)

    props = [0, 0.1] + list(np.geomspace(start=0.01, stop=0.4, num=15))
    props = sorted(props)

    sizes = [int(p * len(g.vs)) for p in props if int(p * len(g.vs)) <= len(sample_known_addr)]

    for seed in tqdm(range(101), desc = 'Seed Progress Bar: '):
        res_cols = [
                        "file", "n_nodes_graph", "n_edges_graph", "prop_graph", "prop_known",
                        "n_clusters", "cluster_sizes", "ami", "homog", "mod", "ars", "urs",
                ]

        res = {c: [] for c in res_cols}
        for i, size in tqdm(enumerate(sizes), desc = 'Sizes Progress Bar: '):
                initial, fixed = get_initial_and_fixed(g, sample_known_addr, known_addr_entity_dict, size, seed=seed)
                cs = g.community_label_propagation(weights="weight", initial=initial, fixed=fixed)
                cs_addr = sorted([g.vs.select(c)["name"] for c in cs], key=len)
                cs_sizes = [len(c) for c in cs_addr]

                labels_true, labels_pred = get_labels(cs_addr, known_addr_entity_dict, sample_known_addr)
                ami = adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels_pred)
                urs = rand_score(labels_true=labels_true, labels_pred=labels_pred)

                (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

                if fn == 0 and fp == 0:
                        ars = 1.0
                else:
                        ars = 2. * (tp.astype(np.float64) * tn.astype(np.float64) - fn.astype(np.float64) * fp.astype(np.float64)) / ((tp.astype(np.float64) + fn.astype(np.float64)) * (fn.astype(np.float64) + tn.astype(np.float64)) + (tp.astype(np.float64) + fp.astype(np.float64)) * (fp.astype(np.float64) + tn.astype(np.float64)))

                homog = homogeneity_score(labels_true=labels_true, labels_pred=labels_pred)
                mod = cs.modularity

                res["file"].append(file_name)
                res["n_nodes_graph"].append(len(sample_addr))
                res["n_edges_graph"].append(len(df))
                res["prop_graph"].append(props[i])
                res["prop_known"].append(size/len(sample_known_addr))
                res["n_clusters"].append(len(cs))
                res["cluster_sizes"].append(cs_sizes)
                res["ami"].append(ami)
                res["ars"].append(ars)
                res["urs"].append(urs)
                res["homog"].append(homog)
                res["mod"].append(mod)

        df_res = pd.DataFrame(res)
        file_name_output = f"{file_name}_{seed}_res.csv"
        output_path = output_dir + file_name_output
        df_res.to_csv(output_path, index=False)

