import argparse
import pandas as pd
import yaml
from metrics import *
import re

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args):
    # Load categories
    categories_yaml = load_yaml(args.categories_yaml)
    
    # Load datasets
    testset = pd.read_csv(args.testset)
    trainset = pd.read_csv(args.trainset)
    outdir = args.outdir #inclues model
    
    # Example: print loaded data shapes
    print(f"Loaded testset: {testset.shape}")
    print(f"Loaded trainset: {trainset.shape}")
    print(f"Loaded categories: {list(categories_yaml.keys())}")


    #Break up test set in multilocalizing and single localizing proteins for later analysis
    
    #These lables are hierarchical but do not count as true multilocalization
    implicitly_multi = [
        "actin-filaments",
        "intermediate-filaments",
        "centrosome",
        "microtubules",
        "endosomes",
        "lysosomes",
        "peroxisomes"
        "lipid-droplets"
    ]
    pattern = "|".join(map(re.escape, implicitly_multi))

    single_testset = testset[~testset.level3.str.contains(";")]
    single_testset.loc[single_testset.level2.str.contains(";"), "level2"] = pd.NA
    single_testset.loc[
        (single_testset.level1.str.contains(";")) &
        ~((single_testset.level1.str.contains(pattern, na=False)) & (single_testset['level1'].str.count(";") == 1)), 
        "level1"] = pd.NA
    multi_testset = testset[~testset.uniprot_id.isin(single_testset.uniprot_id)]
    multi_testset.loc[~multi_testset.level2.str.contains(";"), "level2"] = pd.NA
    multi_testset.loc[~multi_testset.level1.str.contains(";"), "level1"] = pd.NA

    for tag in ["", "_single", "_multi"]:
    
        val_avg_df = []
        test_metrics_avg_df = []
        for level in [1,2,3]:
            val_perclass_df = []
            test_targets_all = []
            test_probs_all = []
            test_preds_all = []

            for fold in range(5):
                val_path = f"{outdir}/{fold}_1Layer_combined_level{level}_testout.pkl"
                val_df = pd.read_pickle(val_path)
                val_df = val_df.merge(trainset, 
                                    left_on="ACC", 
                                    right_on="uniprot_id", 
                                    how="inner")
                val_probs = np.stack(val_df.preds.to_numpy())
                val_targets = []
                for locs in val_df[f"level{level}"].str.split(";").to_list():
                    val_targets.append([1 if loc in locs else 0 
                                        for loc in categories_yaml[f"level{level}"]])
                val_targets = np.array(val_targets)

                thresholds = [get_best_threshold_mcc(val_targets[:, i], val_probs[:, i]) 
                            for i in range(val_targets.shape[1])]
                thresholds = np.array(thresholds)
                #TODO: save thresholds

                _, val_metrics_perclass, val_metrics_avg = all_metrics(val_targets, 
                                                                    val_probs, 
                                                                    thresholds=thresholds)
                val_metrics_perclass["label"] = categories_yaml[f"level{level}"]
                val_metrics_perclass["fold"] = fold
                val_metrics_avg["level"] = level
                val_metrics_avg["fold"] = fold
                val_perclass_df.append(val_metrics_perclass)
                val_avg_df.append(val_metrics_avg)

                test_path = f"{outdir}/{fold}_1Layer_combined_level{level}_hou_testout.pkl"
                test_df = pd.read_pickle(test_path)
                test_df = test_df.merge(testset, 
                                        left_on="ACC", 
                                        right_on="uniprot_id",
                                        how="inner")
                if tag=="_single":
                    test_df = test_df[test_df.uniprot_id.isin(single_testset[single_testset[f"level{level}"].notna()].uniprot_id)]
                elif tag=="_multi":
                    test_df = test_df[test_df.uniprot_id.isin(multi_testset[multi_testset[f"level{level}"].notna()].uniprot_id)]

                
                test_probs = np.stack(test_df.preds.to_numpy())
                test_targets = []
                for locs in test_df[f"level{level}"].str.split(";").to_list():
                    test_targets.append([1 if loc in locs else 0 
                                        for loc in categories_yaml[f"level{level}"]])
                test_targets = np.array(test_targets)


                test_preds = test_probs > thresholds[np.newaxis, :]

                test_targets_all.append(test_targets)
                test_probs_all.append(test_probs)   
                test_preds_all.append(test_preds)

            val_perclass_df = pd.concat(val_perclass_df)
            val_perclass_df.to_csv(f"{outdir}/val_metrics_perclass_level{level}{tag}.csv", index=False)

            test_targets_all = np.array(test_targets_all)
            assert np.all(test_targets_all[0, :, :] == test_targets_all)
            test_targets = test_targets_all[0, :, :]
            test_probs = np.array(test_probs_all).mean(axis=0)
            test_preds = (np.array(test_preds_all).mean(axis=0) > 0.5).astype(np.int32)


            labels = np.array(categories_yaml[f"level{level}"])
            predicted_labels = [set(labels[np.where(pred==1)[0]]) for pred in test_preds]

            #Cut out empty categories in testset like plastid
            idxs = np.where(test_targets.sum(axis=0) != 0)[0]
            test_targets = test_targets[:, idxs]
            test_probs = test_probs[:, idxs]
            test_preds = test_preds[:, idxs]
            thresholds = thresholds[idxs]

            _, test_metrics_perclass, test_metrics_avg = all_metrics(
                                                                    test_targets, 
                                                                    test_probs, 
                                                                    y_pred_bin = test_preds,
                                                                    thresholds=thresholds
                                                                    )
            test_metrics_perclass["label"] = np.array(categories_yaml[f"level{level}"])[idxs]
            test_metrics_perclass.to_csv(
                f"{outdir}/test_metrics_perclass_level{level}{tag}.csv", index=False)
            
            test_metrics_avg["level"] = level
            test_metrics_avg_df.append(test_metrics_avg)

        val_avg_df = pd.concat(val_avg_df)
        val_avg_df.to_csv(f"{outdir}/val_metrics_avg{tag}.csv", index=False)
        test_metrics_avg_df = pd.concat(test_metrics_avg_df)
        test_metrics_avg_df.to_csv(f"{outdir}/test_metrics_avg{tag}.csv", index=False)

    # ... Insert your metrics calculation code here ...
    # For example:
    # results = calculate_metrics(testset, trainset, categories_yaml)
    # Save results to outdir
    # os.makedirs(args.outdir, exist_ok=True)
    # results.to_csv(os.path.join(args.outdir, "metrics.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from testset and trainset.")
    parser.add_argument("--categories_yaml", type=str, required=True, help="Path to categories YAML file")
    parser.add_argument("--testset", type=str, required=True, help="Path to testset CSV file")
    parser.add_argument("--trainset", type=str, required=True, help="Path to trainset CSV file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for results")
    args = parser.parse_args()
    main(args)