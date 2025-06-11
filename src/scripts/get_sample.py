import os
import pickle  # nosec B403
import random
import shutil

if __name__ == "__main__":
    input_folder = "/data/driving/waymo/new_processed_scenarios_testing"
    input_meta_file = "/data/driving/waymo/new_processed_scenarios_test_infos.pkl"
    output_folder = "./samples"
    scenario_folder = f"{output_folder}/scenarios"
    sample_size = 10
    os.makedirs(scenario_folder, exist_ok=True)

    if not os.path.isdir(input_folder):
        raise AssertionError(f"Input folder '{input_folder}' does not exist.")

    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    # print(files)
    if not len(files) > sample_size:
        raise AssertionError(f"Not enough files in input folder. Found {len(files)}, need {sample_size}.")

    if not os.path.exists(input_meta_file):
        raise AssertionError(f"Metadata file '{input_meta_file}' does not exist.")

    with open(input_meta_file, "rb") as f:
        metas = pickle.load(f)  # nosec B301

    sample_metas = []
    sample_files = random.sample(files, sample_size)
    for filename in sample_files:
        src = os.path.join(input_folder, filename)
        dst = os.path.join(scenario_folder, filename)

        scenario_id = filename.split(".")[0].split("_")[-1]
        for meta in metas:
            if meta["scenario_id"] == scenario_id:
                sample_metas.append(meta)
                break

        shutil.copy2(src, dst)

    with open(os.path.join(output_folder, "sample_infos.pkl"), "wb") as f:
        pickle.dump(sample_metas, f)

    print(f"Copied {sample_size} files to '{output_folder}'.")
