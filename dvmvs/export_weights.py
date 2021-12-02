import os
import shutil

from path import Path
from dvmvs.config import Config

def export(weights, src_dir, dst_dir, model_names):
    weight_files = []
    prev_module = -1
    prev_max_epoch = -1
    for file in sorted((src_dir / weights).files("module_*")):
        elem = file.split('_')
        module = int(elem[1])
        epoch = int(elem[3][6:])
        if module != prev_module:
            weight_files.append(file)
            prev_module = module
            prev_max_epoch = epoch
        elif epoch > prev_max_epoch:
            weight_files[-1] = file
            prev_max_epoch = epoch

    assert len(weight_files) == len(model_names), "number of weight files (%d) and model names (%d) differ" % (len(weight_files), len(model_names))

    print('loss for %s:' % weights)
    print('\t' + ', '.join(weight_files[0].split('_')[4:]))

    os.makedirs(dst_dir / weights, exist_ok=True)
    for i, model_name in enumerate(model_names):
        file_name = "%d_%s" % (i, model_name)
        shutil.copy2(src_dir / weights / weight_files[i], dst_dir / weights / file_name)

    print('saved in "%s"' % (dst_dir / weights))


def main():
    pairnet_src_dir = fusionnet_src_dir = Path(Config.train_run_directory)
    pairnet_weights = Path(Config.pairnet_weights)
    fusionnet_weights = Path(Config.fusionnet_weights)
    pairnet_dst_dir = Path(os.getcwd()) / Path("pairnet")
    fusionnet_dst_dir = Path(os.getcwd()) / Path("fusionnet")
    pairnet_model_names = ["feature_extractor", "feature_pyramid", "encoder", "decoder"]
    fusionnet_model_names = ["feature_extractor", "feature_pyramid", "encoder", "lstm_fusion", "decoder"]
    export(pairnet_weights, pairnet_src_dir, pairnet_dst_dir, pairnet_model_names)
    export(fusionnet_weights, fusionnet_src_dir, fusionnet_dst_dir, fusionnet_model_names)

    print('copy from "%s"' % (pairnet_dst_dir / pairnet_weights))

    os.makedirs(fusionnet_dst_dir / pairnet_weights, exist_ok=True)
    for i, model_name in enumerate(pairnet_model_names[:-1]):
        file_name = "%d_%s" % (i, model_name)
        shutil.copy2(pairnet_dst_dir / pairnet_weights / file_name, fusionnet_dst_dir / pairnet_weights / file_name)

    print('saved in "%s"' % (fusionnet_dst_dir / pairnet_weights))


if __name__ == '__main__':
    main()