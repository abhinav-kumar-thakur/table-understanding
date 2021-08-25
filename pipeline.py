import os

import yaml

from block_extractor.block_extractor_psl_v2 import BlockExtractorPSLV2
from cell_classifier.psl_cell_classifier import PSLCellClassifier
from data_loader.load_majid_data import LoadCell2VecData


def main(config):
    data_loader = LoadCell2VecData(config['jl_path'])
    indices = [i for i in range(len(data_loader.tables))]
    sheet_list, celltype_list, blocktype_list, layouttype_list = data_loader.get_tables_from_indices(indices)

    result_path = os.path.join(config["model_path"], config["dataset"])
    model_path = os.path.join(result_path, config["c2v"]["cell_classifier_model_file"] + '0' + ".model")
    cell_classifier = PSLCellClassifier(model_path, config)
    c2v_tags = cell_classifier.classify_cells_all_tables(sheet_list)

    be_model_file = os.path.join(result_path, config["c2v"]["block_extractor_model_file"] + '0' + ".model")
    extractor = BlockExtractorPSLV2(be_model_file, config, beta=0.01, lmd=10)
    blocks = extractor.extract_blocks_all_tables(sheet_list, c2v_tags)

    for i, blk in enumerate(blocks[2]):
        print(i, blk)


if __name__ == "__main__":
    with open('./cfg/test_config.yaml') as ymlfile:
        c = yaml.load(ymlfile)

    main(c)
