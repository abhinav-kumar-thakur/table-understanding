import os

import yaml

from block_extractor.block_extractor_psl_v2 import BlockExtractorPSLV2
from cell_classifier.psl_cell_classifier import PSLCellClassifier
from data_loader.load_majid_data import LoadCell2VecData


def main(config):
    data_loader = LoadCell2VecData(config['jl_path'])
    data_loader.load_data_from_sheet('/Users/informationist/Projects/src/github.com/abhinav-kumar-thakur/table-understanding/files/AMIS_Data.xls', config)

    sheet_list = data_loader.tables

    result_path = os.path.join(config["model_path"], config["dataset"])
    model_path = os.path.join(result_path, config["c2v"]["cell_classifier_model_file"] + '0' + ".model")
    cell_classifier = PSLCellClassifier(model_path, config)
    c2v_tags = cell_classifier.classify_cells_all_tables(sheet_list)

    be_model_file = os.path.join(result_path, config["c2v"]["block_extractor_model_file"] + '0' + ".model")
    extractor = BlockExtractorPSLV2(be_model_file, config, beta=0.01, lmd=10)
    sheet_with_blocks = extractor.extract_blocks_all_tables(sheet_list, c2v_tags)

    for count, sheet in enumerate(sheet_with_blocks):
        print('--------------------------------------------------------------')
        print(count)
        print('--------------------------------------------------------------')
        for i, blk in enumerate(sheet):
            print(i, blk)


if __name__ == "__main__":
    with open('./cfg/test_config.yaml') as ymlfile:
        c = yaml.load(ymlfile)

    main(c)
