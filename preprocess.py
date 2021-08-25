import csv

import numpy as np
import pyexcel as pyx
import yaml

from cell_classifier.c2v_cell_classifier import C2VCellClassifier
from reader.sheet import Sheet
from src.excel_toolkit import get_sheet_names, get_sheet_tarr, get_feature_array


def convert(fname, config):
    file_type = fname.split(".")[-1]

    c2v_cc = C2VCellClassifier(config)
    if file_type == "csv":

        with open(fname) as f:
            reader = csv.reader(f, delimiter=',')
            values = [["" if ((cell is None) or (cell == "None")) else cell for cell in row] for row in reader]
            pyx.save_as(array=values, dest_file_name=fname + ".xlsx")

            fname = fname + ".xlsx"
            file_type = "xlsx"

    sheet_list = []

    for sid, sname in enumerate(get_sheet_names(fname, file_type=file_type)):
        tarr, n, m = get_sheet_tarr(fname, sname, file_type=file_type, max_cols=99999, max_rows=99999)
        print("row:", n, "col:", m)
        ftarr = get_feature_array(fname, sname, file_type=file_type)
        tarr = np.array([["" if ((cell is None) or (cell == "None")) else cell for cell in row] for row in tarr])

        temp = c2v_cc.generate_features(Sheet(tarr, {"farr": ftarr}))

        embs = [[_.cpu().detach().numpy().tolist() for _ in item] for item in temp]

        sheet = Sheet(tarr,
                {"farr": ftarr,
                 "name": sname,
                 "embeddings": embs
                })

        sheet_list.append(sheet)
        #break

    return sheet_list


if __name__=='__main__':
    with open('./cfg/test_config.yaml') as ymlfile:
        c = yaml.load(ymlfile)

    convert('./files/election-polls.xlsx', c)