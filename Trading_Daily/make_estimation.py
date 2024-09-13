import pandas as pd
import argparse as ap

from utils.log import printLog, printError
from utils.dataframe import ReadDf
from utils.estimate import Estimate


def parsing():
    parser = ap.ArgumentParser()
    parser.add_argument()


if __name__ == '__main__':
    try:
        args = parsing()

    except Exception as error:
        printError(error)