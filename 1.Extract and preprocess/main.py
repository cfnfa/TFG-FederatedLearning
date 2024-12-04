""" Author:Clara Fuertes """

from cleaning import clean
from process import preprocess_cgm,  preprocess_bolus, preprocess_basal
from visualize import visualize
from merge import merge
 


def main():
    #patients = ['AGM','AML','CGL','CNM','GAG','GCQ','GML','JLG','JMPM','MAVS','SVB','VVC']
    patients = ['VVC']
    print(f'Hello Clara, you are preprocessing the data from  {len(patients)} patients.')
    clean(patients)
    preprocess_cgm (patients)
    preprocess_bolus(patients)
    preprocess_basal(patients)
    #visualize()
    merge()

if __name__ == "__main__":
    main()