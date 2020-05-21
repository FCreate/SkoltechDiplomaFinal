from flask import Flask, jsonify, request
import rdkit
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds, CalcExactMolWt
from providers import robust_standardizer
import requests
from rdkit import Chem
import torch
from copy import deepcopy
import yaml
from sklearn.preprocessing import StandardScaler
import pandas  as pd
from providers import MolecularNotationDataset, SpecialTokenWrapperModel, SmilesModel, robust_standardizer
from legogram import LegoGram
from legogram.apps import LegoGramRNNSampler
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import RNN
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
path_model = "experiments/two_layer_predictor_0.001/model.pt"

endpoints = requests.get("https://backend.syntelly.com/endpoints").json()
endpoints_id2name = dict(zip([e['id'] for e in endpoints], [e['view'] for e in endpoints]))
dataset = torch.load('lg.bin')
smiles_dataset = dataset.smiles
scaler = torch.load("temp/scaler.pt")
model = torch.load(path_model)
model.eval()
print("All has been loaded")



@app.route('/predict/', methods= ['POST'])
def predict():
    req_data = request.get_json()
    print("Data requested")
    print(req_data)
    conditions = req_data["conditions"]
    num_rounds = req_data["num_rounds"]
    loyality = req_data["loyality"]
    num_of_mols = req_data["num_of_mols"]

    # molecules closer to aspirin
    # "Melting point", "Boiling point", "Water Solubility", loyality to drug design rules, number of rounds, number of molecules
    #conditions = [120, 285, -2.1, 0.7, 10, 10]
    #data = conditions[]
    result_arr = []
    for round in range(num_rounds):
        print(f"round {round}")
        number_generate = 100
        endp = torch.tensor(scaler.transform(np.array([conditions])))
        print(endp.shape)

        c = deepcopy(endp)
        c = [str(l) for l in list(c.numpy())]
        # endp = endp.unsqueeze(0)
        endp = endp.repeat(100, 1)
        endp = endp.unsqueeze(0)
        endp = endp.repeat(3, 1, 1)

        endp = endp.float()
        endp = endp.cuda()
        res = model.sample(endp, number_generate, dataset.model)
        valid = len(res) * 100 / number_generate
        print("valid : {} %".format(valid))
        # writer.add_scalar("Valid", valid, cnt)
        res = [robust_standardizer(mol) for mol in res]
        res = list(filter(lambda x: x is not None, res))
        mols = res
        print("Mols obtained")
        print(mols)
        vals_another = requests.post("https://backend.syntelly.com/tempSmilesArrToPredict",
                                     json={'smiles': mols}).json()
        for idx in range(len(vals_another)):
            elem = vals_another[idx]['data']
            for e in elem:
                e["endpoint_id"] = endpoints_id2name[e["endpoint_id"]]
        e2v = []
        for idx in range(len(vals_another)):
            e2v.append(dict(zip([e['endpoint_id'] for e in vals_another[idx]['data']],
                                [e['value'] for e in vals_another[idx]['data']])))
        smiles = [val['smiles'] for val in vals_another]
        mols = [robust_standardizer(mol) for mol in smiles]
        mols = [Chem.MolFromSmiles(mol) for mol in mols]
        molecular_weights = [CalcExactMolWt(mol) for mol in mols]
        logp = [MolLogP(mol) for mol in mols]
        atom_count = [mol.GetNumAtoms() for mol in mols]
        molar_reflactivity = [MolMR(mol) for mol in mols]
        numRings = [CalcNumRings(mol) for mol in mols]
        numRotBonds = [CalcNumRotatableBonds(mol) for mol in mols]
        numHAcceptors = [NumHAcceptors(mol) for mol in mols]
        numHDonors = [NumHDonors(mol) for mol in mols]
        bcf = [e['Bioconcentration factor'] for e in e2v]
        dev_tox = [e['Developmental toxicity'] for e in e2v]
        flash_point = [e['Flash point'] for e in e2v]
        boiling_point = [e['Boiling point'] for e in e2v]
        melting_points = [e['Melting point'] for e in e2v]
        water_solubility = [e['Water Solubility'] for e in e2v]

        result = [0] * len(smiles)
        for idx in range(len(smiles)):
            val = 0
            if (molecular_weights[idx] <= 480 and molecular_weights[idx] >= 160):
                val += 1
            if (logp[idx] <= 5.6 and logp[idx] >= -0.4):
                val += 1
            if (atom_count[idx] <= 70 and atom_count[idx] >= 20):
                val += 1
            if (molar_reflactivity[idx] >= 40 and molar_reflactivity[idx] <= 130):
                val += 1
            if (bcf[idx] < 3):
                val += 1
            if (dev_tox[idx] == 'Negative'):
                val += 1
            if (flash_point[idx] > (350 - 273.15)):
                val += 1
            if (boiling_point[idx] > (300 - 273.15)):
                val += 1
            if (numRings[idx] > 0):
                val += 1
            if (numRotBonds[idx] < 5):
                val += 1
            if (numHAcceptors[idx] <= 10):
                val += 1
            if (numHDonors[idx] <= 5):
                val += 1

            if (val / 12 >= loyality):
                result[idx] = val

        print(result)
        for idx in range(len(result)):
            if (result[idx] > 0):
                result_arr.append((smiles[idx], result[idx],
                                   (melting_points[idx], boiling_point[idx], water_solubility[idx]),
                                   mean_squared_error(
                                       scaler.transform(np.array(
                                           [[melting_points[idx], boiling_point[idx], water_solubility[idx]]])),
                                       scaler.transform(np.array([conditions]))
                                   )))

    result_arr.sort(key=lambda x: x[3])

    print(result_arr[:num_of_mols])
    return jsonify(result_arr[:num_of_mols])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12100)

    #p = argparse.ArgumentParser(description='Apply RNN Generator')
    #p.add_argument("--name_task", type=str, help="Name of task")
    #p.add_argument("--conditions", type = str, default = "conditions.yml", help = "Conditions to generate")
    #p.add_argumet("--result", type = str, default = "default_name", help = "Name of result file")

    #args = p.parse_args()


    #molecules closer to aspirin
    #"Melting point", "Boiling point", "Water Solubility", loyality to drug design rules, number of rounds, number of molecules


    #apply(path_model, conditions, path_result)



