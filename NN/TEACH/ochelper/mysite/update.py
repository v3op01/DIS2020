from smiles_db.models import Molecule, Conformer
import pubchempy as pcp
import os


def MakeMoleculeModel(smiles):
    molecule = Molecule()
    molecule.smiles = smiles
    # molecule.name = getname()
    molecule.calculation = False
    molecule.save()
    return molecule


def SearchConformer(xyz):
    # xyz0, xyz1, xyz2 = ConformationSearch()
    return [xyz0, xyz1, xyz2]


def OptimizeConformer(xyz):

    return opted_xyz, energy


def main():

    # File open for SMILES STRING
    lst = ["CC", "CCC", "CCCC"]
    for SMILES in lst:
        try:
            # check if the smiles string already exists in the database
            Molecule.objects.get(smiles=SMILES)
        except:
            molecule = MakeMoleculeModel(SMILES)
            quit()
            conformer_results = SearchConformer(SMILES)
            # download sdf from Pubchem or make xyz from rdkit
            # search conformers
            # pick lowest 3 xyz
            for i in range(3):
                molecule.conformer_set.create(conformer_id=i, xyz=conformer_results[i])
        
main()
