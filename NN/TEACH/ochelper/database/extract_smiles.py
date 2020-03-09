import pubchempy as pcp
import rdkit

# Training set to be mw less than 250
# from cid 1 to 50000 for the first example

compound = pcp.Compound.from_cid(206)
print(compound.iupac_name)
print(compound.molecular_formula)
print(compound.molecular_weight)
print(compound.isomeric_smiles, "\n")

quit()


for cid in range(1,20):
    print(cid)
    compound = pcp.Compound.from_cid(cid)
    print(compound.iupac_name)
    print(compound.molecular_formula)
    print(compound.molecular_weight)
    print(compound.isomeric_smiles,"\n")

