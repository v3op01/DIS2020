import datetime

from django.db import models
from django.utils import timezone
from rdkit import Chem


# #   Checker for smiles
# #       Validate if SMILES string is valid
# #   if smiles exists:
# #       pull Molecule
# #   else:
# #       Create new Molecule


class Molecule(models.Model):
    """
    model class that holds information on 
    attributes:
        id: Automatically assigned number (/path/)
        smiles: SMILES string

    """

    smiles = models.CharField(max_length=100)
    cid = models.IntegerField(default=0)
    name = models.CharField(max_length=100)
    calculation = models.BooleanField()
    
    def __str__(self):
        return self.smiles

    def validate_smiles(self):
        m = Chem.MolFromSmiles(self.smiles)
        self.assertIs(m,None)




class Conformer(models.Model):
    """
    model that is specific to xyz
    attribute:
        xyz:
        energy:
    """
    molecule = models.ForeignKey(Molecule, on_delete=models.CASCADE)
    conformer_id = models.IntegerField(default=0)
    xyz = models.CharField(max_length=10000)
    energy = models.FloatField(default=0)
    def __str__(self):
        return self.conformer_id



