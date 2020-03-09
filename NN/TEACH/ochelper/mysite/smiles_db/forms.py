from django import forms

class MoleculeForm(forms.Form):
    molecule_smiles = forms.CharField(label="SMILES", max_length=200)


class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)
