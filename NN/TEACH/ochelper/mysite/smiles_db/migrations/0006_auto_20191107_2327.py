# Generated by Django 2.2.6 on 2019-11-07 23:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('smiles_db', '0005_molecule_cid'),
    ]

    operations = [
        migrations.AlterField(
            model_name='molecule',
            name='calculation',
            field=models.BooleanField(),
        ),
    ]
