# Generated by Django 2.2.6 on 2019-10-30 21:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('smiles_db', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='molecule',
            name='name',
            field=models.CharField(default=1, max_length=100),
            preserve_default=False,
        ),
    ]
