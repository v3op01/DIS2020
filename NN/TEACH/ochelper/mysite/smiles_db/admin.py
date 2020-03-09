from django.contrib import admin

from .models import Molecule, Conformer

# StackedInline (vertical) vs TabularInline (Horizontal)
class ConformerInline(admin.TabularInline):
    model = Conformer
    # extra = 3


class MoleculeAdmin(admin.ModelAdmin):
    fieldsets = [
        ('NAME',      {'fields': ['name']}),
        ('SMILES',  {'fields': ['smiles']}),
        ('Calculation',  {'fields': ['calculation']}),
    ]
    inlines = [ConformerInline]
    search_fields = ['smiles']
    #list_filter = ['pub_date']
    # Displayed information on /smiles_db/molecules
    list_display = ('name', 'smiles', 'calculation')


admin.site.register(Molecule, MoleculeAdmin)
# admin.site.register(Conformer)
