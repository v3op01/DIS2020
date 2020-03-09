from django.contrib import admin

from .models import Molecule, Conformer

# StackedInline (vertical) vs TabularInline (Horizontal)
class ConformerInline(admin.TabularInline):
    model = Conformer
    extra = 3


class MoleculeAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,      {'fields': ['question_text']}),
        ('SMILES',  {'fields': ['smiles']}),
    ]
    inlines = [ConformerInline]
    search_fields = ['question_text']
    list_filter = ['pub_date']
    list_display = ('question_text','pub_date','was_published_recently')

admin.site.register(Molecule, MoleculeAdmin)
admin.site.register(Conformer)
