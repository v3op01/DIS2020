from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.urls import reverse
from django.views import generic, View
from django.utils import timezone
from django.template import loader
from .models import Molecule, Conformer
from .forms import MoleculeForm, NameForm


# class HomeView(generic.DetailView):
#     template_name = 'smiles_db/home.html'
#     context_object_name = 'molecule_list'

#     def get_queryset(self):
#         """
#         Return the last five published questions (not including those set to be
#         published in the future).
#         """
#         return Molecule.objects.all()


class Search(View):
    def get(self, request, *args, **kwargs):
        return HttpResponse("Hello, world (searched)")


class SearchResultsView(generic.ListView):
    model = Molecule
    template_name = 'search_results.html'
    def get_queryset(self):
        query = self.request.GET.get('q')
        object_list = Molecule.objects.filter(name__icontains=query)
        return object_list



class HomeView(generic.TemplateView):
    template_name = 'home.html'


class IndexView(generic.ListView):
    template_name = 'index.html'
    context_object_name = 'molecule_list'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        return Molecule.objects.filter(calculation=True)[:]

class DetailView(generic.DetailView):
    model = Molecule
    template_name = 'smiles_db/detail.html'
    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return Molecule.objects.all()

# class ResultsView(generic.DetailView):
#     model = Question
#     template_name = 'polls/results.html'



# FUNCTIONS
def searching(request):
    #context = {}
    if request.method == 'POST':
        form = MoleculeForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('/thanks/')
    else:
        form = MoleculeForm()
    return render(request, 'test.html', {'form': form})

    #     molecule = request.POST.get('searching')
    #     object_list = Molecule.objects.filter(name__icontains=query)
    #     result = molecule + "something"
    #     context['molecule'] = result
    # return render(request, 'search_result.html', object_list)


def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()

    return render(request, 'name.html', {'form': form})


# def search_results(request):
#     query = request.GET.get('q')
#     object_list = Molecule.objects.filter(name__icontains=query)
#     return render(request, 'search_results.html', object_list)
def search_results(request):
    molecule_list = Molecule.objects.filter(name__icontains=query)
    context = {'molecule_list': molecule_list}
    return render(request, 'smiles_db/index.html', context)

    # if request.method == "POST":
    #     input = request.POST['smiles_submission']
    #     # print(input)
    #     return HttpResponse(input, content_type='text/plain')

def submit_smiles(request):
    if request.method == "POST":
        input = request.POST['smiles_submission']
        # print(input)
        return HttpResponse(input, content_type='text/plain')
    #query = request.GET.get('smiles_submission')
    #print(query)

def home(request):
    molecule_list = Molecule.objects
    context = {'molecule_list': molecule_list}
    return render(request, 'smiles_db/home.html', context)

def index(request):
    molecule_list = Molecule.objects
    context = {'molecule_list': molecule_list}
    return render(request, 'smiles_db/index.html', context)

def detail(request, molecule_id):
    molecule = get_object_or_404(Molecule, pk=molecule_id)
    return render(request, 'smiles_db/detail.html',{'molecule': molecule})

# def results(request, question_id):
#     response = "You're looking at the results of question %s. "
#     return HttpResponse(response % question_id)

# def vote(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     try:
#         selected_choice = question.choice_set.get(pk=request.POST['choice'])
#     except (KeyError, Choice.DoesNotExist):
#         # Redisplay the question voting form.
#         return render(request, 'polls/detail.html', {
#             'question': question,
#             'error_message': "You didn't select a choice.",
#         })
#     else:
#         selected_choice.votes += 1
#         selected_choice.save()
#         # Always return an HttpResponseRedirect after successfully dealing
#         # with POST data. This prevents data from being posted twice if a
#         # user hits the Back button.
#         return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))



# # Create your views here.
