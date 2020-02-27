from TensorMol import *
import time
import random
import itertools as it
PARAMS["max_checkpoints"] = 3
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Takes two nearly identical crystal lattices and interpolates a core/shell structure, must be oriented identically and stoichiometric
def InterpolateGeometries():
	a=MSet('cspbbr3_tess')
	#a.ReadGDB9Unpacked(path='/media/sdb2/jeherr/TensorMol/datasets/cspbbr3/pb_tess_6sc/')
	#a.Save()
	a.Load()
	mol1 = a.mols[0]
	mol2 = a.mols[1]
	mol2.RotateX()
	mol1.AlignAtoms(mol2)
	optimizer = Optimizer(None)
	optimizer.Interpolate_OptForce(mol1, mol2)
	mol1.WriteXYZfile(fpath='./results/cspbbr3_tess', fname='cspbbr3_6sc_pb_tess_goopt', mode='w')
	# mol2.WriteXYZfile(fpath='./results/cspbbr3_tess', fname='cspbbr3_6sc_ortho_rot', mode='w')

def read_unpacked_set(set_name="chemspider12", paths="/media/sdb2/jeherr/TensorMol/datasets/chemspider12/*/", properties=["name", "energy", "gradients", "dipole"]):
	import glob
	a=MSet(set_name)
	for path in glob.iglob(paths):
		a.read_xyz_set_with_properties(paths, properties)
	print(len(a.mols), " Molecules")
	a.Save()

def TrainKRR(set_ = "SmallMols", dig_ = "GauSH", OType_ ="Force"):
	a=MSet("SmallMols_rand")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	d = Digester(TreatedAtoms, name_=dig_,OType_ =OType_)
	tset = TensorData(a,d)
	tset.BuildTrainMolwise("SmallMols",TreatedAtoms)
	manager=TFManage("",tset,True,"KRR_sqdiff")
	return

def RandomSmallSet(set_, size_):
	""" Returns an MSet of random molecules chosen from a larger set """
	print("Selecting a subset of "+str(set_)+" of size "+str(size_))
	a=MSet(set_)
	a.Load()
	b=MSet(set_+"_rand")
	mols = random.sample(range(len(a.mols)), size_)
	for i in mols:
		b.mols.append(a.mols[i])
	b.Save()
	return b

def TestMetadynamics():
	a = MSet("nicotine_opt")
	a.ReadXYZ()
	m = a.mols[-1]
	# ForceField = lambda x: QchemDFT(Mol(m.atoms,x),basis_ = '6-311g**',xc_='wB97X-D', jobtype_='force', filename_='jmols2', path_='./qchem/', threads=8)
	manager = TFMolManageDirect(name="BehlerParinelloDirectSymFunc_nicotine_vib_Tue_Nov_21_09.11.26_2017", network_type = "BehlerParinelloDirectSymFunc")
	def force_field(coords):
		energy, forces = manager.evaluate_mol(Mol(m.atoms, coords), True)
		return energy, forces * JOULEPERHARTREE
	masses = np.array(list(map(lambda x: ATOMICMASSESAMU[x-1],m.atoms)))
	print("Masses:", masses)
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 50000
	PARAMS["MDThermostat"] = "Andersen"
	PARAMS["MDTemp"]= 300.0
	PARAMS["MDV0"] = "Thermal"
	PARAMS["MetaMDBumpHeight"] = 0.00
	PARAMS["MetaMDBumpWidth"] = 0.01
	meta = MetaDynamics(force_field, m, EandF_=force_field)
	meta.Prop()

def test_md():
	PARAMS["tf_prec"] = "tf.float32"
	a = MSet("water10")
	a.ReadXYZ()
	mol = a.mols[1]
	network = BehlerParinelloGauSHv2(name="BPGauSH_water_wb97xd_6311gss_Sat_Apr_14_00.23.37_2018")
	def force_field(coords, forces=True):
		m=Mol(mol.atoms, coords)
		if forces:
			energy, forces = network.evaluate_mol(m, forces)
			return energy, JOULEPERHARTREE*forces
		else:
			energy = network.evaluate_mol(m, forces)
			return energy
	masses = np.array(map(lambda x: ATOMICMASSESAMU[x-1], mol.atoms))
	print("Masses:", masses)
	PARAMS["MDdt"] = 0.5
	PARAMS["RemoveInvariant"]=True
	PARAMS["MDMaxStep"] = 20000
	PARAMS["MDThermostat"] = None
	PARAMS["MDTemp"]= 300.0
	md = VelocityVerlet(force_field, mol, EandF_=force_field)
	md.Prop()

def TestTFBond():
	a=MSet("chemspider_all_rand")
	a.Load()
	d = MolDigester(a.BondTypes(), name_="CZ", OType_="AtomizationEnergy")
	tset = TensorMolData_BPBond_Direct(a,d)
	manager=TFMolManage("",tset,True,"fc_sqdiff_BPBond_Direct")

def TestTFGauSH():
	tf_precision = eval(PARAMS["tf_prec"])
	TensorMol.RawEmbeddings.data_precision = tf_precision
	np.set_printoptions(threshold=100000)
	a=MSet("SmallMols_rand")
	a.Load()
	MaxNAtom = a.MaxNAtom()
	zlist = []
	xyzlist = []
	labelslist = []
	natomlist = []
	for i, mol in enumerate(a.mols):
		paddedxyz = np.zeros((MaxNAtom,3), dtype=np.float32)
		paddedxyz[:mol.atoms.shape[0]] = mol.coords
		paddedz = np.zeros((MaxNAtom), dtype=np.int32)
		paddedz[:mol.atoms.shape[0]] = mol.atoms
		paddedlabels = np.zeros((MaxNAtom, 3), dtype=np.float32)
		paddedlabels[:mol.atoms.shape[0]] = mol.properties["forces"]
		xyzlist.append(paddedxyz)
		zlist.append(paddedz)
		labelslist.append(paddedlabels)
		natomlist.append(mol.NAtoms())
		if i == 999:
			break
	xyzstack = tf.stack(xyzlist)
	zstack = tf.stack(zlist)
	labelstack = tf.stack(labelslist)
	natomstack = tf.stack(natomlist)
	gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
	atomic_embed_factors = tf.Variable(PARAMS["ANES"], trainable=True, dtype=tf.float32)
	elements = tf.constant([1, 6, 7, 8], dtype=tf.int32)
	tmp = tf_gaussian_spherical_harmonics_channel(xyzstack, zstack, elements, gaussian_params, 4)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	# for i in range(a.mols[0].atoms.shape[0]):
	# 	print(a.mols[0].atoms[i], "   ", a.mols[0].coords[i,0], "   ", a.mols[0].coords[i,1], "   ", a.mols[0].coords[i,2])
	tmp2 = sess.run(tmp, options=options, run_metadata=run_metadata)
	print(tmp2)
	# print(tmp2[1])
	# print(tmp2.shape)
	# print(tmp3)
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open('timeline_step_tmp_tm_nocheck_h2o.json', 'w') as f:
		f.write(chrome_trace)
	# print(tmp2[3].shape)
	# print(a.mols[0].atoms.shape)
	# TreatedAtoms = a.AtomTypes()
	# d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	# # tset = TensorData(a,d)
	# mol_ = a.mols[0]
	# print(d.Emb(mol_, -1, mol_.coords[0], MakeOutputs=False)[0])
	# print(mol_.atoms[0])

def test_gaussian_overlap():
	gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
	tf_precision = eval(PARAMS["tf_prec"])
	TensorMol.RawEmbeddings.data_precision = tf_precision
	tmp = tf_gaussian_overlap(gaussian_params)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	tmp2 = sess.run(tmp)
	print(tmp2)

def train_forces_GauSH_direct(set_ = "SmallMols"):
	PARAMS["RBFS"] = np.array([[0.35, 0.35], [0.70, 0.35], [1.05, 0.35], [1.40, 0.35], [1.75, 0.35], [2.10, 0.35], [2.45, 0.35],[2.80, 0.35], [3.15, 0.35], [3.50, 0.35], [3.85, 0.35], [4.20, 0.35], [4.55, 0.35], [4.90, 0.35]])
	PARAMS["ANES"] = np.array([2.20, 1.0, 1.0, 1.0, 1.0, 2.55, 3.04, 3.44]) #pauling electronegativity
	PARAMS["SH_NRAD"] = 14
	PARAMS["SH_LMAX"] = 4
	PARAMS["HiddenLayers"] = [512, 512, 512, 512, 512, 512, 512]
	PARAMS["max_steps"] = 20000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 200
	PARAMS["NeuronType"] = "elu"
	PARAMS["learning_rate"] = 0.0001
	a=MSet(set_)
	a.Load()
	TreatedAtoms = a.AtomTypes()
	print("Number of Mols: ", len(a.mols))
	d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	tset = TensorDataDirect(a,d)
	manager=TFManage("",tset,True,"fc_sqdiff_GauSH_direct")

def test_tf_neighbor():
	np.set_printoptions(threshold=100000)
	a=MSet("SmallMols_rand")
	a.Load()
	MaxNAtom = a.MaxNAtom()
	zlist = []
	xyzlist = []
	labelslist = []
	for i, mol in enumerate(a.mols):
		paddedxyz = np.zeros((MaxNAtom,3), dtype=np.float32)
		paddedxyz[:mol.atoms.shape[0]] = mol.coords
		paddedz = np.zeros((MaxNAtom), dtype=np.int32)
		paddedz[:mol.atoms.shape[0]] = mol.atoms
		paddedlabels = np.zeros((MaxNAtom, 3), dtype=np.float32)
		paddedlabels[:mol.atoms.shape[0]] = mol.properties["forces"]
		xyzlist.append(paddedxyz)
		zlist.append(paddedz)
		labelslist.append(paddedlabels)
		if i == 99:
			break
	xyzstack = tf.stack(xyzlist)
	zstack = tf.stack(zlist)
	labelstack = tf.stack(labelslist)
	gaussian_params = tf.Variable(PARAMS["RBFS"], trainable=True, dtype=tf.float32)
	atomic_embed_factors = tf.Variable(PARAMS["ANES"], trainable=True, dtype=tf.float32)
	element = tf.constant(1, dtype=tf.int32)
	r_cutoff = tf.constant(5.0, dtype=tf.float32)
	element_pairs = tf.constant([[1,1,1], [1,1,6], [1,1,7], [1,1,8], [1,6,6], [1,6,7], [1,6,8], [1,7,7], [1,7,8], [1,8,8],
								[6,6,6], [6,6,7], [6,6,8], [6,7,7], [6,7,8], [6,8,8], [7,7,7], [7,7,8], [7,8,8], [8,8,8]], dtype=tf.int32)
	tmp = tf_triples_list(xyzstack, zstack, r_cutoff, element_pairs)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	# for i in range(a.mols[0].atoms.shape[0]):
	# 	print(a.mols[0].atoms[i], "   ", a.mols[0].coords[i,0], "   ", a.mols[0].coords[i,1], "   ", a.mols[0].coords[i,2])
	tmp3 = sess.run([tmp], options=options, run_metadata=run_metadata)
	# print(tmp3)
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open('timeline_step_tmp_tm_nocheck_h2o.json', 'w') as f:
		f.write(chrome_trace)
	print(tmp3)
	# print(tmp4[1])
	# print(tmp4)
	# TreatedAtoms = a.AtomTypes()
	# d = Digester(TreatedAtoms, name_="GauSH", OType_="Force")
	# # tset = TensorData(a,d)
	# mol_ = a.mols[0]
	# print(d.Emb(mol_, -1, mol_.coords[0], MakeOutputs=False)[0])
	# print(mol_.atoms[0])

def train_energy_pairs_triples():
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 200
	PARAMS["NeuronType"] = "relu"
	# PARAMS["tf_prec"] = "tf.float64"
	# PARAMS["self.profiling"] = True
	a=MSet("SmallMols")
	a.Load()
	TreatedAtoms = a.AtomTypes()
	print("Number of Mols: ", len(a.mols))
	d = Digester(TreatedAtoms, name_="GauSH", OType_="AtomizationEnergy")
	tset = TensorMolData_BP_Direct(a,d)
	manager=TFMolManage("",tset,True,"pairs_triples", Trainable_=True)

def train_energy_symm_func(mset):
	PARAMS["train_energy_gradients"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 200
	PARAMS["NeuronType"] = "elu"
	PARAMS["tf_prec"] = "tf.float64"
	a=MSet(mset)
	a.Load()
	print("Number of Mols: ", len(a.mols))
	manager = TFMolManageDirect(a, network_type = "BPSymFunc")

def train_energy_GauSH(mset):
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.30, 16)), axis=1)
	PARAMS["SH_NRAD"] = 16
	PARAMS["SH_LMAX"] = 5
	PARAMS["SH_rot_invar"] = False
	PARAMS["EECutoffOn"] = 0.0
	PARAMS["Elu_Width"] = 6.0
	PARAMS["train_gradients"] = False
	PARAMS["train_dipole"] = False
	PARAMS["train_rotation"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.00005
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	PARAMS["Profiling"] = False
	PARAMS["train_sparse"] = False
	PARAMS["sparse_cutoff"] = 7.0
	network = BehlerParinelloGauSH(mset)
	network.start_training()

def train_energy_GauSHv2(mset):
	PARAMS["RBFS"] = np.stack((np.linspace(0.1, 6.0, 16), np.repeat(0.30, 16)), axis=1)
	PARAMS["SH_LMAX"] = 5
	PARAMS["train_gradients"] = True
	PARAMS["train_dipole"] = False
	PARAMS["train_rotation"] = False
	PARAMS["train_sparse"] = False
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [512, 512, 512]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 1000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float32"
	network = BehlerParinelloGauSHv2(mset)
	network.start_training()

def train_energy_univ(mset):
	PARAMS["train_gradients"] = True
	PARAMS["train_charges"] = True
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [1024, 1024, 1024]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 20000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 32
	PARAMS["Profiling"] = False
	PARAMS["NeuronType"] = "relu"
	PARAMS["tf_prec"] = "tf.float64"
	network = UniversalNetwork_v2(mset)
	network.start_training()

def eval_test_set_univ(mset,ef):
	PARAMS["train_gradients"] = True
	PARAMS["train_charges"] = True
	PARAMS["weight_decay"] = None
	PARAMS["HiddenLayers"] = [1024, 1024, 1024]
	PARAMS["learning_rate"] = 0.0001
	PARAMS["max_steps"] = 10000
	PARAMS["test_freq"] = 5
	PARAMS["batch_size"] = 100
	PARAMS["Profiling"] = False
	PARAMS["NeuronType"] = "shifted_softplus"
	PARAMS["tf_prec"] = "tf.float64"
	network = UniversalNetwork(name=ef)
	molset = MSet(mset)
	molset.Load()
	energy_errors, gradient_errors, charge_errors = network.evaluate_set(molset)
	mae_e = np.mean(np.abs(energy_errors))
	mse_e = np.mean(energy_errors)
	rmse_e = np.sqrt(np.mean(np.square(energy_errors)))
	mae_g = np.mean(np.abs(gradient_errors))
	mse_g = np.mean(gradient_errors)
	rmse_g = np.sqrt(np.mean(np.square(gradient_errors)))
	mae_c = np.mean(np.abs(charge_errors))
	mse_c = np.mean(charge_errors)
	rmse_c = np.sqrt(np.mean(np.square(charge_errors)))
	print("MAE  Energy: ", mae_e, " Gradients: ", mae_g, " Charges: ", mae_c)
	print("MSE  Energy: ", mse_e, " Gradients: ", mse_g, " Charges: ", mse_c)
	print("RMSE  Energy: ", rmse_e, " Gradients: ", rmse_g, " Charges: ", rmse_c)


train_energy_univ("ForDis")
#eval_test_set_univ("ForDis","SF_Universal_")
