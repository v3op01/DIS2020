import numpy as np
import time
import scipy, math, sys, re
import pyscf
import pyscf.dft
from  pyscf import gto
import tensorflow as tf
from pyscf.scf import diis
np.set_printoptions(linewidth=220, suppress = True,precision = 7)

# geom = """
# N 0. 0. 0.
# N 0. 0. 1.1
# """
geom = """
H 0 0 0
H 0 0 0.75
"""
# geom = """
# F 0. 0. 0.
# F 0. 0. 1.1
# """

# benzene is not starting at ground state (WHY?)
# geom = """
# C         1.3635847210    0.2612695173    0.0000000000
# C         0.4555261987    1.3115340049    0.0000000000
# C        -0.9080593696    1.0502638144    0.0000000000
# C        -1.3635847210   -0.2612695173    0.0000000000
# C        -0.4555261987   -1.3115340049    0.0000000000
# C         0.9080593696   -1.0502638144    0.0000000000
# H         2.4174307273    0.4631905059    0.0000000000
# H         0.8075792146    2.3251505666    0.0000000000
# H         1.6098541443   -1.8619558368    0.0000000000
# H        -0.8075792146   -2.3251505666    0.0000000000
# H        -2.4174307273   -0.4631905059    0.0000000000
# H        -1.6098541443    1.8619558368    0.0000000000
# """
#ethylene does not suffer from wrong ground state
# geom = """
# C         0.6584709661    0.0000000000    0.0000000000
# C        -0.6584709661    0.0000000000    0.0000000000
# H        -1.2256586821    0.9144109968    0.0000000000
# H        -1.2256586821   -0.9144109968    0.0000000000
# H         1.2256586821    0.9144109968    0.0000000000
# H         1.2256586821   -0.9144109968    0.0000000000
# """
output = re.sub("py","dat",sys.argv[0])
mol = gto.Mole()
mol.atom = geom
mol.basis = 'cc-pvtz'
mol.build()

# H core
print "GENERATING Hcore"
H = mol.intor_symmetric('cint1e_kin_sph') + mol.intor_symmetric('cint1e_nuc_sph')
if mol._ecp:
	H += mol.intor_symmetric('ECPscalar_sph')
print "GENERATED"
S = mol.intor_symmetric('cint1e_ovlp_sph')
nao = n_ao = mol.nao_nr()
ijkl = mol.intor('cint2e_sph')
ijkl = np.reshape(ijkl,(nao,nao,nao,nao))
Vao = np.einsum('ijkl->ikjl',ijkl)
Enuc = mol.energy_nuc()

rho = 0.5*pyscf.scf.hf.init_guess_by_1e(mol)
n = int(round(np.trace(np.dot(rho,S))))
J = 2.*np.einsum("bija,ai->bj",Vao,rho)
K = -1.*np.einsum("biaj,ai->bj",Vao,rho)
F = H + J + K
# eigs, C = scipy.linalg.eigh(F,S)

E0 = np.trace(np.dot(rho,H+F)) + Enuc

print "\n\n"
print "========================="
print "            SCF"
print "========================="
conv = 10**-15
gamma = 0.001
SCF = True
it = 0
Rho0 = rho.copy()
while SCF:
    eigs, C = scipy.linalg.eigh(F,S)
    rho = np.eye(n_ao)
    rho[n:,n:] *= 0
    Rho1 = reduce(np.dot,[C,rho,C.T])
    J = 2.*np.einsum("bija,ai->bj",Vao,Rho1)
    K = -1.*np.einsum("biaj,ai->bj",Vao,Rho1)
    F = H + J + K #(AO)
    E1 = np.trace(np.dot(Rho1,H+F)) + Enuc
    dE = E1 - E0
    # if abs(dE) < 10**-10:
    if abs(dE) < conv or it > 10000:
        SCF = False
        print "Exiting SCF"
    E0 = E1
    Rho0 = Rho1.copy()
    it += 1
print it
print eigs
print "Final Energy",E1
# print "H\n",2*np.trace(np.dot(Rho1,H))
# print "J\n",np.trace(np.dot(Rho1,J))
# print "K\n",np.trace(np.dot(Rho1,K))

print "Transforming V(ao) to V(mo)"
VV = np.einsum('pqrs,qj->pjrs',Vao,C)
VV = np.einsum('pjrs,pi->ijrs',VV,C)
VV = np.einsum('ijrs,rk->ijks',VV,C)
Vmo = np.einsum('ijks,sl->ijkl',VV,C)

dip_ints = mol.intor('cint1e_r_sph', comp=3)
dip_mo = np.einsum('kij,ia,jb->kab', dip_ints,C,C)

print "Initiating class of TDCIS"

class TDCIS:
    def __init__(self,eigs_,Vmo_,C_, n_, nao_, dip_, field_ = "Imp"):
        # self.F = 0.5*(F_ + F_.T)
        self.Epsilon = eigs_ #MO
        self.nao = int(round(nao_))
        self.n = int(round(n_))
        self.V = Vmo_
        self.field = field_.upper()
        self.dip = dip_
        T0 = 1.0 + 0.0j
        T = np.zeros((self.nao-self.n,self.n)).astype(complex)
        self.rho0 = self.density(T0,T)
        self.pol0 = np.einsum("xij,ji->x",dip_,self.rho0)
        self.HCisEnergy = None
        self.HT_tf = None
        self.Prepare()
    def Prepare(self):
        self.T_pl = tf.placeholder(tf.float64,shape=tuple([self.nao - self.n, self.n]))
        self.L_pl = tf.placeholder(tf.float64,shape=tuple([self.nao - self.n, self.n]))
        self.Ei = tf.Variable(self.Epsilon)
        self.V_tf = tf.Variable(self.V)
        self.nao_tf = tf.Variable(self.nao)
        self.n_tf = tf.Variable(self.n)
        init = tf.global_variables_initializer()
        #self.HT_tf = tf.gradients(self.CISH_tf(self.L_pl, self.T_pl),self.L_pl)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        #self.summary_writer = tf.summary.FileWriter(self.train_dir, self.sess.graph)
        self.sess.run(init)
    def density(self,t0, t):
        rho0 = np.eye(self.nao,dtype=np.complex128)
        rho0[self.n:, self.n:] *= 0.0
        rho = rho0.copy()
        # print rho
        # rho = np.real(t0*np.conj(t0))*rho0
        rho[self.n:,:self.n] += 1.0/np.sqrt(2.0)*np.conj(t0)*t
        rho[:self.n,self.n:] += 1.0/np.sqrt(2.0)*np.conj(t.T)*t0
        rho[self.n:,self.n:] += 0.5*np.einsum('ai,bi->ab',np.conj(t),t)
        rho[:self.n,:self.n] -= 0.5*np.einsum('ai,aj->ij',np.conj(t),t)
        return rho
    def HT(self, T0, T, Mu):
        """
        This is strictly the singlet sub-block of the CIS equations.
        http://vergil.chemistry.gatech.edu/notes/cis/cis.pdf
        L indicates the LHS of <Psi|H|Psi>
        L0 = -\sqrt(2) mu_ia T_ia
        L_ai = T_ai(e_a - e_i) + sum_bj c_bj(2*(ai|jb)-(ab|ji)) - (mu_ai*T0*sqrt(2) + \sum_a' mu_aa' T_ia'+ \sum_i' mu_i'i T_ai')
        """
        if 0:
            n = self.n
            e_ai = self.Epsilon[n:,np.newaxis] - self.Epsilon[np.newaxis,:n]
            Jlike = 2.0*np.einsum("aijb,bj->ai",self.V[n:,:n,:n,n:],T)
            Klike = -1.0*np.einsum("abji,bj->ai",self.V[n:,n:,:n,:n],T)
            Mulike = math.sqrt(2.0)*T0*Mu[n:,:n] + np.dot(Mu[n:,n:],T) - np.dot(T,Mu[:n,:n])
            HT1 = -1.0j*(T*e_ai + Jlike + Klike - Mulike)
            HT0 = -1.0j*(-math.sqrt(2.0)*np.sum(np.dot(T,Mu[:n,n:])))
        else:
            HT0 = 0
            HT1 = np.zeros((self.nao-self.n,self.n)).astype(complex)
            for a in range(self.nao-self.n):
                for i in range(self.n):
                    HT0 += 1.0j*T[a,i]*Mu[i,self.n+a]
                    HT1[a,i] += -1.0j*(self.Epsilon[self.n+a] - self.Epsilon[i])*T[a,i]
                    HT1[a,i] += 1.0j*T0*Mu[i,self.n+a]

                    for b in range(self.nao-self.n):
                        HT1[a,i] += 1.0j/math.sqrt(2)*T[b,i]*Mu[self.n+a,self.n+b]
                        for j in range(self.n):
                            HT1[a,i] += -1.0j*T[b,j]*(2*self.V[self.n+a,j,i,self.n+b] - self.V[self.n+a,j,self.n+b,i])
                    for j in range(self.n):
                        HT1[a,i] += -1.0j/math.sqrt(2) * T[a,j] * Mu[i,j]
        return HT0 , HT1
    def CISH_tf(self,L,T):
        """
        This is strictly the singlet sub-block of the CIS equations.
        http://vergil.chemistry.gatech.edu/notes/cis/cis.pdf
        L_ai = T_ai(e_a - e_i) + sum_bj c_bj(2*(ai|jb)-(ab|ji))
        """
        e_ai = tf.transpose(self.Ei[self.n_tf:]) - self.Ei[:self.n_tf] # E_ai
        Jlike = 2.0*tf.einsum("aijb,bj->ai",self.V_tf,T)
        Klike = -1.0*tf.einsum("abji,bj->ai",self.V_tf,T)
        HT = T*e_ai + Jlike + Klike
        return tf.reduce_sum(L*HT, axis = [0,1])
    def field_setup(self,amp_=0.01,freq_=0.6,tau_=0.07,tOn_=0.1,pol_ = np.array([1.0,1.0,1.0])):
        self.amp = amp_
        self.freq = freq_
        self.tau = tau_
        self.tOn = tOn_
        self.pol = pol_
    def step(self,T0,T,it,dt = 0.02):
        tnow = it * dt
        # print "\n\n",tnow
        newT = T.copy()
        newT0 = T0
        mu = self.field_apply(tnow)
        k10, k1 = self.HT(newT0, newT, mu)
        v2 = (0.5 * dt)*k1 + T
        v20 = (0.5 * dt)*k10 + T0
        k20, k2 = self.HT(v20, v2, mu)
        v3 = (0.5 * dt)*k2 + T
        v30 = (0.5 * dt)*k20 + T0
        k30, k3 = self.HT(v30, v3, mu)
        v4 = (0.5 * dt)*k2 + T
        v40 = (0.5 * dt)*k20 + T0
        k40, k4 = self.HT(v40, v4, mu)
        newT += dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        newT0 += dt/6.0 * (k10 + 2.0*k20 + 2.0*k30 + k40)
        total_norm = np.sqrt(np.conj(newT0)*newT0 + np.sum(np.conj(newT)*newT)).real
        newT /= total_norm
        newT0 /= total_norm
        return newT0, newT
    def field_apply(self,tnow_):
        if self.field == "IMP":
            amp = self.amp*np.sin(self.freq*tnow_)*\
            (1.0/math.sqrt(2.0*3.1415*self.tau*self.tau))*\
            np.exp(-1.0*np.power(tnow_-self.tOn,2.0)/(2.0*self.tau*self.tau))
        elif self.field == "CW":
            amp = self.amp * np.sin(self.freq*tnow_)
        else:
            print "No field applied"
            print "No point of running propagation, (unless testing for stability of propagation)"
            amp = 0
        Field = np.einsum("kij,k->ij",self.dip,self.pol * amp)
        return 2.0 * Field

print "Preparing the propagation of CIS"
filewant = 1
if (filewant):
    fpop = open('pop.out','a')
    fdip = open('dipole.out','a')
# Initializer
#tda = TDTDA(Hmo,Vmo,C,n,nao,dip_mo,field_ = "Imp")
cis = TDCIS(eigs,Vmo,C,n,nao,dip_mo,field_ = "CW")
# Field Parameter Setup
if mol.basis == "sto-6g":
    evfr = 25.5218
elif mol.basis == "6-31g":
    evfr = 15.1419
elif mol.basis == "cc-pvtz":
    evfr = 13.5617
else:
    0
cis.field_setup(amp_ = 0.01, freq_ = evfr/27.2113)
max_iter = 100000
Prop = True
it = 0
pop_t = np.zeros((max_iter+1,1 + nao))
dip_t = np.zeros((max_iter+1,4))

T0 = 1.0+0.0j
T = np.zeros((nao-n,n),dtype=np.complex128)
dt = 0.005
start = time.time()
while Prop:
    T0,T = cis.step(T0,T,it,dt)
    rhot = cis.density(T0,T)
    dip = np.einsum('xij,ji->x', cis.dip, rhot) - cis.pol0
    tnow = np.array(it * dt)
    w,v = np.linalg.eig(rhot)
    pop_t[it,:] = np.append(tnow,rhot.diagonal().real).astype(float)
    dip_t[it,:] = np.append(tnow,dip).astype(float)
    if it%200==0:
        print "\n"
        print it, np.min(w.real), np.max(w.real), np.sum(w.real)
        print rhot.diagonal().real
        print sum(rhot.diagonal().real)
        # print rhot
        # print T0,T
        # print T0*T0.conj(),T*T.conj()
    if it >= max_iter:
        print "Propagation Ended"
        Prop = False
    it += 1
end = time.time()
print end - start, " sec"
if (filewant):
    fpop.close()
    fdip.close()

np.savetxt('dipole.out',dip_t,delimiter=' ')
np.savetxt('pop.out',pop_t,delimiter=' ')
