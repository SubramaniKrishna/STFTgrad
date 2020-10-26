"""
Code for the adaptive classifier with the differentiable STFT front-end
This will be trained on our test input signal, alternating sinusoids of 2 frequencies
"""
# Dependencies
import numpy as np
from tqdm import tqdm
import haiku as hk
import jax.numpy as jnp
import jax
import optax
from dstft import diff_stft
import sys

# Order of input arguments:
"""
1 : list of N to initialize classifier with
2 : learning rate
3 : number of epochs
"""

n = len(sys.argv[1]) 
a = sys.argv[1][1:n-1] 
a = a.split(',') 
  
list_N = [int(i) for i in a]
lr = float(sys.argv[2])
nepochs = int(sys.argv[3])


# Construct the test signal to classify:
# Sampling rate
fs = 200

# Durations and frequencies of the 2 sines
dur_sins = [0.2,0.2]
freqs = [20,80]
Ns = [int(fs*i) for i in dur_sins]
# adding some noise in the sine to prevent the classifier from overfitting
list_sin = [(np.sin(2*np.pi*(freqs[i]/fs)*np.arange(Ns[i])) + 0.2*np.random.randn(Ns[i])) for i in range(len(dur_sins))]
one_period = np.concatenate(list_sin)

# Repeat this Nr times
Nr = 20
signal = np.tile(one_period,Nr)
P = sum(dur_sins)
I1 = np.arange(0,Nr*P,P)
I2 = np.arange(0.2,Nr*P,P)

# Input dimension to the classifier after the differentiable STFT (it is zero-padded to ensure this dimension)
Nzp = 50
# Differentiable FT as pre-processor to classifier
def forward(x):
    mlp = hk.Sequential([
    hk.Linear(2), jax.nn.softmax
    ])
    return mlp(x)

net = hk.without_apply_rng(hk.transform(forward))

def loss_fn(param_dict, signal):
    
    params = param_dict["nn"]
    sigma = param_dict["s"]
    hf = 1
    N = int(jnp.round(6*sigma))
    # Adding some more noise during training to prevent classifier from overfitting on irrelevant aspects of the spectra
    signal = signal + 0.2*np.random.randn(signal.shape[0])
    x = diff_stft(signal, s = sigma,hf = hf)

    li = []
    l1 = jnp.array([[1,0]])
    l2 = jnp.array([[0,1]])
    l_c = []
    for i in range(x.shape[1]):
        timi = i*int(hf*N)/fs
        d1 = np.min(np.abs(I1 - timi))
        d2 = np.min(np.abs(I2 - timi))
        if(d1  < d2):
            li.append(1)
            l_c.append(l1)
        else:
            li.append(2)
            l_c.append(l2)

    li = np.array(li)
    l_c = np.concatenate(l_c,axis = 0).T

    xzp = jnp.concatenate([x,jnp.zeros((Nzp - (N//2 + 1),x.shape[1]))],axis = 0)
    logits = net.apply(params,xzp.T)

    # Regularized loss (Cross entropy + regularizer to avoid small windows)
    cel = -jnp.mean(logits*l_c.T) + (0.1/sigma)
    
    return cel

def update(
    param_dict,
    opt_state,
    signal
    ):
    grads = jax.grad(loss_fn)(param_dict, signal)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(param_dict, updates)
    return new_params, opt_state


# Training the Classifier
nH_evol_fin = []
# list_N = [10,15,20,25,30,45,55,65,70]
opt = optax.adam(lr)
rng = jax.random.PRNGKey(42)

for Ni in list_N:
    params = net.init(rng,np.random.randn(1,Nzp))
    sinit = (Ni/6)
    param_dict = {"nn":params,"s":sinit}
    opt_state = opt.init(param_dict)
    
    pfdict = 0
    nH = []
    for t in tqdm(range(nepochs)):
        param_dict, opt_state = update(param_dict, opt_state, signal)
        pfdict = param_dict
        nH.append(6*param_dict["s"])
    nH_evol_fin.append(nH)

# Plotting the evolution of the window length across epochs for different initializations
import matplotlib.pyplot as pyp
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

pyp.figure()
pyp.title('Convergence from varying initial values')
pyp.xlabel('Epoch')
pyp.ylabel('Window Length (N)')
for l,i in enumerate(nH_evol_fin):
    pyp.plot(i,'b')
pyp.show()