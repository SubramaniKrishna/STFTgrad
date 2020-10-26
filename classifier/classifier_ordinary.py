"""
Code for a normal classifier (to obtain the loss function as a function of the window length)
This will be trained on our test input signal, alternating sinusoids of 2 frequencies
"""

# Dependencies
from tqdm import tqdm
import haiku as hk
import jax.numpy as jnp
import jax
import optax
from dstft import diff_stft

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

# Constructing the classifier
def forward(x):
    mlp = hk.Sequential([
    hk.Linear(2), jax.nn.log_softmax
    ])
    return mlp(x)

net = hk.without_apply_rng(hk.transform(forward))

def loss_fn(params, inp, labels):
    logits = net.apply(params,inp)
    cel = -jnp.mean(logits * labels)
    return cel

def update(
    params: hk.Params,
    opt_state,
    x,l
    ):
    grads = jax.grad(loss_fn)(params, x,l)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# Training the classifier
loss_arrs_N = []
loss_fin = []
N_sweep = [5,10,40,50,60,70]

opt = optax.adam(1e-1)
rng = jax.random.PRNGKey(42)

for N in N_sweep:
    Nc = N
    Nzp = 50
    hf = 1
    signal = signal + 0.1*np.random.randn(signal.shape[0])
    x = diff_stft(signal, s = Nc/6,hf = hf)

    li = []
    l1 = jnp.array([[1,0]])
    l2 = jnp.array([[0,1]])
    l_c = []
    for i in range(x.shape[1]):
        timi = i*int(hf*Nc)/fs
        d1 = np.min(np.abs(I1 - timi))
        d2 = np.min(np.abs(I2 - timi))
        if(d1  < d2):
            li.append(1)
            l_c.append(l1)
        else:
            li.append(2)
            l_c.append(l2)

    li = np.array(li)
    l_c = np.concatenate(l_c,axis = 0)
    xzp = jnp.concatenate([x,jnp.zeros((Nzp - (Nc//2 + 1),x.shape[1]))],axis = 0)
    
    params = net.init(rng,np.random.randn(1,Nzp))
    opt_state = opt.init(params)
    
    paramsf = 0
    liter = []
    for t in tqdm(range(2000)):
        params, opt_state = update(params, opt_state, xzp.T, l_c)
        paramsf = params
        liter.append(loss_fn(paramsf,xzp.T, l_c) + (0.6/N))
    
    loss_arrs_N.append(liter)
    loss_fin.append(loss_fn(paramsf,xzp.T, l_c))

# Plotting the spectrograms and final loss for the different N's
costs_fin = loss_fin
import matplotlib
from matplotlib.pylab import register_cmap
cdict = {
    'red':   ((0.0,  1.0, 1.0), (1.0,  0.0, 0.0)),
    'green': ((0.0,  1.0, 1.0), (1.0,  .15, .15)),
    'blue':  ((0.0,  1.0, 1.0), (1.0,  0.4, 0.4)),
    'alpha': ((0.0,  0.0, 0.0), (1.0,  1.0, 1.0))}
register_cmap(name='InvBlueA', data=cdict)

matplotlib.rcParams.update({'font.size': 16})
def plot_various_window_size(sigi):
    pyp.figure(figsize=(22, 4))
    szs = [5,10,40,50,60,70]
    for i in range(len(szs)):
        sz, hp = szs[i], szs[i]
        a = diff_stft(sigi,s = szs[i]*1.0/6,hf = 1)
        pyp.gcf().add_subplot(1, len(szs), i + 1), pyp.gca().pcolorfast(a,cmap = "InvBlueA")
        pyp.gca().set_title(f'FFT size: {sz}, \n Loss: {costs_fin[i]:.5f}')
        pyp.xlabel('Time Frame')
        pyp.ylabel('Frequency Bin')
    pyp.gcf().tight_layout()

plot_various_window_size(signal)