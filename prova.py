import tensorflow as tf
import matplotlib.pyplot as plt
import network

W=network.new_create_recurrent_weights_ring_attractor(N=256, g=2.1, rho=1.6, Si=2000.)

plt.imshow(W)

plt.savefig('prova.png')