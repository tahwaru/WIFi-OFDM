import numpy as np
import matplotlib.pyplot as plt

from OFDM_transmit import OFDM_Tx 
from OFDM_receive import OFDM_Rx
from Channel import channel_noise

#import scipy


# -------------------------------------------Parameters

K = 64 # total number of OFDM subcarriers for (I)FFT computation
# OFDM_symb = 40 # number of complex numbers per OFDM symbol
CP = K//4  # length of the cyclic prefix: 25% of the block

# subcarrier indexing
all_subcarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilot_subcarriers = np.hstack([11, 25, 39, 53])

DC_subcarrier = np.hstack([32])

upper_training = all_subcarriers[:6]
lower_training = all_subcarriers[-5:]
training_subcarriers = np.concatenate((upper_training, lower_training))


data_subcarriers = np.delete(all_subcarriers, np.concatenate((pilot_subcarriers, training_subcarriers, DC_subcarrier)))
#print(len(data_subcarriers))
length_data_subcarriers = len(data_subcarriers)
modulations = ["bpsk", "qpsk", "16qam", "64qam"]
SNRdb = np.arange (5, 15, 10)
rician_factor = 0 # "0 " means Rayleigh fading
Num_OFDM_symbols = 1
BER ={}
mu = {modulations[0] : 1,
      modulations[1] : 2,
      modulations[2] : 4,
      modulations[3] : 6
      }
for i_mod in range(len(modulations)):
    modulation = modulations[i_mod]
    print(modulation)
    BER[i_mod] = np.zeros((len(SNRdb),), dtype=float)
# ------------------------------------Tx & Rx objects
    OFDM_Tx1 = OFDM_Tx(modulation)
    channel = channel_noise()
    OFDM_Rx1 = OFDM_Rx(modulation)
    bit_error_counter = np.zeros((len(SNRdb), Num_OFDM_symbols), dtype=float)       # keeps the error rate of each OFDM symbol 
    OFDM_bits_payload_counter = np.zeros((len(SNRdb), Num_OFDM_symbols), dtype=float)
    
    for i_snr in range(len(SNRdb)):
        #i_symb = np.random.randint(0, Num_OFDM_symbols)
        for i_OFDM_symbol in range(Num_OFDM_symbols):
            bits = np.random.binomial(n=1, p=0.5, size=(length_data_subcarriers*mu[modulation], ))       
            OFDM_withCP = OFDM_Tx1.bits2complex_stream(bits)
            OFDM_Rx_signal = channel.apply_channel(OFDM_withCP, rician_factor, SNRdb[i_snr])
            bits_est = OFDM_Rx1.complex2bits(OFDM_Rx_signal)
            bit_error_counter[i_snr][i_OFDM_symbol]= np.sum(abs(bits[:len(bits_est)]-bits_est))
            OFDM_bits_payload_counter[i_snr][i_OFDM_symbol] = len(bits_est)
            # Checking out some estimated bits
            
           # if (i_OFDM_symbol == i_symb):
               # print("Displaying some decoded bits from OFDM symbol No: ", i_symb, "at SNR=", i_snr)
               # print(bits[::len(bits) // 15])
               # print(bits_est[::len(bits) // 15])
            plt.figure(i_mod)
            f = np.linspace(-K/2, K/2, K, endpoint=False)
            plt.plot(f, 20*np.log10(abs(np.fft.fftshift(np.fft.fft(OFDM_Rx_signal, K)/np.sqrt(K)))))
            #plt.plot(20*np.log10(abs(np.fft.fftshift(np.fft.fft(OFDM_Rx_signal, K)/np.sqrt(K)))))
            plt.show()

        BER[i_mod][i_snr] = sum(bit_error_counter[i_snr])/sum(OFDM_bits_payload_counter[i_snr])
   # plt.semilogy(SNRdb, BER[i_mod], label=modulation)

#plt.xlabel("SNR[dB]")
#plt.ylabel("BER")
#plt.legend()
#plt.show()
