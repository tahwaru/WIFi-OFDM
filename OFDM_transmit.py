import numpy as np
class OFDM_Tx:
    def __init__(self, modulation):
        self.modulation = modulation  # string containing "bpsk"or "qpsk" or "16qam" or "64qam"
        self.N_FFT = 64 # total number of subcarriers
        self.all_subcarriers = np.arange(self.N_FFT)
        self.training_subcarriers = np.concatenate((self.all_subcarriers[:6], self.all_subcarriers[-5:])) # indices of the training subcarriers
        self.pilot_subcarriers = np.hstack([11, 25, 39, 53]) # indices of the pilot subcarriers
        self.DC_subcarrier = np.hstack([32])
        self.data_subcarriers = np.delete(self.all_subcarriers, np.concatenate((self.pilot_subcarriers, self.training_subcarriers, self.DC_subcarrier))) # indices of the data subcarriers
        self.CP = self.N_FFT // 4 # length of the cyclic prefix: 25% of the block
      
        self.length_data_subcarriers = len(self.data_subcarriers)
        self.pilot_symbols = np.array([1 + 1j*0, 1 + 1j*0, 1 + 1j*0, -1 + 1j*0])

        self.bandwidth = 5e6 # 5 or 10 0r 20 MHz
        self.subcarrier_spacing = self.bandwidth/self.N_FFT
        self.center_freq = 2.4e9 # 2.4 GHz or 5GHz
        self.Fs = 2.3 * self.center_freq
        self.freq_index = self.Fs * np.arange(-self.N_FFT//2, self.N_FFT//2)/self.N_FFT
        self.baud_rate     = 10  #baud_rate
        self.bits_per_baud = 1 #bits_per_baud
        self.oversampling_factor = 2.1
        
        
    def sampling_rate(self) -> float:
        return self.oversampling_factor * self.subcarrier_spacing * self.N_FFT
        
        
           
    def bits2complex_stream(self, bits):
        if (self.modulation == "bpsk"):
            mu = 1 
            payloadBits_per_OFDM = self.length_data_subcarriers*mu
            bits_bpsk = bits[:payloadBits_per_OFDM]
            assert(payloadBits_per_OFDM == len(bits_bpsk))
            bits_SP = bits_bpsk.reshape((self.length_data_subcarriers, mu)) # serial to parallel
            mapping_table = self.mapping_bpsk()
            complex_symbols = np.array([mapping_table[tuple(b)] for b in bits_SP]) 
            
        if (self.modulation == "qpsk"):
            mu = 2 
            payloadBits_per_OFDM = self.length_data_subcarriers*mu
            bits_qpsk = bits[:payloadBits_per_OFDM]
            assert(payloadBits_per_OFDM == len(bits_qpsk))
            bits_SP = bits_qpsk.reshape((self.length_data_subcarriers, mu)) 
            mapping_table = self.mapping_qpsk()
            complex_symbols = np.array([mapping_table[tuple(b)] for b in bits_SP])
        
        if (self.modulation == "16qam"):
            mu = 4 
            payloadBits_per_OFDM = self.length_data_subcarriers*mu
            bits_16qam = bits[:payloadBits_per_OFDM]
            assert(payloadBits_per_OFDM == len(bits_16qam))
            bits_SP = bits_16qam.reshape((self.length_data_subcarriers, mu)) 
            mapping_table = self.mapping_16qam()
            complex_symbols = np.array([mapping_table[tuple(b)] for b in bits_SP])
            
        if (self.modulation == "64qam"):
            mu = 6 
            payloadBits_per_OFDM = self.length_data_subcarriers*mu
            bits_64qam = bits[:payloadBits_per_OFDM]
            assert(payloadBits_per_OFDM == len(bits_64qam))
            bits_SP = bits_64qam.reshape((self.length_data_subcarriers, mu)) 
            mapping_table = self.mapping_64qam()
            complex_symbols = np.array([mapping_table[tuple(b)] for b in bits_SP])
         
        ofdm_symbol = self.OFDM_symbol(complex_symbols)
        ofdm_time = self.ifft_signal(ofdm_symbol)
        OFDM_withCP = self.add_CP(ofdm_time)
        
        return OFDM_withCP #complex_stream
           
    def mapping_bpsk(self): # 0 --> -1   ,  1 --> 1
        
        mapping_table = {
          (0,) : -1 + 0*1j,
          (1,) : 1 + 0*1j
         }
        return mapping_table
    
    def mapping_qpsk(self):
        a=np.sqrt(2)
        
        mapping_table = {
        (0,0) : (-1-1j)/a,
        (0,1) : (-1+1j)/a,
        (1,0) : (1-1j)/a,
        (1,1) : (1+1j)/a
        
        }
        return  mapping_table
        
    def mapping_16qam(self):
        
        a=np.sqrt(10)

        mapping_table = {
        (0,0,0,0) : (-3-3j)/a,
        (0,0,0,1) : (-3-1j)/a,
        (0,0,1,0) : (-3+3j)/a,
        (0,0,1,1) : (-3+1j)/a,
        (0,1,0,0) : (-1-3j)/a,
        (0,1,0,1) : (-1-1j)/a,
        (0,1,1,0) : (-1+3j)/a,
        (0,1,1,1) : (-1+1j)/a,
        (1,0,0,0) :  (3-3j)/a,
        (1,0,0,1) :  (3-1j)/a,
        (1,0,1,0) :  (3+3j)/a,
        (1,0,1,1) :  (3+1j)/a,
        (1,1,0,0) :  (1-3j)/a,
        (1,1,0,1) :  (1-1j)/a,
        (1,1,1,0) :  (1+3j)/a,
        (1,1,1,1) :  (1+1j)/a
        }
        
    
        return  mapping_table

    def mapping_64qam(self):
        
        a=np.sqrt(42)
        mapping_table = {
        (0,0,0,0,0,0) : (+7+7j)/a,
        (0,0,0,0,0,1) : (+5+7j)/a,
        (0,0,0,0,1,0) : (+1+7j)/a,
        (0,0,0,0,1,1) : (+3+7j)/a,
        (0,0,0,1,0,0) : (-7+7j)/a,
        (0,0,0,1,0,1) : (-5+7j)/a,
        (0,0,0,1,1,0) : (-1+7j)/a,
        (0,0,0,1,1,1) : (-3+7j)/a,
        (0,0,1,0,0,0) : (+7+5j)/a,
        (0,0,1,0,0,1) : (+5+5j)/a,
        (0,0,1,0,1,0) : (+1+5j)/a,
        (0,0,1,0,1,1) : (+3+5j)/a,
        (0,0,1,1,0,0) : (-7+5j)/a,
        (0,0,1,1,0,1) : (-5+5j)/a,
        (0,0,1,1,1,0) : (-1+5j)/a,
        (0,0,1,1,1,1) : (-3+5j)/a,#00
        (0,1,0,0,0,0) : (+7+1j)/a,
        (0,1,0,0,0,1) : (+5+1j)/a,
        (0,1,0,0,1,0) : (+1+1j)/a,
        (0,1,0,0,1,1) : (+3+1j)/a,
        (0,1,0,1,0,0) : (-7+1j)/a,
        (0,1,0,1,0,1) : (-5+1j)/a,
        (0,1,0,1,1,0) : (-1+1j)/a,
        (0,1,0,1,1,1) : (-3+1j)/a,
        (0,1,1,0,0,0) : (+7+3j)/a,
        (0,1,1,0,0,1) : (+5+3j)/a,
        (0,1,1,0,1,0) : (+1+3j)/a,
        (0,1,1,0,1,1) : (+3+3j)/a,
        (0,1,1,1,0,0) : (-7+3j)/a,
        (0,1,1,1,0,1) : (-5+3j)/a,
        (0,1,1,1,1,0) : (-1+3j)/a,
        (0,1,1,1,1,1) : (-3+3j)/a,# 10
        (1,0,0,0,0,0) : (+7-7j)/a,
        (1,0,0,0,0,1) : (+5-7j)/a,
        (1,0,0,0,1,0) : (+1-7j)/a,
        (1,0,0,0,1,1) : (+3-7j)/a,
        (1,0,0,1,0,0) : (-7-7j)/a,
        (1,0,0,1,0,1) : (-5-7j)/a,
        (1,0,0,1,1,0) : (-1-7j)/a,
        (1,0,0,1,1,1) : (-3-7j)/a,
        (1,0,1,0,0,0) : (+7-5j)/a,
        (1,0,1,0,0,1) : (+5-5j)/a,
        (1,0,1,0,1,0) : (+1-5j)/a,
        (1,0,1,0,1,1) : (+3-5j)/a,
        (1,0,1,1,0,0) : (-7-5j)/a,
        (1,0,1,1,0,1) : (-5-5j)/a,
        (1,0,1,1,1,0) : (-1-5j)/a,
        (1,0,1,1,1,1) : (-3-5j)/a,#11
        (1,1,0,0,0,0) : (+7-1j)/a,
        (1,1,0,0,0,1) : (+5-1j)/a,
        (1,1,0,0,1,0) : (+1-1j)/a,
        (1,1,0,0,1,1) : (+3-1j)/a,
        (1,1,0,1,0,0) : (-7-1j)/a,
        (1,1,0,1,0,1) : (-5-1j)/a,
        (1,1,0,1,1,0) : (-1-1j)/a,
        (1,1,0,1,1,1) : (-3-1j)/a,
        (1,1,1,0,0,0) : (+7-3j)/a,
        (1,1,1,0,0,1) : (+5-3j)/a,
        (1,1,1,0,1,0) : (+1-3j)/a,
        (1,1,1,0,1,1) : (+3-3j)/a,
        (1,1,1,1,0,0) : (-7-3j)/a,
        (1,1,1,1,0,1) : (-5-3j)/a,
        (1,1,1,1,1,0) : (-1-3j)/a,
        (1,1,1,1,1,1) : (-3-3j)/a
        }
        return mapping_table
    

    def OFDM_symbol(self, modulated_payload): 
        symbol = np.zeros(self.N_FFT, dtype=complex) 
        symbol[self.data_subcarriers] = modulated_payload  
        symbol[self.pilot_subcarriers] = 1 + 1j*0 
        symbol[self.pilot_subcarriers[-1]] = -1 + 1j*0 
        symbol[self.DC_subcarrier] = 0 + 1j*0    
        return symbol
    
    
    def ifft_signal(self, OFDM_freq_domain):
        return np.fft.fftshift(np.fft.ifft(OFDM_freq_domain))

    
    def add_CP(self, OFDM_time):
        cp = OFDM_time[-self.CP:]
        return np.hstack([cp, OFDM_time])

    # ----------Creating a preambule-----
    def random_qam(ofdm): # data for the preambule
        qam = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        return np.random.choice(qam, size=(self.N_FFT), replace=True)
    
    def addCFO(signal_CP, cfo):  # Add carrier frequency offset 
        return signal_CP * np.exp(1j*2*np.pi*cfo*np.arange(len(signal_CP)))


    def addSTO(signal_CP, sto):  # add some time offset
        return np.hstack([np.zeros(sto), signal_CP])
