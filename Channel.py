import numpy as np

#-------------------------------Channel + noise function-------------------------------

class channel_noise:
    
    def __init__(self):
        pass
    
       
    def apply_channel(self, signal_CP, rician_factor, snr_dB): # channel comes from one channel model like Rayleigh, Rician, nakagami, etc
        fading_channel = self.rician_fading(rician_factor) # rician = 0, 3, 10, 20
        convolved = np.convolve(signal_CP, fading_channel)
        
        return self.awgn(convolved, snr_dB)    
        
    def awgn(self, after_channel_signal, snr_dB): # with noise only
        signal_power = np.mean(abs(after_channel_signal)**2)
        sigma2 = signal_power * 10**(-snr_dB/10)  # calculate noise power based on signal power and SNR
    
        # Generate complex noise with given variance
        noise = np.sqrt(sigma2/2) * (np.random.randn(*after_channel_signal.shape)+1j*np.random.randn(*after_channel_signal.shape))
        return  after_channel_signal + noise

#---------------------------------Channel --------------
    def rayleigh_fading(self): # apply channel to signal_CP
    # channel_length : number of discrete-time instants
    #......Basic code from PySDR.com................
        # based on Clarke's sum of sinusoids
        v_mph = 0.01 # 42.50179 # velocity in miles per hour
        center_freq = 2.4e9 # 5.8e9 WiFi 
        Fs = 1e4 # sample rate of simulation, sampling period Ts = 1/Fs
        N = 3 # number of sinusoid to sum

        One_mile = 1609.34  # 1 mile in meters
        v = v_mph * One_mile /3600 # velocity in m/s
        speed_of_light = 3e8
        fd = v*center_freq/speed_of_light # max Doppler shift

        Ts = 1/Fs
        #sim_time = 100000 # number of discrete-time instants
        #t = np.arange(0, sim_time * Ts, Ts)
        t = np.arange(0, 1, Ts)
        z = np.zeros((len(t)), dtype=complex) # 
        for i in range(N):
            alpha = (np.random.rand() - 0.5) * 2 * np.pi
            phi = (np.random.rand() - 0.5)* 2 * np.pi
            z.real += np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(alpha) + phi)
            z.imag += np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(alpha) + phi)
        z = (1/np.sqrt(N))* z
        
        return z

    def rician_fading(self, rician_factor): # apply channel to signal_CP
   
    #rician_factor = np.array([0, 3, 5, 10, 20]) # for the factor 0, rician becomes Rayleigh
        rayleigh_fading_channel = self.rayleigh_fading() 
    #..........Rician fading from DigiCommPy.............
        K = 10**(rician_factor/10)  # factor in a linear scale
        mu = np.sqrt(K/(2*(K+1)))  # mean
        sigma = np.sqrt(1/(2*(K+1)))
        h = sigma *  rayleigh_fading_channel
        return h


   

   
     
