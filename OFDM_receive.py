import numpy as np
from OFDM_transmit import OFDM_Tx


class OFDM_Rx(OFDM_Tx):
    
    def complex2bits(self, complex_stream): 
        OFDM_RX_noCP = self.remove_CP(complex_stream)
        rx_after_fft = self.FFT_signal(OFDM_RX_noCP)
        equalized_signal = self.channel_estimation_equalization(rx_after_fft)
        rx_after_fft_data_subcarriers = self.get_payload(equalized_signal)
        PS_est, hardDecision = self.Demapping(rx_after_fft_data_subcarriers)
        return  self.PS(PS_est)
    
    def de_mapping(self):
        
        if (self.modulation == "bpsk"):
            de_mapping_table =  self.mapping_bpsk()
            
        if (self.modulation == "qpsk"):
            de_mapping_table = self.mapping_qpsk()
        
        if (self.modulation == "16qam"):
            de_mapping_table =  self.mapping_16qam()
            
        if (self.modulation == "64qam"):
            de_mapping_table = self.mapping_64qam()
        
        return de_mapping_table
    
    def remove_CP(self, channel_output_signal):
        return channel_output_signal[self.CP:(self.CP + self.N_FFT)]
    
    def FFT_signal(self, rx_time_domain_noCP):
        return np.fft.fft(np.fft.ifftshift(rx_time_domain_noCP))

    def channel_estimation_equalization(self, signal_noisy): # signal_noisy has been through the channel before having noise added to it
        rx_pilots = signal_noisy[self.pilot_subcarriers] # receive signals from pilots subcarriers
        length_rx_pilots = len(rx_pilots)
        channel_estimates = np.zeros((length_rx_pilots,), dtype=complex)
        channel_estimate_interp_mag = np.zeros((self.N_FFT,), dtype=float) # K is the total number of subcarriers
        channel_estimate_interp_phase = np.zeros((self.N_FFT,), dtype=float)
        channel_estimate_interp = np.zeros((self.N_FFT,), dtype=complex)
        for i in range(length_rx_pilots):
            channel_estimates[i] = rx_pilots[i]/self.pilot_symbols[i]
     # interpolating rx_pilots for all subcarriers
        bounds = np.linspace(min(self.pilot_subcarriers), max(self.pilot_subcarriers), self.N_FFT)
        channel_estimate_interp_mag =  np.interp(bounds, self.pilot_subcarriers, np.abs(channel_estimates))
        for j in range(self.N_FFT):
            if (channel_estimate_interp_mag[j] == 0):
                channel_estimate_interp_mag[j] = np.mean(channel_estimate_interp_mag)
                assert channel_estimate_interp_mag[j] != 0, f"division by 0 from channel values interpolation"
        channel_estimate_interp_phase =  np.interp(bounds, self.pilot_subcarriers, np.angle(channel_estimates))
        #channel_estimate_interp = channel_estimate_interp_mag * np.exp(1j*channel_estimate_interp_phase)
        channel_estimate_interp =  np.interp(bounds, self.pilot_subcarriers, channel_estimates) # complex interpolation
        #channel_estimate_interp[self.pilot_subcarriers] = channel_estimates

    
        return signal_noisy/channel_estimate_interp
    
    def get_payload(self, after_fft):
        return after_fft[self.data_subcarriers]
    
    def Demapping(self, rx_constellation):
        de_mapping_table = {v : k for k, v in self.de_mapping().items()}
        constellation = np.array([x for x in de_mapping_table.keys()])
        dists = abs(rx_constellation.reshape((-1,1)) - constellation.reshape((1,-1)))
        const_index = dists.argmin(axis=1)
        hardDecision = constellation[const_index]
        return np.vstack([de_mapping_table[C] for C in hardDecision]), hardDecision
    
    def PS(self, bits):
        return bits.reshape((-1,))
