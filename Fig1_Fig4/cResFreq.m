clc
clear all
close all
format long
rng('default');

L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[-5/L,0/L,5/L,5.7/L,8/L,13/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
amp(tgt_num-2)=10^(-10/20);
amp(tgt_num)=10^(-20/20);
sig=zeros(1,L);
nfft=4096;
search_f=-0.5:1/nfft:0.5-1/nfft;
for i=1:tgt_num
    theta=2*pi*rand();
    sig=sig+amp(i)*exp(1i*theta)*exp(1i*w(i)*(0:(L-1)));
end
sig=sig/sqrt(mean(abs(sig.^2)));
%% 
SNR=20;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_20dB=sig*10^(SNR/20)+noise;
SNR=0;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_0dB=sig*10^(SNR/20)+noise;

% load noisedSig_20dB
% load noisedSig_0dB
%% cResFreq
delete('data1_resfreq.mat')
if ~exist('matlab_real2.h5','file')==0
    delete('matlab_real2.h5')
end
if ~exist('matlab_imag2.h5','file')==0
    delete('matlab_imag2.h5')   
end
mv=max(abs(noisedSig_20dB));
noisedSig=noisedSig_20dB/mv;
h5create('matlab_real2.h5','/matlab_real2',size(noisedSig));
h5write('matlab_real2.h5','/matlab_real2',real(noisedSig));
h5create('matlab_imag2.h5','/matlab_imag2',size(noisedSig));
h5write('matlab_imag2.h5','/matlab_imag2',imag(noisedSig));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py');
h=figure(4)
set(h,'position',[100 100 1000 600]);

for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),3,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end

set(gca,'FontSize',20); 
set(get(gca,'XLabel'),'FontSize',20);
set(get(gca,'YLabel'),'FontSize',20);
if flag==0
    load data1_resfreq.mat
 
    normDeepFreq=10*log10(data1_resfreq.^2/max(data1_resfreq.^2)+1e-13);
    plot(x_label,real(normDeepFreq),'b:.','linewidth',3);
    axis([-0.1 0.25 -60 3])
    ylabel('Normalized PSD / dB');
    xlabel('Normalized freq. / Hz');

    hold on
end

%% 
delete('data1_resfreq.mat')
if ~exist('matlab_real2.h5','file')==0
    delete('matlab_real2.h5')
end
if ~exist('matlab_imag2.h5','file')==0
    delete('matlab_imag2.h5')   
end
mv=max(abs(noisedSig_0dB));
noisedSig0=noisedSig_0dB/mv;
h5create('matlab_real2.h5','/matlab_real2',size(noisedSig0));
h5write('matlab_real2.h5','/matlab_real2',real(noisedSig0));
h5create('matlab_imag2.h5','/matlab_imag2',size(noisedSig0));
h5write('matlab_imag2.h5','/matlab_imag2',imag(noisedSig0));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py');
if flag==0
    
    load data1_resfreq.mat

    normDeepFreq0=10*log10(data1_resfreq.^2/max(data1_resfreq.^2)+1e-13);
    plot(x_label,real(normDeepFreq0),'k:.','linewidth',3);

    legend('SNR = 20dB','SNR = 0dB');
    ylabel('Normalized Power / dB');
    xlabel('Normalized freq. / Hz');
    grid on;
end