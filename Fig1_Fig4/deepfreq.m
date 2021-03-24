clc
clear all


L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[0,1/L,5/L,5.7/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);

sig=zeros(1,L);
nfft=4096;
search_f=-0.5:1/nfft:0.5-1/nfft;
for i=1:tgt_num
    theta=2*pi*rand();
    sig=sig+amp(i)*exp(1i*theta)*exp(1i*w(i)*(0:(L-1)));
end
sig=sig/sqrt(mean(abs(sig.^2)));
%% 
% SNR=20;
% noisedSig_20dB=sig*10^(SNR/20)+wgn(size(sig,1),size(sig,2),0,'complex');
% SNR=5;
% noisedSig_0dB=sig*10^(SNR/20)+wgn(size(sig,1),size(sig,2),0,'complex');

load noisedSig_20dB
load noisedSig_0dB
%% DeepFreq
if ~exist('matlab_real1.h5','file')==0
    delete('matlab_real1.h5')
end
if ~exist('matlab_imag1.h5','file')==0
    delete('matlab_imag1.h5')   
end
mv=max(abs(noisedSig_20dB));
noisedSig=noisedSig_20dB/mv;
h5create('matlab_real1.h5','/matlab_real1',size(noisedSig));
h5write('matlab_real1.h5','/matlab_real1',real(noisedSig));
h5create('matlab_imag1.h5','/matlab_imag1',size(noisedSig));
h5write('matlab_imag1.h5','/matlab_imag1',imag(noisedSig));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe deepfreq_model.py');
h=figure(3)
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
    load data1_deepfreq.mat
    normDeepFreq=10*log10(data1_deepfreq.^2/max(data1_deepfreq.^2)+1e-13);
    plot(x_label,real(normDeepFreq),'b:.','linewidth',3);
    axis([-0.08 0.22 -40 3])
    ylabel('Normalized PSD / dB');
    xlabel('Normalized freq. / Hz');
    grid on;
    hold on
end
%% 
if ~exist('matlab_real1.h5','file')==0
    delete('matlab_real1.h5')
end
if ~exist('matlab_imag1.h5','file')==0
    delete('matlab_imag1.h5')   
end
mv=max(abs(noisedSig_0dB));
noisedSig=noisedSig_0dB/mv;
h5create('matlab_real1.h5','/matlab_real1',size(noisedSig));
h5write('matlab_real1.h5','/matlab_real1',real(noisedSig));
h5create('matlab_imag1.h5','/matlab_imag1',size(noisedSig));
h5write('matlab_imag1.h5','/matlab_imag1',imag(noisedSig));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe deepfreq_model.py');
if flag==0
    load data1_deepfreq.mat
    normDeepFreq=10*log10(data1_deepfreq.^2/max(data1_deepfreq.^2)+1e-13);
    plot(x_label,real(normDeepFreq),'k:.','linewidth',3);

    legend('DeepFreq, SNR = 30dB','DeepFreq, SNR = 10dB');
    ylabel('Normalized Power / dB');
    xlabel('Normalized freq. / Hz');
    grid on;
end