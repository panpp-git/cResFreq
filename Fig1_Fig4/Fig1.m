clc
clear all
close all

L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[0,1/L,5/L,5.7/L,8/L];
% w=2*pi*[0];
% w=2*pi*(0.105*rand(1,5)+(-0.05));
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
amp(tgt_num)=10^(-25/20);
sig=zeros(1,L);

SNR=20;
for i=1:tgt_num
    theta=2*pi*rand();
    sig=sig+amp(i)*exp(1i*theta)*exp(1i*w(i)*(0:(L-1)));
end
sig=sig/sqrt(mean(abs(sig.^2)));
noisedSig=sig*10^(SNR/20)+wgn(size(sig,1),size(sig,2),0,'complex');

%% periodogram
win=hamming(length(noisedSig)).';
periodogram_win=abs(fftshift(fft(noisedSig.*win,nfft))).^2/nfft;
periodogram=abs(fftshift(fft(noisedSig,nfft))).^2/nfft;
h=figure()
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
normPeriodogram=10*log10(periodogram/max(periodogram)+1e-13);
plot(x_label,normPeriodogram,'m:.','linewidth',3);
hold on;
normPeriodogram_win=10*log10(periodogram_win/max(periodogram_win)+1e-13);
plot(x_label,normPeriodogram_win,'r:.','linewidth',3);
hold on;

%% Capon
caponWinLen=L/2;
snapNum=L-caponWinLen+1;
x=0;
sk=zeros(caponWinLen,snapNum);
for i=1:snapNum
    B1=noisedSig(i:(i+caponWinLen-1));
    x=x+1;
    z1=B1(:);
    sk(:,x)=z1;
end

Rss= sk*sk'/snapNum;
invR=inv(Rss);
P_capon=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
for i=1:length(search_f)
    steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
    P_capon(i)=1/(steeringVec'*invR*steeringVec);
end

normCapon=10*log10(P_capon/max(P_capon)+1e-13);
plot(x_label,real(normCapon).','g:.','linewidth',3);
hold on;

%% MUSIC
musicWinLen=L/2;
[EV,D] = eig(Rss);
[EVA,I] = sort(diag(D).');
EV = fliplr(EV(:,I));
G = EV(:,tgt_num+1:end);
P_music=zeros(1,nfft);
for i=1:length(search_f)
    steeringVec=exp(1i*2*pi*search_f(i)*(0:musicWinLen-1)).';
    P_music(i)=1/(steeringVec'*G*G'*steeringVec);
end

normMusic=10*log10(P_music/max(P_music)+1e-13);
plot(x_label,real(normMusic),'b:.','linewidth',3);
hold on;

%% DeepFreq
if ~exist('matlab_real1.h5','file')==0
    delete('matlab_real1.h5')
end
if ~exist('matlab_imag1.h5','file')==0
    delete('matlab_imag1.h5')   
end
mv=max(abs(noisedSig));
noisedSig=noisedSig/mv;
h5create('matlab_real1.h5','/matlab_real1',size(noisedSig));
h5write('matlab_real1.h5','/matlab_real1',real(noisedSig));
h5create('matlab_imag1.h5','/matlab_imag1',size(noisedSig));
h5write('matlab_imag1.h5','/matlab_imag1',imag(noisedSig));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe deepfreq_model.py');
if flag==0

    load data1_deepfreq.mat
    normDeepFreq=10*log10(data1_deepfreq.^2/max(data1_deepfreq.^2)+1e-13);
    plot(x_label,real(normDeepFreq),'k:.','linewidth',3);
    axis([-0.1 0.2 -80 3])
    legend('Periodogram','Capon','MUSIC','DeepFreq');
    title('Clearly Separated Freqs., SNR = 0 dB');
    ylabel('Normalized PSD in dB');
    xlabel('Normalized frequency in Hz');
    grid on;
end
%% ResFreq
if ~exist('matlab_real2.h5','file')==0
    delete('matlab_real2.h5')
end
if ~exist('matlab_imag2.h5','file')==0
    delete('matlab_imag2.h5')   
end
h5create('matlab_real2.h5','/matlab_real2',size(noisedSig));
h5write('matlab_real2.h5','/matlab_real2',real(noisedSig));
h5create('matlab_imag2.h5','/matlab_imag2',size(noisedSig));
h5write('matlab_imag2.h5','/matlab_imag2',imag(noisedSig));
system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py')
% figure;
load data1_resfreq.mat
normResFreq=10*log10(data1_resfreq.^2/max(data1_resfreq.^2)+1e-13);

plot(x_label,real(normResFreq));
axis([-0.1 0.2 -60 3])
xlabel('Normalized frequency in Hz');
ylabel('Normalized PSD in dB');
