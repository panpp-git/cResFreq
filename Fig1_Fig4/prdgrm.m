clc
clear all
close all

L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[0,1.5/L,5/L,6.5/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
amp(tgt_num)=10^(-15/20);
sig=zeros(1,L);

for i=1:tgt_num
    theta=2*pi*rand();
    sig=sig+amp(i)*exp(1i*theta)*exp(1i*w(i)*(0:(L-1)));
end
sig=sig/sqrt(mean(abs(sig.^2)));
%% 
SNR=30;
noisedSig_20dB=sig*10^(SNR/20)+wgn(size(sig,1),size(sig,2),0,'complex');

%% periodogram
win=hamming(length(noisedSig_20dB)).';
periodogram_win_20dB=abs(fftshift(fft(noisedSig_20dB.*win,nfft))).^2/nfft;
periodogram_nowin_20dB=abs(fftshift(fft(noisedSig_20dB,nfft))).^2/nfft;

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
normPeriodogram=10*log10(periodogram_nowin_20dB/max(periodogram_nowin_20dB)+1e-13);
plot(x_label,normPeriodogram,'b:.','linewidth',3);
hold on;
normPeriodogram_win=10*log10(periodogram_win_20dB/max(periodogram_win_20dB)+1e-13);
plot(x_label,normPeriodogram_win,'k:.','linewidth',3);
hold on;
axis([-0.08 0.22 -40 3])
hh=legend('no window func.','with window func.');
set(hh,'Fontsize',20)
ylabel('Normalized Power / dB');
xlabel('Normalized freq. / Hz');
grid on;
