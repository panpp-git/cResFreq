clc
clear all
close all
format long
rng('default');

L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[-5/L,0/L,5/L,5.7/L];

y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
amp(tgt_num)=10^(-10/20);
sig=zeros(1,L);
nfft=4096;
search_f=-0.5:1/nfft:0.5-1/nfft;
for i=1:tgt_num
    theta=2*pi*rand();
    sig=sig+amp(i)*exp(1i*theta)*exp(1i*w(i)*(0:(L-1)));
end
sig=sig/sqrt(mean(abs(sig.^2)));

SNR=20;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_20dB=sig*10^(SNR/20);
SNR=0;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_0dB=sig*10^(SNR/20)+noise;

%% dictionary construction(1)
nfft=4096;
dict_freq=-0.5:1/nfft:0.5-1/nfft;
t=0:(L-1);
dict=exp(1i*2*pi*dict_freq.'*t).';
[omp20]=(OMP(dict,noisedSig_20dB.',tgt_num));
[omp0]=(OMP(dict,noisedSig_0dB.',tgt_num));


%%
h=figure(1)
set(h,'position',[100 100 1000 600]);

for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),3,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
axis([-0.1 0.15 -40 3])
set(gca,'FontSize',20); 
set(get(gca,'XLabel'),'FontSize',20);
set(get(gca,'YLabel'),'FontSize',20);
normomp=20*log10(omp20/max(omp20)+1e-13);
plot(x_label,real(normomp).','b:.','linewidth',3);
hold on
normomp=20*log10(omp0/max(omp0)+1e-13);
plot(x_label,real(normomp).','k:.','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
ylabel('Normalized Power / dB');
xlabel('Normalized freq. / Hz');
grid on;
hold on;

% save omp.mat omp20 omp0