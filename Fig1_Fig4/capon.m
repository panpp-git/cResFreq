clc
clear all
close all

L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[0,1/L,5/L,5.7/L];
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
% SNR=30;
% noisedSig_20dB=sig*10^(SNR/20)+wgn(size(sig,1),size(sig,2),0,'complex');
% SNR=10;
% noisedSig_0dB=sig*10^(SNR/20)+wgn(size(sig,1),size(sig,2),0,'complex');
% save noisedSig_20dB.mat noisedSig_20dB
% save noisedSig_0dB.mat noisedSig_0dB
load noisedSig_20dB
load noisedSig_0dB
%% Capon
caponWinLen=L/2;
snapNum=L-caponWinLen+1;
x=0;
sk=zeros(caponWinLen,snapNum);
for i=1:snapNum
    B1=noisedSig_20dB(i:(i+caponWinLen-1));
    x=x+1;
    z1=B1(:);
    sk(:,x)=z1;
end

Rss= sk*sk'/snapNum;
invR=inv(Rss);
P_capon_20dB=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
for i=1:length(search_f)
    steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
    P_capon_20dB(i)=1/(steeringVec'*invR*steeringVec);
end

%%
caponWinLen=L/2;
snapNum=L-caponWinLen+1;
x=0;
sk=zeros(caponWinLen,snapNum);
for i=1:snapNum
    B1=noisedSig_0dB(i:(i+caponWinLen-1));
    x=x+1;
    z1=B1(:);
    sk(:,x)=z1;
end

Rss= sk*sk'/snapNum;
invR=inv(Rss);
P_capon_0dB=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
for i=1:length(search_f)
    steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
    P_capon_0dB(i)=1/(steeringVec'*invR*steeringVec);
end
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
axis([-0.08 0.22 -40 3])
set(gca,'FontSize',20); 
set(get(gca,'XLabel'),'FontSize',20);
set(get(gca,'YLabel'),'FontSize',20);
normCapon=10*log10(P_capon_20dB/max(P_capon_20dB)+1e-13);
plot(x_label,real(normCapon).','b:.','linewidth',3);
hold on
normCapon=10*log10(P_capon_0dB/max(P_capon_0dB)+1e-13);
plot(x_label,real(normCapon).','k:.','linewidth',3);
legend('Capon, SNR = 30dB','Capon, SNR = 10dB');
ylabel('Normalized Power / dB');
xlabel('Normalized freq. / Hz');
grid on;
hold on;

