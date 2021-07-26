clc
clear all
close all
format long
rng('default');

L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
% w=2*pi*[0,1/L,5/L,5.7/L];
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
%% 
SNR=20;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_20dB=sig*10^(SNR/20)+noise;
SNR=0;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_0dB=sig*10^(SNR/20)+noise;

% load noisedSig_20dB
% load noisedSig_0dB

%% MUSIC
musicWinLen=L/2;
snapNum=L-musicWinLen+1;
x=0;
sk=zeros(musicWinLen,snapNum);
for i=1:snapNum
    B1=noisedSig_20dB(i:(i+musicWinLen-1));
    x=x+1;
    z1=B1(:);
    sk(:,x)=z1;
end
Rss= sk*sk'/snapNum;

[EV,D] = eig(Rss);
[EVA,I] = sort(diag(D).');
EV = fliplr(EV(:,I));
G = EV(:,tgt_num+1:end);
P_music_20dB=zeros(1,nfft);
for i=1:length(search_f)
    steeringVec=exp(1i*2*pi*search_f(i)*(0:musicWinLen-1)).';
    P_music_20dB(i)=1/(steeringVec'*G*G'*steeringVec);
end

%%
musicWinLen=L/2;
snapNum=L-musicWinLen+1;
x=0;
sk=zeros(musicWinLen,snapNum);
for i=1:snapNum
    B1=noisedSig_0dB(i:(i+musicWinLen-1));
    x=x+1;
    z1=B1(:);
    sk(:,x)=z1;
end
Rss= sk*sk'/snapNum;

[EV,D] = eig(Rss);
[EVA,I] = sort(diag(D).');
EV = fliplr(EV(:,I));
G = EV(:,tgt_num+1:end);
P_music_0dB=zeros(1,nfft);
for i=1:length(search_f)
    steeringVec=exp(1i*2*pi*search_f(i)*(0:musicWinLen-1)).';
    P_music_0dB(i)=1/(steeringVec'*G*G'*steeringVec);
end
%%
h=figure(2)
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
normMusic=10*log10(P_music_20dB/max(P_music_20dB)+1e-13);
plot(x_label,real(normMusic).','b:.','linewidth',3);
hold on
normMusic=10*log10(P_music_0dB/max(P_music_0dB)+1e-13);
plot(x_label,real(normMusic).','k:.','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
ylabel('Normalized Power / dB');
xlabel('Normalized freq. / Hz');
grid on;
hold on;

% save music.mat P_music_20dB P_music_0dB