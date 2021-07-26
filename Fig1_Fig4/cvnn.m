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
noisedSig_20dB=sig*10^(SNR/20)+noise;
SNR=0;
noise=wgn(size(sig,1),size(sig,2),0,'complex');
noisedSig_0dB=sig*10^(SNR/20)+noise;

%%
Ns=1;
Nsnap=8;
mv=max(abs(noisedSig_20dB));
RPP=noisedSig_20dB/mv;
t=0:63;
nfft=4096;
len=size(RPP,2);
snapLen=len-Nsnap+1;
freqs = -0.5:1/nfft:0.5-1/nfft;

for indix=1:size(RPP,1)
% for indix=1:1
    ss=RPP(indix,:);
    zI=zeros(Ns,Nsnap,snapLen);
    for si=1:Ns
        for i=1:Nsnap
            zI(si,i,:)=ss(si,i:i+snapLen-1);
        end
    end
    zO_teach_set=zeros(Ns,snapLen);
    zI_set=zI;
    [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);


    zI=zeros(length(freqs),Nsnap,snapLen);
    for tgt=1:length(freqs)
        zIk=exp(1i*(2*pi*freqs(tgt)*t));
        zIk=zIk/max(abs(zIk));
        for i=1:Nsnap
            zI(tgt,i,:)=zIk(i:i+snapLen-1);
        end
    end

    load wgt_freq22.mat
    zI_set = zI;
    [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
    final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
    P_ah20(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
end

Ns=1;
Nsnap=8;
mv=max(abs(noisedSig_0dB));
RPP=noisedSig_0dB/mv;
t=0:63;
nfft=4096;
len=size(RPP,2);
snapLen=len-Nsnap+1;
freqs = -0.5:1/nfft:0.5-1/nfft;
final_ret20=zeros(size(RPP,1),length(freqs));
for indix=1:size(RPP,1)
% for indix=1:1
    ss=RPP(indix,:);
    zI=zeros(Ns,Nsnap,snapLen);
    for si=1:Ns
        for i=1:Nsnap
            zI(si,i,:)=ss(si,i:i+snapLen-1);
        end
    end
    zO_teach_set=zeros(Ns,snapLen);
    zI_set=zI;
    [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);


    zI=zeros(length(freqs),Nsnap,snapLen);
    for tgt=1:length(freqs)
        zIk=exp(1i*(2*pi*freqs(tgt)*t));
        zIk=zIk/max(abs(zIk));
        for i=1:Nsnap
            zI(tgt,i,:)=zIk(i:i+snapLen-1);
        end
    end

    load wgt_freq22.mat
    zI_set = zI;
    [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
    final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
    P_ah0(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
end

%%
h=figure(1)
set(h,'position',[100 100 1000 600]);

for i=1:tgt_num
    h1=stem(y(i),-40,'r-','Marker','none','linewidth',2);
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
normah=20*log10(P_ah20/max(P_ah20)+1e-13);
plot(x_label,real(normah).','b:.','linewidth',3);
hold on
normah=20*log10(P_ah0/max(P_ah0)+1e-13);
plot(x_label,real(normah).','k:.','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
ylabel('Normalized Power / dB');
xlabel('Normalized freq. / Hz');
grid on;
hold on;

% save p_ah.mat P_ah20 P_ah0