clc
clear
close all
rng('default')

h=figure();
set(h,'position',[100 100 1600 400]);

ha=tight_subplot(1,4,[0.03 0.03],[.2 .08],[.05 .01])
fsz=13;
L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[0,4/L];
% w=2*pi*[0,1.2/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
% amp(tgt_num)=10^(-15/20);
sig=zeros(1,L);

SNR=30;
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
axes(ha(1))

for i=1:tgt_num
    h1=stem(y(i),-100,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),10,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end


set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% normPeriodogram=10*log10(periodogram/max(periodogram)+1e-13);
% plot(x_label,normPeriodogram,'m:.','linewidth',3);
% hold on;
normPeriodogram_win=10*log10(periodogram_win/max(periodogram_win)+1e-13);
plot(x_label,normPeriodogram_win,'m-.','linewidth',2);
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
% invR=inv(Rss);
% P_capon=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
% for i=1:length(search_f)
%     steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
%     P_capon(i)=1/(steeringVec'*invR*steeringVec);
% end
% 
% normCapon=10*log10(P_capon/max(P_capon)+1e-13);
% plot(x_label,real(normCapon).','g:.','linewidth',2);
% hold on;

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
plot(x_label,real(normMusic),'-.','color','#4dbeee','linewidth',2);
hold on;

%% OMP
nfft=4096;
L=tgt_num;
dict_freq=-0.5:1/nfft:0.5-1/nfft;
t=0:63;
dict=exp(1i*2*pi*dict_freq.'*t).';
for indix=1:size(noisedSig,1)
% for indix=1:1
    [A]=(OMP(dict,noisedSig(indix,:).',L));
    xx=abs(full(A));
    P_omp(indix,:)=abs(xx)/max(abs(xx));
end

plot(x_label,10*log10(abs(P_omp).^2+1e-13),'-.','color','#006400','linewidth',2);

%% CVNN by AH
% Ns=1;
% Nsnap=8;
% mv=max(abs(noisedSig));
% RPP=noisedSig/mv;
% t=0:63;
% nfft=4096;
% len=size(RPP,2);
% snapLen=len-Nsnap+1;
% freqs = -0.5:1/nfft:0.5-1/nfft;
% final_ret=zeros(size(RPP,1),length(freqs));
% for indix=1:size(RPP,1)
% % for indix=1:1
%     ss=RPP(indix,:);
%     zI=zeros(Ns,Nsnap,snapLen);
%     for si=1:Ns
%         for i=1:Nsnap
%             zI(si,i,:)=ss(si,i:i+snapLen-1);
%         end
%     end
%     zO_teach_set=zeros(Ns,snapLen);
%     zI_set=zI;
%     [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);
% 
% 
%     zI=zeros(length(freqs),Nsnap,snapLen);
%     for tgt=1:length(freqs)
%         zIk=exp(1i*(2*pi*freqs(tgt)*t));
%         zIk=zIk/max(abs(zIk));
%         for i=1:Nsnap
%             zI(tgt,i,:)=zIk(i:i+snapLen-1);
%         end
%     end
% 
%     load wgt_freq22.mat
%     zI_set = zI;
%     [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
%     final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
%     P_ah(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
% end
% save P_ah_axes1.mat P_ah
load P_ah_axes1.mat
plot(x_label,10*log10(abs(P_ah).^2),'-.','color','#A0522D','linewidth',2);

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
    plot(x_label,real(normDeepFreq),'-.','color','#edb120','linewidth',2);
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

plot(x_label,real(normResFreq),'k-.','linewidth',2);
% title('Clearly Separated Freqs., SNR = 30 dB');
legend('periodogram','MUSIC','OMP','CVNN','DeepFreq','cResFreq');
xlabel({'f / Hz';'(a)'});
ylabel('Normalized Power / dB');
xlim([-0.03 0.12])
ylim([-100 10])
%% axes(2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng('default')
L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
w=2*pi*[0,4/L];
% w=2*pi*[0,1.2/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
% amp(tgt_num)=10^(-15/20);
sig=zeros(1,L);

SNR=0;
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
axes(ha(2))

for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end


set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% normPeriodogram=10*log10(periodogram/max(periodogram)+1e-13);
% plot(x_label,normPeriodogram,'m:.','linewidth',3);
% hold on;
normPeriodogram_win=10*log10(periodogram_win/max(periodogram_win)+1e-13);
plot(x_label,normPeriodogram_win,'m-.','linewidth',2);
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
% invR=inv(Rss);
% P_capon=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
% for i=1:length(search_f)
%     steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
%     P_capon(i)=1/(steeringVec'*invR*steeringVec);
% end
% 
% normCapon=10*log10(P_capon/max(P_capon)+1e-13);
% plot(x_label,real(normCapon).','g:.','linewidth',2);
% hold on;

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
plot(x_label,real(normMusic),'-.','color','#4dbeee','linewidth',2);
hold on;

%% OMP
nfft=4096;
L=tgt_num;
dict_freq=-0.5:1/nfft:0.5-1/nfft;
t=0:63;
dict=exp(1i*2*pi*dict_freq.'*t).';
for indix=1:size(noisedSig,1)
% for indix=1:1
    [A]=(OMP(dict,noisedSig(indix,:).',L));
    xx=abs(full(A));
    P_omp(indix,:)=abs(xx)/max(abs(xx));
end

plot(x_label,10*log10(abs(P_omp).^2+1e-13),'-.','color','#006400','linewidth',2);

%% CVNN by AH
% Ns=1;
% Nsnap=8;
% mv=max(abs(noisedSig));
% RPP=noisedSig/mv;
% t=0:63;
% nfft=4096;
% len=size(RPP,2);
% snapLen=len-Nsnap+1;
% freqs = -0.5:1/nfft:0.5-1/nfft;
% final_ret=zeros(size(RPP,1),length(freqs));
% for indix=1:size(RPP,1)
% % for indix=1:1
%     ss=RPP(indix,:);
%     zI=zeros(Ns,Nsnap,snapLen);
%     for si=1:Ns
%         for i=1:Nsnap
%             zI(si,i,:)=ss(si,i:i+snapLen-1);
%         end
%     end
%     zO_teach_set=zeros(Ns,snapLen);
%     zI_set=zI;
%     [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);
% 
% 
%     zI=zeros(length(freqs),Nsnap,snapLen);
%     for tgt=1:length(freqs)
%         zIk=exp(1i*(2*pi*freqs(tgt)*t));
%         zIk=zIk/max(abs(zIk));
%         for i=1:Nsnap
%             zI(tgt,i,:)=zIk(i:i+snapLen-1);
%         end
%     end
% 
%     load wgt_freq22.mat
%     zI_set = zI;
%     [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
%     final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
%     P_ah(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
% end
% save P_ah_axes2.mat P_ah
load P_ah_axes2.mat
plot(x_label,10*log10(abs(P_ah).^2),'-.','color','#A0522D','linewidth',2);
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
    plot(x_label,real(normDeepFreq),'-.','color','#edb120','linewidth',2);
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

plot(x_label,real(normResFreq),'k-.','linewidth',2);
axis([-0.03 0.12 -40 5])
% title('Clearly Separated Freqs., SNR = 0 dB');
legend('periodogram','MUSIC','OMP','CVNN','DeepFreq','cResFreq');
xlabel({'f / Hz';'(b)'});
% ylabel('Normalized Power / dB');
% ylim([-40 5])
%% axes(3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L=64;
rng(456)
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
% w=2*pi*[0,4/L];
w=2*pi*[0,1.2/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
% amp(tgt_num)=10^(-15/20);
sig=zeros(1,L);

SNR=30;
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

axes(ha(3))
for i=1:tgt_num
    h1=stem(y(i),-100,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),10,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end


set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% normPeriodogram=10*log10(periodogram/max(periodogram)+1e-13);
% plot(x_label,normPeriodogram,'m:.','linewidth',3);
% hold on;
normPeriodogram_win=10*log10(periodogram_win/max(periodogram_win)+1e-13);
plot(x_label,normPeriodogram_win,'m-.','linewidth',2);
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
% invR=inv(Rss);
% P_capon=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
% for i=1:length(search_f)
%     steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
%     P_capon(i)=1/(steeringVec'*invR*steeringVec);
% end
% 
% normCapon=10*log10(P_capon/max(P_capon)+1e-13);
% plot(x_label,real(normCapon).','g:.','linewidth',2);
% hold on;

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
plot(x_label,real(normMusic),'-.','color','#4dbeee','linewidth',2);
hold on;
%% OMP
nfft=4096;
L=tgt_num;
dict_freq=-0.5:1/nfft:0.5-1/nfft;
t=0:63;
dict=exp(1i*2*pi*dict_freq.'*t).';
for indix=1:size(noisedSig,1)
% for indix=1:1
    [A]=(OMP(dict,noisedSig(indix,:).',L));
    xx=abs(full(A));
    P_omp(indix,:)=abs(xx)/max(abs(xx));
end

plot(x_label,10*log10(abs(P_omp).^2+1e-13),'-.','color','#006400','linewidth',2);

%% CVNN by AH
% Ns=1;
% Nsnap=8;
% mv=max(abs(noisedSig));
% RPP=noisedSig/mv;
% t=0:63;
% nfft=4096;
% len=size(RPP,2);
% snapLen=len-Nsnap+1;
% freqs = -0.5:1/nfft:0.5-1/nfft;
% final_ret=zeros(size(RPP,1),length(freqs));
% for indix=1:size(RPP,1)
% % for indix=1:1
%     ss=RPP(indix,:);
%     zI=zeros(Ns,Nsnap,snapLen);
%     for si=1:Ns
%         for i=1:Nsnap
%             zI(si,i,:)=ss(si,i:i+snapLen-1);
%         end
%     end
%     zO_teach_set=zeros(Ns,snapLen);
%     zI_set=zI;
%     [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);
% 
% 
%     zI=zeros(length(freqs),Nsnap,snapLen);
%     for tgt=1:length(freqs)
%         zIk=exp(1i*(2*pi*freqs(tgt)*t));
%         zIk=zIk/max(abs(zIk));
%         for i=1:Nsnap
%             zI(tgt,i,:)=zIk(i:i+snapLen-1);
%         end
%     end
% 
%     load wgt_freq22.mat
%     zI_set = zI;
%     [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
%     final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
%     P_ah(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
% end
% save P_ah_axes3.mat P_ah
load P_ah_axes3.mat
plot(x_label,10*log10(abs(P_ah).^2),'-.','color','#A0522D','linewidth',2);
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
    plot(x_label,real(normDeepFreq),'-.','color','#edb120','linewidth',2);
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

plot(x_label,real(normResFreq),'k-.','linewidth',2);
axis([-0.03 0.08 -100 10])
% title('Closely Separated Freqs., SNR = 30 dB');
legend('periodogram','MUSIC','OMP','CVNN','DeepFreq','cResFreq');
xlabel({'f / Hz';'(c)'});
% ylabel('Normalized Power / dB');
% ylim([-100 5])
%% axes(4)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(123)
L=64;
nfft=L*64;
x_label=-0.5:1/nfft:0.5-1/nfft;
% w=2*pi*[0,4/L];
w=2*pi*[0,1.2/L];
y=w/2/pi;
tgt_num=length(w);
amp=ones(1,tgt_num);
% amp(tgt_num)=10^(-15/20);
sig=zeros(1,L);

SNR=0;
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

axes(ha(4))
for i=1:tgt_num
    h1=stem(y(i),-40,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end


set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% normPeriodogram=10*log10(periodogram/max(periodogram)+1e-13);
% plot(x_label,normPeriodogram,'m:.','linewidth',3);
% hold on;
normPeriodogram_win=10*log10(periodogram_win/max(periodogram_win)+1e-13);
plot(x_label,normPeriodogram_win,'m-.','linewidth',2);
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
% invR=inv(Rss);
% P_capon=zeros(1,nfft);
search_f=-0.5:1/nfft:0.5-1/nfft;
% for i=1:length(search_f)
%     steeringVec=exp(1i*2*pi*search_f(i)*(0:caponWinLen-1)).';
%     P_capon(i)=1/(steeringVec'*invR*steeringVec);
% end
% 
% normCapon=10*log10(P_capon/max(P_capon)+1e-13);
% plot(x_label,real(normCapon).','g:.','linewidth',2);
% hold on;

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
plot(x_label,real(normMusic),'-.','color','#4dbeee','linewidth',2);
hold on;
%% OMP
nfft=4096;
L=tgt_num;
dict_freq=-0.5:1/nfft:0.5-1/nfft;
t=0:63;
dict=exp(1i*2*pi*dict_freq.'*t).';
for indix=1:size(noisedSig,1)
% for indix=1:1
    [A]=(OMP(dict,noisedSig(indix,:).',L));
    xx=abs(full(A));
    P_omp(indix,:)=abs(xx)/max(abs(xx));
end

plot(x_label,10*log10(abs(P_omp).^2+1e-13),'-.','color','#006400','linewidth',2);

%% CVNN by AH
% Ns=1;
% Nsnap=8;
% mv=max(abs(noisedSig));
% RPP=noisedSig/mv;
% t=0:63;
% nfft=4096;
% len=size(RPP,2);
% snapLen=len-Nsnap+1;
% freqs = -0.5:1/nfft:0.5-1/nfft;
% final_ret=zeros(size(RPP,1),length(freqs));
% for indix=1:size(RPP,1)
% % for indix=1:1
%     ss=RPP(indix,:);
%     zI=zeros(Ns,Nsnap,snapLen);
%     for si=1:Ns
%         for i=1:Nsnap
%             zI(si,i,:)=ss(si,i:i+snapLen-1);
%         end
%     end
%     zO_teach_set=zeros(Ns,snapLen);
%     zI_set=zI;
%     [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);
% 
% 
%     zI=zeros(length(freqs),Nsnap,snapLen);
%     for tgt=1:length(freqs)
%         zIk=exp(1i*(2*pi*freqs(tgt)*t));
%         zIk=zIk/max(abs(zIk));
%         for i=1:Nsnap
%             zI(tgt,i,:)=zIk(i:i+snapLen-1);
%         end
%     end
% 
%     load wgt_freq22.mat
%     zI_set = zI;
%     [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
%     final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
%     P_ah(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
% end
% save P_ah_axes4.mat P_ah
load P_ah_axes4.mat
plot(x_label,10*log10(abs(P_ah).^2),'-.','color','#A0522D','linewidth',2);
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
    plot(x_label,real(normDeepFreq),'-.','color','#edb120','linewidth',2);
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

plot(x_label,real(normResFreq),'k-.','linewidth',2);
axis([-0.03 0.08 -40 5])
% title('Closely Separated Freqs., SNR = 0 dB');
legend('periodogram','MUSIC','OMP','CVNN','DeepFreq','cResFreq');
xlabel({'f / Hz';'(d)'});
% ylabel('Normalized Power / dB');
% ylim([-40 5])