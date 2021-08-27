clc
clear
close all

load b727r.mat
X=X.';
X=ifft(X,[],2);
fsz=13;
dr=3e8/2/300e6;
r=size(X,2)*dr;
r_label=0:r/4096:r-r/4096;
figure;
win=ones(size(X,1),1)*hamming(size(X,2)).';
spc=20*log10(fftshift(abs(fft2(X.*win,size(X,1),4096)),1));
imagesc(r_label,1:size(X,1),spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('Boeing 727');
xlabel('Relative Range / m');
ylabel('Doppler Cell')

sig=X(1:2:end,1:2:end);

for i=1:size(sig,1)
    RPP(i,:)=sig(i,:)/max(abs(sig(i,:)));
end
bz=size(sig,1);

if ~exist('matlab_real2.h5','file')==0
    delete('matlab_real2.h5')
end

if ~exist('matlab_imag2.h5','file')==0
    delete('matlab_imag2.h5')   
end


if ~exist('bz.h5','file')==0
    delete('bz.h5')   
end

h5create('matlab_real2.h5','/matlab_real2',size(RPP));
h5write('matlab_real2.h5','/matlab_real2',real(RPP));
h5create('matlab_imag2.h5','/matlab_imag2',size(RPP));
h5write('matlab_imag2.h5','/matlab_imag2',imag(RPP));
h5create('bz.h5','/bz',size(bz));
h5write('bz.h5','/bz',bz)

system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py')
load data1_resfreq.mat

win=ones(size(sig,1),1)*hamming(size(sig,2)).';
spc=fftshift(abs(fft(sig.*win,4096,2)),2);
fsz=13;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVNN by AH
% Ns=1;
% Nsnap=8;
% 
% t=0:63;
% nfft=4096;
% len=size(RPP,2);
% snapLen=len-Nsnap+1;
% freqs = -0.5:1/nfft:0.5-1/nfft;
% final_ret=zeros(size(RPP,1),length(freqs));
% for indix=1:size(RPP,1)
%     indix
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
%         zIk=exp(1i*(-2*pi*freqs(tgt)*t));
%         zIk=zIk/max(abs(zIk));
%         for i=1:Nsnap
%             zI(tgt,i,:)=zIk(i:i+snapLen-1);
%         end
%     end
% 
%     load wgt_freq22.mat
%     zI_set = zI;
%     [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
%     final_ret(indix,:)=ones(1,nfft)./(zO_set_test);
%     final_ret(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
% end
% P_ah=final_ret;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deepfreq
if ~exist('matlab_real1.h5','file')==0
    delete('matlab_real1.h5')
end
if ~exist('matlab_imag1.h5','file')==0
    delete('matlab_imag1.h5')   
end
if ~exist('bz.h5','file')==0
    delete('bz.h5')   
end

h5create('bz.h5','/bz',size(bz));
h5write('bz.h5','/bz',bz)
noisedSig=RPP;

h5create('matlab_real1.h5','/matlab_real1',size(noisedSig));
h5write('matlab_real1.h5','/matlab_real1',real(noisedSig));
h5create('matlab_imag1.h5','/matlab_imag1',size(noisedSig));
h5write('matlab_imag1.h5','/matlab_imag1',imag(noisedSig));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe deepfreq_model.py');
load data1_deepfreq.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1-D diagram
r=r_label/2;
idx=1;
h=figure();
set(h,'position',[100 100 1400 400]);
ha=tight_subplot(1,3,[0.01 0.01],[.2 .08],[.05 .03]);
axes(ha(1))
plot(r,spc(idx,:)./max(spc(idx,:)),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('periodogram');
xlabel({'Relative Range / m';'(a)'});
ylabel('Normalized Amp.');


% axes(ha(2))
% plot(r,abs(P_ah(1,:))./max(abs(P_ah(1,:))),'k-.','linewidth',2);
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% % title('AH');
% xlabel({'Relative Range / m';'(b)'});
% % ylabel('Normalized Amp.');
% % set(gca,'YTick',[]);

axes(ha(2))
data1_deepfreq1=((data1_deepfreq));
plot(r,abs(data1_deepfreq1(idx,:))/max(abs(data1_deepfreq1(idx,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('DeepFreq');
xlabel({'Relative Range / m';'(b)'});
% ylabel('Normalized Amp.');
set(gca,'YTick',[]);
% 
axes(ha(3))
data1_resfreq1=(((data1_resfreq)));
plot(r,abs(data1_resfreq1(idx,:))/max(abs(data1_resfreq1(idx,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
xlabel({'Relative Range / m';'(c)'});
% ylabel('Normalized Amp.');
set(gca,'YTick',[]);
% 

% h=figure();
% set(h,'position',[100 100 800 400]);
% ha=tight_subplot(1,2,[0.08 0.08],[.2 .08],[.08 .05]);
% axes(ha(1))
% plot(r_label,spc(idx,:)./max(spc(idx,:)),'k-.','linewidth',2);
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% title('periodogram');
% xlabel({'Relative Range / m';'(a)'});
% ylabel('Normalized Amp.');
% 
% axes(ha(2))
% plot(r_label,abs(data1_resfreq(idx,:))/max(abs(data1_resfreq(idx,:))),'k-.','linewidth',2);
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
% xlabel({'Relative Range / m';'(b)'});
% ylabel('Normalized Amp.');
% 


%% 2-D diagram
sig=fftshift(fft(sig,[],1),1);
RPP=sig/max(max(abs(sig)));

h=figure();
set(h,'position',[100 100 1400 400]);
ha=tight_subplot(1,3,[0.01 0.01],[.2 .08],[.05 .03]);

axes(ha(1))
win=ones(size(sig,1),1)*hamming(64).';
spc=fftshift(abs(fft(sig.*win,4096,2)),2);
imagesc(r_label/2,1:size(spc,1),spc/max(max(spc)));
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

% title('periodogram');
xlabel({'Relative Range / m';'(a)'});
ylabel('Pulse Index');
% view(30,65)

% axes(ha(2))
% mesh(r_label,1:size(spc,1),abs(P_ah)/max(max(abs(P_ah))));
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% % title('periodogram');
% xlabel({'Relative Range / m';'(a)'});
% ylabel('Pulse Index');
% view(35,65)

%deepfreq

if ~exist('matlab_real1.h5','file')==0
    delete('matlab_real1.h5')
end
if ~exist('matlab_imag1.h5','file')==0
    delete('matlab_imag1.h5')   
end
if ~exist('bz.h5','file')==0
    delete('bz.h5')   
end

h5create('bz.h5','/bz',size(bz));
h5write('bz.h5','/bz',bz)
noisedSig=RPP;

h5create('matlab_real1.h5','/matlab_real1',size(noisedSig));
h5write('matlab_real1.h5','/matlab_real1',real(noisedSig));
h5create('matlab_imag1.h5','/matlab_imag1',size(noisedSig));
h5write('matlab_imag1.h5','/matlab_imag1',imag(noisedSig));
flag=system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe deepfreq_model.py');
load data1_deepfreq.mat
data1_deepfreq2=data1_deepfreq;
axes(ha(2))
imagesc(r_label/2,1:size(spc,1),abs(data1_deepfreq2)/max(max(abs(data1_deepfreq2))));
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'YTick',[]);
% title('periodogram');
xlabel({'Relative Range / m';'(b)'});
% ylabel('Pulse Index');
% view(30,65)

%cResfreq
if ~exist('matlab_real2.h5','file')==0
    delete('matlab_real2.h5')
end

if ~exist('matlab_imag2.h5','file')==0
    delete('matlab_imag2.h5')   
end


if ~exist('bz.h5','file')==0
    delete('bz.h5')   
end

h5create('matlab_real2.h5','/matlab_real2',size(RPP));
h5write('matlab_real2.h5','/matlab_real2',real(RPP));
h5create('matlab_imag2.h5','/matlab_imag2',size(RPP));
h5write('matlab_imag2.h5','/matlab_imag2',imag(RPP));
h5create('bz.h5','/bz',size(bz));
h5write('bz.h5','/bz',bz)

system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py')
load data1_resfreq.mat
data1_resfreq2=data1_resfreq;
axes(ha(3))
imagesc(r_label/2,1:size(spc,1),abs(data1_resfreq2)/max(max(abs(data1_resfreq2))))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'YTick',[]);
% title('cResFreq');
xlabel({'Relative Range / m';'(c)'});
% ylabel('Pulse Index');
% view(30,65)

% h=figure();
% set(h,'position',[100 100 1000 600]);
% ha=tight_subplot(1,2,[0.08 0.08],[.2 .08],[.08 .05]);
% 
% axes(ha(1))
% win=ones(size(sig,1),1)*hamming(64).';
% spc=fftshift(abs(fft(sig.*win,4096,2)),2);
% mesh(r_label,1:size(spc,1),spc/max(max(spc)));
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% title('periodogram');
% xlabel({'Relative Range / m';'(a)'});
% ylabel('Pulse Index');
% view(35,65)
% 
% 
% axes(ha(2))
% mesh(r_label,1:size(spc,1),abs(data1_resfreq)/max(max(data1_resfreq)))
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
% xlabel({'Relative Range / m';'(b)'});
% ylabel('Pulse Index');
% view(35,65)


