clc
clear
close all

load boeing727.mat
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
idx=46;


win=hamming(size(sig,1))*hamming(size(sig,2)).';
spc=fftshift(abs(fft(sig.*win,4096,2)),2);
fsz=13;
h=figure();
set(h,'position',[100 100 800 400]);
ha=tight_subplot(1,2,[0.08 0.08],[.2 .08],[.08 .05]);
axes(ha(1))
plot(spc(idx,:)./max(spc(idx,:)),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('periodogram');
xlabel({'Range Cell/ m';'(a)'});
ylabel('Normalized Amp.');



axes(ha(2))
plot(abs(data1_resfreq(idx,:))/max(abs(data1_resfreq(idx,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range Cell';'(b)'});
ylabel('Normalized Amp.');


h=figure();
set(h,'position',[100 100 800 400]);
ha=tight_subplot(1,2,[0.08 0.08],[.2 .08],[.08 .05]);

axes(ha(1))
win=hamming(size(RPP,1))*hamming(64).';
spc=fftshift(abs(fft(sig.*win,4096,2)),2);
imagesc(spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('periodogram');
xlabel({'Range Cell';'(a)'});
ylabel('Pulse Index');


axes(ha(2))
imagesc(abs(data1_resfreq))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range Cell';'(b)'});
ylabel('Pulse Index');

figure;
win=ones(size(RPP,1),1)*hamming(64).';
spc=fftshift(abs(fft2(sig,128,4096)),2);
imagesc(spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('Boeing 727');
xlabel('Range Cell');
ylabel('Doppler Cell')
