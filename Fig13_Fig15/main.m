%%
clear all
close all
Rmin=20000;
Rmax=35150;
Fc=8.5e9;
T=100e-6; %脉冲宽度
B=0.3e9;
k=B/T;
K=k;
Fs=1e9;
dt=1/Fs;
Ts=dt;
c=3e8;
Rwid=Rmax-Rmin;                          
Twid=2*Rwid/c;                      
Nwid=ceil(Twid/Ts);                       
Nchirp=ceil(T/Ts);                               
t=linspace(2*Rmin/c,2*Rmax/c,Nwid); 
nb=T*B;                       
df=1/T;                      
f=(-(nb-1)/2:(nb-1)/2)*df;      
%%
center=[Rmin+T*c/4+50,0,0];
x=1*[0,-1, -1,-3.5,-6.5,-6.5,-3.5,  -2.5,-2.5, 2.5,  2.5,   3.5,6.5, 6.5,3.5,1,1]+center(1);
y=1*[9, 7,  4, 2.5, 0.5,  -1,-0.5,  -3.5,  -4,  -4, -3.5,  -0.5, -1, 0.5,2.5,4,7];
h=figure();
% set(h,'position',[100 100 400 400]);
% ha=tight_subplot(1,1,[0.05 0.05],[.2 .08],[.05 .01])
% axes(ha(1))
fsz=13;
scatter(y,x-center(1),'k','filled');
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('Target Layout');
xlabel({'Radial Range / m';'(a)'});
ylabel('Cross Range / m');
xlim([-8 13])
ylim([-8 8])
grid on


tgt_num=length(x);
theta=1/2*pi;
for i=1:length(x)
    huai=atan2(y(i)-center(2),x(i)-center(1));
    xx(i)=sqrt((x(i)-center(1))^2+(y(i)-center(2))^2)*cos(huai+theta)+center(1);
    yy(i)=sqrt((x(i)-center(1))^2+(y(i)-center(2))^2)*sin(huai+theta);
end

%%
x=xx;
y=yy;
w=0.15;
bz=512;
for i=1:bz
    for ii=1:length(x)
        theta=atan2(y(ii)-center(2),x(ii)-center(1));
        RR(i,ii)=sqrt((x(ii)-center(1))^2+(y(ii)-center(2))^2)*cos(theta+w*(i)/180*pi)+center(1);
    end
end

len=fix(T/dt);
lin=-len/2:len/2-1; 
S_Ref=exp(j*pi*k*(lin*dt).^2);
% amp=abs(randn(1,tgt_num));
amp=ones(1,tgt_num)
SNR=20;

sig=zeros(bz,size(f,2));
for i1=1:bz
    i1
    for tgt=1:length(x)
        tau(tgt)=2*RR(i1,tgt)/c;
        sig(i1,:)=sig(i1,:)+amp(tgt)*exp(-1i*2*pi*(f+Fc)*tau(tgt)); 
    end
    sig(i1,:)=sig(i1,:)/sqrt(mean(abs(sig(i1,:).^2)));
    sig(i1,:)=sig(i1,:)*10^(SNR/20)+wgn(size(sig(i1,:),1),size(sig(i1,:),2),0,'complex');
    ys1=sig(i1,:).*exp(-1i*2*pi*4700*(1:length(sig(1,:)))/length(sig(1,:)));
    ys2=fft(resample(ys1,1,60));  
    tmp=ys2(172:235);
    
    sig2(i1,:)=ifft(fftshift(tmp));
    RPP(i1,:)=sig2(i1,:)/max(max(abs(sig2(i1,:))));
end

df=df*468;
nfft=4096;
deltaR=3e8/2/df/nfft;
r=(-(nfft-1)/2:(nfft-1)/2).*deltaR;

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
% for i=1:bz
%     data_resfreq_norm(i,:)=data1_resfreq(i,:)/max(abs(data1_resfreq(i,:)));
% end
win=ones(size(sig2,1),1)*hamming(64).';
spc=fftshift(abs(fft(sig2.*win,4096,2)),2);

h=figure();
set(h,'position',[100 100 800 400]);
ha=tight_subplot(1,2,[0.05 0.05],[.2 .08],[.08 .01]);
axes(ha(1))
plot(r,spc(1,:)./max(spc(1,:)),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('periodogram');
xlabel({'Range / m';'(a)'});
ylabel('Normalized Amp.');
xlim([-8 13])


axes(ha(2))
plot(r,abs(data1_resfreq(1,:))/max(abs(data1_resfreq(1,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range / m';'(b)'});
ylabel('Normalized Amp.');
xlim([-8 13])

h=figure();
set(h,'position',[100 100 800 400]);
ha=tight_subplot(1,2,[0.05 0.05],[.2 .08],[.08 .01])

axes(ha(1))
win=ones(size(sig2,1),1)*hamming(64).';
spc=fftshift(abs(fft(sig2.*win,4096,2)),2);
imagesc(r,1:512,spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range / m';'(a)'});
ylabel('Pulse Index');
xlim([-10 10])

axes(ha(2))
imagesc(r,1:512,abs(data1_resfreq))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range / m';'(b)'});
xlim([-10 10])



