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
x=1*[0,-1,-1, -1,-3.5,-6.5,-6.5,-3.5,-1,  -1,  -2.5,-2.5,2.5, 2.5,   1, 1, 3.5,6.5,6.5,3.5,1,1,1,0]+center(1);
y=1*[9, 7,5.5, 4, 2.5, 0.5,-1,  -0.5, 0, -2.5, -3.5, -4,  -4, -3.5,-2.5,0,-0.5,-1, 0.5,2.5,4,5.5,7,9];
% x=1*[0,-1,-1]+center(1);
% y=1*[9, 7,5.5];

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
w=1.5;
for i=1:128
    for ii=1:length(x)
        theta=atan2(y(ii)-center(2),x(ii)-center(1));
        RR(i,ii)=sqrt((x(ii)-center(1))^2+(y(ii)-center(2))^2)*cos(theta+w*(i)/180*pi)+center(1);
    end
end

len=fix(T/dt);
lin=-len/2:len/2-1; 
S_Ref=exp(j*pi*k*(lin*dt).^2);
amp=abs(randn(1,tgt_num));
SNR=20;
bz=120;
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
figure;imagesc(abs(data1_resfreq))
win=ones(size(sig2,1),1)*hamming(64).';

spc=fftshift(abs(fft(sig2.*win,4096,2)),2);
figure;imagesc(spc);
% SNR=20;
% for ll=1:64       
%     ll
%     R=RR(ll,:);
%     RCS=ones(1,length(R))*1;
%     xs=zeros(1,len);    
%     %==================================================================
%     %%Gnerate the echo      
%     t=linspace(2*Rmin/c,2*Rmax/c,Nwid);       %receive wind
%     M=length(R);                                                                
%     td=ones(M,1)*t-2*R'/c*ones(1,Nwid);
%     xs=RCS*((ones(Nwid,1)*exp(-1i*2*pi*Fc*2*R/c)).'.*exp(1i*pi*K*td.^2).*(abs(td)<T/2));%radar echo from point targets 
% %     tmp(:,:)=wgn(size(xs,1),size(xs,2),0,'complex');
% %     xs=xs+tmp(:,:);
%     xs=xs/sqrt(mean(abs(xs.^2)));
%     xs=xs*10^(SNR/20)+wgn(size(xs,1),size(xs,2),0,'complex');
% 
%     Srw=fft(xs);
%     len1=length(xs);
%     Sw=fft(S_Ref,len1); 
%     ys=Srw.*conj(Sw); 
%     ys1=ys.*exp(-1i*2*pi*100700*(1:len1)/len1);
%     ys2=fftshift(fft(resample(ys1,1,800)));  
%     tmp=ys2(36:99);
%     
%     RPP(ll,:)=ifft(ifftshift(tmp));
% %      tmp2=fftshift(fft(ys2));
% % 
% %      tmp=fftshift(tmp2(50:50+127));
% % 
% % %      temp=fftshift(ys2);
% %      RPP(ll,:)=ifft(tmp); 
% end

% RPP=RPP(16,:);
% figure;plot((abs(fft(RPP))));
% if ~exist('matlab_real.h5','file')==0
%     delete('matlab_real.h5')
% end
% 
% if ~exist('matlab_imag.h5','file')==0
%     delete('matlab_imag.h5')   
% end
% 
% if ~exist('tgt_num.h5','file')==0
%     delete('tgt_num.h5')   
% end
% 
% 
% h5create('matlab_real.h5','/matlab_real',size(RPP));
% h5write('matlab_real.h5','/matlab_real',real(RPP/max(max(abs(RPP)))));
% h5create('matlab_imag.h5','/matlab_imag',size(RPP));
% h5write('matlab_imag.h5','/matlab_imag',imag(RPP/max(max(abs(RPP)))));
% h5create('tgt_num.h5','/tgt_num',size(tgt_num));
% h5write('tgt_num.h5','/tgt_num',tgt_num)
% 
% system('E:\ProgramData\Anaconda3\envs\pytorch\python.exe py_model.py')
% py_data = (h5read('python_data.h5','/python_data')).';


% SNR=20;
% for ll=1:64       
%     ll
%     R=RR(ll,:);
%     RCS=ones(1,length(R))*1;
%     xs=zeros(1,len);    
%     %==================================================================
%     %%Gnerate the echo      
%     t=linspace(2*Rmin/c,2*Rmax/c,Nwid);       %receive wind
%     M=length(R);                                                                
%     td=ones(M,1)*t-2*R'/c*ones(1,Nwid);
%     xs=RCS*((ones(Nwid,1)*exp(-1i*2*pi*Fc*2*R/c)).'.*exp(1i*pi*K*td.^2).*(abs(td)<T/2));%radar echo from point targets 
% %     tmp(:,:)=wgn(size(xs,1),size(xs,2),0,'complex');
% %     xs=xs+tmp(:,:);
%     xs=xs/sqrt(mean(abs(xs.^2)));
%     xs=xs*10^(SNR/20)+wgn(size(xs,1),size(xs,2),0,'complex');
% 
%     Srw=fft(xs);
%     len1=length(xs);
%     Sw=fft(S_Ref,len1); 
%     ys=Srw.*conj(Sw); 
%     ys1=ys.*exp(-1i*2*pi*100670*(1:len1)/len1);
%     ys2=resample(ys1,1,300);  %采样因子500
%     tmp=fftshift(ys2);
%     tmp=tmp(107:234);
%     RPP(ll,:)=tmp;
% %      tmp2=fftshift(fft(ys2));
% % 
% %      tmp=fftshift(tmp2(50:50+127));
% % 
% % %      temp=fftshift(ys2);
% %      RPP(ll,:)=ifft(tmp); 
% end