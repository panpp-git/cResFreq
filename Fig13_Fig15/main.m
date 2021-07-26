%% Wide Band LFM Echo for ISAR
clear all;
close all
clc
rng('default')
% format long
%% 雷达基本参数设置
C=299792458;                    %光速
% C=3e8;
FC=9e9;                         %中心频率
FS=1e9;                         %采样频率
TS=1/FS;                        %采样间隔
rng_gate_res=C/2/FS;            %距离波门分辨率
BW=0.3e9;                         %带宽
PRF=1;                        %脉冲重复频率
PRI=1/PRF;                      %脉冲重复间隔
TW=100e-6;                      %脉冲宽度
sweep_slope=BW/TW;              %调频率
n_pulse=512;                     %脉冲总数
% r_rwnd=15900;                   %波门大小（距离）
% t_rwnd=2*r_rwnd/C;              %波门大小（时间）
% n_rwnd=ceil(t_rwnd/TS);         %波门大小（点数）
n_rwnd=120000;
df=1/TW;   
%-------------------------------------------------------------------------%
%% 散射点模型建立
center=[8000,0,0];                                             %模型中心

x=[5,3.2,2.8,1,3]+center(1);      %散射点相对坐标
y=[-1,1,2,0,-2];

fsz=13;
scatter(x-center(1),y,'k','filled');
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('Target Layout');
xlabel({'Relative Radial Range / m';'(a)'});
ylabel('Cross Range / m');
ylim([-3 3])
xlim([-1 7])
grid on



theta=0;
for i=1:length(x)
    huai=atan2(y(i)-center(2),x(i)-center(1));
    xx(i)=sqrt((x(i)-center(1))^2+(y(i)-center(2))^2)*cos(huai+theta)+center(1);
    yy(i)=sqrt((x(i)-center(1))^2+(y(i)-center(2))^2)*sin(huai+theta);
end
x=xx;
y=yy;


n_tgt=length(x);                                %散射点数目

%-------------------------------------------------------------------------%
%% 旋转速度、速度、加速度、幅度、距离门初始设置
%旋转速度不能过快，需考虑距离单元大小的关系
%目标速度值选取，需考虑与PRI、波门改变时间、波门大小的关系
%目标加速度不能过大，否则测速精度降低，影响补偿效果
rotate_w=0.15;
tgt_vel=zeros(n_pulse,n_tgt);                 %目标速度
tgt_dist=zeros(n_pulse,n_tgt);                %目标距离
rng_gate=zeros(n_pulse,1);                    %距离门前沿

sig_echo=[];                                  %接收信号
tgt_dist_init=zeros(n_tgt,1);                 %目标初始距离
tgt_vel_init=zeros(n_tgt,1);                  %目标初始速度
tgt_acc_init=zeros(n_tgt,1);                  %目标初始加速度
amp=zeros(n_tgt,1);                           %接收信号幅度
time_rec=[];
for k = 1:n_tgt
    tgt_dist_init(k)=center(1);           %目标初始距离
    tgt_vel_init(k)=0;                      %目标初始速度
    tgt_acc_init(k)=0;                        %目标加速度
    amp(k)=1;                                 %目标反射系数权值
end
%-------------------------------------------------------------------------%
%% 生成LFM回波信号
%构造参考信号
dt=TS;
t=dt:dt:n_rwnd*dt;
bz=n_pulse;
% sig_ref=exp(1i*sweep_slope*pi*t.*t);          %本地参考信号
% sig_dechirped=[];
% %产生回波信号（去斜，下采样）
% ll=1;  
% SNR=20;
% %进度条刻度
% 
% for j = 1:n_pulse
%     h=waitbar(ll/n_pulse);                    %进度条    
%     ll=ll+1;
% %-------------------------------------------------------------------------%    
%     %目标模型
%     for k = 1:n_tgt
%         %旋转
%         theta=atan2(y(k)-center(2),x(k)-center(1));
%         RR(j,k)=sqrt((x(k)-center(1))^2+(y(k)-center(2))^2)*cos(theta+rotate_w*(j)/180*pi);
% 
%         %平动
%         tgt_vel(j,k)=tgt_vel_init(k)+(j-1)*tgt_acc_init(k)*PRI;
%         tgt_dist(j,k)=tgt_dist_init(k)+tgt_vel_init(k)*((j-1)*PRI)...
%                 +tgt_acc_init(k)*((j-1)*PRI)^2/2+RR(j,k);
%     end
%  %-------------------------------------------------------------------------%
%     %距离门移动，每隔固定脉冲数移动一次
%      if j==1  
%         rng_gate(j)=floor(min(tgt_dist(j,:)-C*TW/4-C*TW/128)/rng_gate_res)*rng_gate_res;
%      else 
%         rng_gate(j)=rng_gate(j-1);
%      end
% %-------------------------------------------------------------------------%    
%     %记录时刻
%     time_rec(j)=(j-1)*PRI;
% %-------------------------------------------------------------------------%    
%     %产生信号
%     sig_echo=EchoSignalGen(TW,BW,rng_gate(j),n_rwnd,tgt_dist(j,:),amp.',FC,TS);
% %     figure;plot(real(sig_echo));
% %-------------------------------------------------------------------------%    
%     %去斜，下采样（下采样比例不能过大，需考虑下采样后的采样频率与去斜信号带宽关系，防止混叠）
%     sig_dechirped=(sig_echo).*(conj(sig_ref));
%     sig_dechirped_ds=sig_dechirped;
%     tmp1=wgn(size(sig_dechirped_ds,1),size(sig_dechirped_ds,2),0,'complex');
%     sig_dechirped_ds=sig_dechirped_ds/sqrt(mean((abs(sig_dechirped_ds)).^2));
%     sig_dechirped_ds_1=sig_dechirped_ds*10^(SNR/20)+tmp1;
%     sig_dechirped_down(j,:)=sig_dechirped_ds_1(1:1625:104000);
%     RPP(j,:)=sig_dechirped_down(j,:)/max((abs(sig_dechirped_down(j,:))));
% end
% 
% 
% save sig_dechirped_down.mat sig_dechirped_down
% save RPP.mat RPP
load sig_dechirped_down.mat
load RPP.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs=104000*dt*sweep_slope;
df=fs/(104000/1625);
nfft=4096;
deltaR=C/2/df/nfft;
unr=(nfft-1)*deltaR;
r=(0:(nfft-1)).*deltaR;

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
win=ones(size(sig_dechirped_down,1),1)*hamming(64).';
spc=((abs(ifft(sig_dechirped_down.*win,4096,2))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %MUSIC
L=64;
tgt_num=n_tgt;
search_f=0:1/nfft:1-1/nfft;
P_music=zeros(size(sig_dechirped_down,1),nfft);
for indix=1:size(sig_dechirped_down,1)
% for indix=1:1
    noisedSig=sig_dechirped_down(indix,:);
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
    musicWinLen=L/2;
   
    [EV,D] = eig(Rss);
    [EVA,I] = sort(diag(D).');
    EV = fliplr(EV(:,I));
    G = EV(:,tgt_num+1:end);

    for i=1:length(search_f)
        steeringVec=exp(-1i*2*pi*search_f(i)*(0:musicWinLen-1)).';
        P_music(indix,i)=1/(steeringVec'*G*G'*steeringVec);     
    end
    P_music(indix,:)=sqrt(P_music(indix,:));
    P_music(indix,:)=P_music(indix,:)/max(P_music(indix,:));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OMP
nfft=4096;
L=n_tgt;
dict_freq=0:1/nfft:1-1/nfft;
t=0:63;
dict=exp(-1i*2*pi*dict_freq.'*t).';
for indix=1:size(sig_dechirped_down,1)
% for indix=1:1
    [A]=(OMP(dict,sig_dechirped_down(indix,:).',L));
    xx=abs(full(A));
    P_omp(indix,:)=abs(xx)/max(abs(xx));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVNN by AH
% Ns=1;
% Nsnap=8;
% 
% t=0:63;
% nfft=4096;
% len=size(RPP,2);
% snapLen=len-Nsnap+1;
% freqs = 0:1/nfft:1-1/nfft;
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
%     final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
% end
% P_ah=final_ret;
load P_ah_simu.mat
for i=1:512
    P_ah(i,:)=P_ah(i,:)/max(abs(P_ah(i,:)));
end
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fsz=13;
%% 1-D diagram
h=figure();
set(h,'position',[100 100 2200 400]);
ha=tight_subplot(1,6,[0.01 0.01],[.2 .08],[.03 .03]);
axes(ha(1))
plot(r,spc(1,:)./max(spc(1,:)),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('periodogram');
xlabel({'Relative Range / m';'(a)'});
ylabel('Normalized Amp.');
xlim([10 18])

axes(ha(2))
plot(r,abs(P_music(1,:))./max(abs(P_music(1,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('MUSIC');
xlabel({'Relative Range / m';'(b)'});
% ylabel('Normalized Amp.');
xlim([10 18])
set(gca,'YTick',[]);
% 
axes(ha(3))
plot(r,abs(P_omp(1,:)./max(abs(P_omp(1,:)))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('OMP');
xlabel({'Relative Range / m';'(c)'});
% ylabel('Normalized Amp.');
xlim([10 18])
set(gca,'YTick',[]);

axes(ha(4))
plot(r,abs(P_ah(1,:))./max(abs(P_ah(1,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('AH');
xlabel({'Relative Range / m';'(d)'});
% ylabel('Normalized Amp.');
xlim([10 18])
set(gca,'YTick',[]);

axes(ha(5))
data1_deepfreq1=fliplr(fftshift(data1_deepfreq,2));
plot(r,abs(data1_deepfreq1(1,:))/max(abs(data1_deepfreq1(1,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('DeepFreq');
xlabel({'Relative Range / m';'(e)'});
% ylabel('Normalized Amp.');
xlim([10 18])
set(gca,'YTick',[]);
% 
axes(ha(6))
data1_resfreq1=fliplr(fftshift((data1_resfreq),2));
plot(r,abs(data1_resfreq1(1,:))/max(abs(data1_resfreq1(1,:))),'k-.','linewidth',2);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
xlabel({'Relative Range / m';'(f)'});
% ylabel('Normalized Amp.');
xlim([10 18])
set(gca,'YTick',[]);
% 


%% 2-D diagram
h=figure();
set(h,'position',[100 100 2000 600]);
ha=tight_subplot(2,6,[0.14 0.02],[.14 .05],[.03 .03]);
% x11=12.5; x22=14.5;
% y11=350;   y22=510;
x11=10; x22=13;
y11=180;   y22=380;

axes(ha(1))
win=ones(size(sig_dechirped_down,1),1)*hamming(64).';
spc=((abs(ifft(sig_dechirped_down.*win,4096,2))));
imagesc(r,1:512,spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('periodogram');
xlabel({'Relative Range / m';'(a)'});
ylabel('Pulse Index');
xlim([5 20])
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);

axes(ha(7))
imagesc(r,1:512,spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('periodogram');
xlabel({'Relative Range / m';'(b)'});
ylabel('Pulse Index');
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

axes(ha(2))
imagesc(r,1:512,abs(P_music));
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('Music');
xlabel({'Relative Range / m';'(c)'});
% ylabel('Pulse Index');
set(gca,'YTick',[]);
xlim([5 20])
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);

axes(ha(8))
imagesc(r,1:512,abs(P_music));
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('Music');
xlabel({'Relative Range / m';'(d)'});
set(gca,'YTick',[]);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

axes(ha(3))
imagesc(r,1:512,(abs(P_omp)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('OMP');
xlabel({'Relative Range / m';'(e)'});
set(gca,'YTick',[]);
xlim([5 20])
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);

axes(ha(9))
imagesc(r,1:512,(abs(P_omp)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('OMP');
xlabel({'Relative Range / m';'(f)'});
set(gca,'YTick',[]);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

axes(ha(4))
imagesc(r,1:512,(abs(P_ah)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('OMP');
xlabel({'Relative Range / m';'(g)'});
set(gca,'YTick',[]);
xlim([5 20])
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);

axes(ha(10))
imagesc(r,1:512,(abs(P_ah)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('OMP');
xlabel({'Relative Range / m';'(h)'});
set(gca,'YTick',[]);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

axes(ha(5))
% data1_resfreq(data1_resfreq<0.5*max(max(data1_resfreq)))=0;
% data1_deepfreq=fliplr(data1_deepfreq);
imagesc(r,1:512,abs(data1_deepfreq1))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
xlabel({'Relative Range / m';'(i)'});
set(gca,'YTick',[]);
xlim([5 20])
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);

axes(ha(11))
imagesc(r,1:512,abs(data1_deepfreq1))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
xlabel({'Relative Range / m';'(j)'});
set(gca,'YTick',[]);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

axes(ha(6))
% data1_resfreq(data1_resfreq<0.5*max(max(data1_resfreq)))=0;
imagesc(r,1:512,abs(data1_resfreq1))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
xlabel({'Relative Range / m';'(k)'});
set(gca,'YTick',[]);
xlim([5 20])
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);

axes(ha(12))
imagesc(r,1:512,abs(data1_resfreq1))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% title('cResFreq');
xlabel({'Relative Range / m';'(l)'});
set(gca,'YTick',[]);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);





