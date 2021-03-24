%% Wide Band LFM Echo for ISAR
clear all;
close all
clc
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
x=[0,-1,-1,-6.5,-6.5,-1,-1,-2.5,-2.5,2.5,2.5,1,1,6.5,6.5,1,1]+center(1);      %散射点相对坐标
y=[9,7,4,0.5,-1,0,-2.5,-3.5,-4,-4,-3.5,-2.5,0,-1,0.5,4,7];



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
sig_ref=exp(1i*sweep_slope*pi*t.*t);          %本地参考信号
sig_dechirped=[];
%产生回波信号（去斜，下采样）
ll=1;  
SNR=20;
%进度条刻度
bz=n_pulse;
for j = 1:n_pulse
    h=waitbar(ll/n_pulse);                    %进度条    
    ll=ll+1;
%-------------------------------------------------------------------------%    
    %目标模型
    for k = 1:n_tgt
        %旋转
        theta=atan2(y(k)-center(2),x(k)-center(1));
        RR(j,k)=sqrt((x(k)-center(1))^2+(y(k)-center(2))^2)*cos(theta+rotate_w*(j-1)/180*pi)+center(1);

        %平动
        tgt_vel(j,k)=tgt_vel_init(k)+(j-1)*tgt_acc_init(k)*PRI;
        tgt_dist(j,k)=tgt_dist_init(k)+tgt_vel_init(k)*((j-1)*PRI)...
                +tgt_acc_init(k)*((j-1)*PRI)^2/2+RR(j,k);
    end
 %-------------------------------------------------------------------------%
    %距离门移动，每隔固定脉冲数移动一次
     if j==1  
        rng_gate(j)=floor(min(tgt_dist(j,:)-C*TW/4-C*TW/128)/rng_gate_res)*rng_gate_res;
     else 
        rng_gate(j)=rng_gate(j-1);
     end
%-------------------------------------------------------------------------%    
    %记录时刻
    time_rec(j)=(j-1)*PRI;
%-------------------------------------------------------------------------%    
    %产生信号
    sig_echo=EchoSignalGen(TW,BW,rng_gate(j),n_rwnd,tgt_dist(j,:),amp.',FC,TS);
%     figure;plot(real(sig_echo));
%-------------------------------------------------------------------------%    
    %去斜，下采样（下采样比例不能过大，需考虑下采样后的采样频率与去斜信号带宽关系，防止混叠）
    sig_dechirped=(sig_echo).*(conj(sig_ref));
    sig_dechirped_ds=sig_dechirped;
    tmp1=wgn(size(sig_dechirped_ds,1),size(sig_dechirped_ds,2),0,'complex');
    sqrt(mean((abs(sig_dechirped_ds)).^2))
    sig_dechirped_ds=sig_dechirped_ds/sqrt(mean((abs(sig_dechirped_ds)).^2));
    sig_dechirped_ds_1=sig_dechirped_ds*10^(SNR/20)+tmp1;
    
    sig_dechirped_down(j,:)=sig_dechirped_ds_1(1:1625:104000);
    
    RPP(j,:)=sig_dechirped_down(j,:)/max((abs(sig_dechirped_down(j,:))));
end
df=df*size(sig_dechirped_ds_1,2)/64;
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
win=ones(size(sig_dechirped_down,1),1)*hamming(64).';
spc=fftshift(abs(fft(sig_dechirped_down.*win,4096,2)),2);
fsz=13;
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
win=ones(size(sig_dechirped_down,1),1)*hamming(64).';
spc=fftshift(abs(fft(sig_dechirped_down.*win,4096,2)),2);
imagesc(r,1:512,spc);
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range / m';'(a)'});
ylabel('Pulse Index');
% xlim([-10 10])

axes(ha(2))
imagesc(r,1:512,abs(data1_resfreq))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
title('cResFreq');
xlabel({'Range / m';'(b)'});
% xlim([-10 10])
x=1

%%

