function EchoSig=EchoSignalGen(TW,BW,RNG_MIN,n_rwnd,tgt_dist,amp,FC,TS)
%% LFM EchoSignal Generation
%C为光速
%TW为信号脉宽
%BW为LFM信号带宽
%sweep_slope为调频率
%RNG_MIN为波门前沿
%RNG_MAX为波门后沿
%r_rwnd为总点数
%tgt_dist为目标位置
%amp为目标回波幅度
%FC为载频
%TS为采样间隔
%==================================================================
%% Parameter
% C=299792458;                                            %光速
C=3e8;
sweep_slope=BW/TW;                                        %调频率

RNG_MAX=RNG_MIN+(n_rwnd-1)*TS*C/2;
%==================================================================
%% Gnerate the echo      
t=linspace(2*RNG_MIN/C,2*RNG_MAX/C,n_rwnd);               %波门（时域）
                                                          %波门前沿 t=2*Rmin/C
                                                          %波门后沿 t=2*Rmax/C                            
M=length(tgt_dist);                                       %目标数目                                       
td=ones(M,1)*t-2*tgt_dist'/C*ones(1,n_rwnd);
Srt=amp*((ones(n_rwnd,1)*exp(-j*2*pi*FC*2*tgt_dist/C)).'.*exp(j*pi*sweep_slope*td.^2).*(abs(td)<TW/2));%radar echo from point targets 
EchoSig=Srt;
%==================================================================

