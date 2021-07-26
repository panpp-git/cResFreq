clc
clear
close all
format long
rng('default')
load peri.mat
load music.mat
load omp.mat
load p_ah.mat
load deepfreq.mat

h=figure();
set(h,'position',[100 100 2200 400]);
ha=tight_subplot(1,5,[0.022 0.022],[.24 .08],[.04 .03]);
%% periodogram
tgt_num=4;
nfft=4096;
x_label=-0.5:1/nfft:0.5-1/nfft;
L=64;
w=2*pi*[-1/L,0/L,5/L,6/L];
y=w/2/pi;
fontsz=16;
axes(ha(1))
for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end

set(gca,'FontSize',fontsz); 
set(get(gca,'XLabel'),'FontSize',fontsz);
set(get(gca,'YLabel'),'FontSize',fontsz);
normPeriodogram=10*log10(periodogram_nowin_20dB/max(periodogram_nowin_20dB)+1e-13);
plot(x_label,normPeriodogram,'b:.','linewidth',2);
hold on;
normPeriodogram_win=10*log10(periodogram_win_20dB/max(periodogram_win_20dB)+1e-13);
plot(x_label,normPeriodogram_win,'k:','linewidth',3);
hold on;
axis([-0.1 0.18 -50 5])
hh=legend('Rect.','Hamm.');
set(hh,'Fontsize',fontsz)
ylabel('Normalized Power / dB');
xlabel({'Normalized freq. / Hz','(a)'});
grid on;

%% MUSIC
axes(ha(2))
w=2*pi*[-5/L,0/L,5/L,5.7/L];
y=w/2/pi;
for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
axis([-0.1 0.18 -50 5])
set(gca,'FontSize',fontsz); 
set(get(gca,'XLabel'),'FontSize',fontsz);
set(get(gca,'YLabel'),'FontSize',fontsz);
normMusic=10*log10(P_music_20dB/max(P_music_20dB)+1e-13);
plot(x_label,real(normMusic).','b:.','linewidth',2);
hold on
normMusic=10*log10(P_music_0dB/max(P_music_0dB)+1e-13);
plot(x_label,real(normMusic).','k:','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
% ylabel('Normalized Power / dB');
xlabel({'Normalized freq. / Hz','(b)'});
grid on;
hold on;

%% omp
axes(ha(3))

for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
axis([-0.1 0.18 -50 5])
set(gca,'FontSize',fontsz); 
set(get(gca,'XLabel'),'FontSize',fontsz);
set(get(gca,'YLabel'),'FontSize',fontsz);
normomp=20*log10(omp20/max(omp20)+1e-13);
plot(x_label,real(normomp).','b:.','linewidth',2);
hold on
normomp=20*log10(omp0/max(omp0)+1e-13);
plot(x_label,real(normomp).','k:','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
% ylabel('Normalized Power / dB');
xlabel({'Normalized freq. / Hz','(c)'});
grid on;
hold on;

%% cvnn
axes(ha(4))

for i=1:tgt_num
    h1=stem(y(i),-50,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end
axis([-0.1 0.18 -50 5])
set(gca,'FontSize',fontsz); 
set(get(gca,'XLabel'),'FontSize',fontsz);
set(get(gca,'YLabel'),'FontSize',fontsz);
normah=20*log10(P_ah20/max(P_ah20)+1e-13);
plot(x_label,real(normah).','b:.','linewidth',2);
hold on
normah=20*log10(P_ah0/max(P_ah0)+1e-13);
plot(x_label,real(normah).','k:','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
% ylabel('Normalized Power / dB');
xlabel({'Normalized freq. / Hz','(d)'});
grid on;
hold on;

%% DeepFreq
tgt_num=6;
w=2*pi*[-5/L,0/L,5/L,5.7/L,8/L,13/L];
y=w/2/pi;

axes(ha(5))
for i=1:tgt_num
    h1=stem(y(i),-80,'r-','Marker','none','linewidth',2);
    hold on;
    h2=stem(y(i),5,'r-','Marker','none','linewidth',2);
    hold on;
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
end

set(gca,'FontSize',fontsz); 
set(get(gca,'XLabel'),'FontSize',fontsz);
set(get(gca,'YLabel'),'FontSize',fontsz);


normDeepFreq=10*log10(data1_deepfreq20.^2/max(data1_deepfreq20.^2)+1e-13);
plot(x_label,real(normDeepFreq),'b:.','linewidth',2);
axis([-0.1 0.25 -50 5])
% ylabel('Normalized PSD / dB');

grid on;
hold on

normDeepFreq=10*log10(data1_deepfreq0.^2/max(data1_deepfreq0.^2)+1e-13);
plot(x_label,real(normDeepFreq),'k:','linewidth',3);
legend('SNR = 20dB','SNR = 0dB');
% ylabel('Normalized Power / dB');
xlabel({'Normalized freq. / Hz','(e)'});
grid on;


