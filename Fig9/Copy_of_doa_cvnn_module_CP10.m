
function [wHI, wOH, zO_set] = Copy_of_doa_cvnn_module_CP10 (zI_set, zO_teach_set)
len=size(zI_set,3);
bsz=size(zI_set,1);
sizeI=size(zI_set,2);
sizeH=12;
sizeO=1;
TDL_n=29;
fc=500e3;
k1= 0.03;                % learning constant
k2= 0.03;
Ts=1e-7;
rng(123)
wHI = rand(sizeH, sizeI);   
wOH = rand(sizeO, sizeH);   
wHI(:,1)=1;
wOH(:,1)=1;

% Power inversion initiation
% zI=zI_set;
% Rxx=squeeze(zI(1,:,:))*squeeze(zI(1,:,:))'/8;
% S=[1,0,0,0,0,0,0,0]';
% Wopt=Rxx\S;
% Wopt=Wopt/abs(Wopt(1));
% for j=1:sizeH
%     for nn=1:TDL_n
%         wHI(j,:,nn)=Wopt; 
%     end
% end
% wOH_amp=randn(sizeO, sizeH, TDL_n);
% wOH_phase=randn(sizeO, sizeH, TDL_n);
% wOH =wOH_amp.*exp(1i*wOH_phase);   
% wHI(:,:,1)=1;
% wOH(:,:,1)=1;

% wHI_amp=randn(sizeH, sizeI, TDL_n);
% wHI_phase=randn(sizeH, sizeI, TDL_n);
% wHI =wHI_amp.*exp(1i*wHI_phase);   
% wOH_amp=randn(sizeO, sizeH, TDL_n);
% wOH_phase=randn(sizeO, sizeH, TDL_n);
% wOH =wOH_amp.*exp(1i*wOH_phase);   
% wHI(:,:,1)=1;
% wOH(:,:,1)=1;


iteration =501;
counter = 1;
er_matrix = zeros();
%% input normalization
% for row=1:s
%     zI_set(row, 1:end-1) = zI_set(row, 1:end-1)/ max(abs(zI_set(row, 1:end-1)));
%     zO_teach_set(row, :) = zO_teach_set(row, :) / max(abs(zO_teach_set(row, :)));
% end
%%
min_er=10000;
while counter < iteration
    er = 0;
    for iter=1:bsz
        for ti=1:len
            uHI=zeros(sizeH,len);
            yHI=zeros(sizeH,len);
            xin=zeros(sizeI,len);
            for j=1:sizeH
                uj=zeros(1,len);
                for i=1:sizeI
                    uji=zeros(1,len);
                    if ~(i==1)
                        ss=squeeze(zI_set(iter,i,:)).';
                        xin(i,:)=ss;
                        wgt_xin=wHI(j,i)*ss;
                        uji=uji+wgt_xin;
                    else
                        uji=squeeze(zI_set(iter,i,:)).';           
                    end
                    uj=uj+uji;
                end
                uHI(j,:)=uj;
                yHI(j,:)=tanh(abs(uHI(j,:))).*exp(1i*angle(uHI(j,:)));
            end

            % output zO
            uOH=zeros(sizeO,len);
            yOH=zeros(sizeO,len);
            xin2=zeros(sizeH,len);
            for j=1:sizeO
                uj=zeros(1,len);
                for i=1:sizeH
                    uji=zeros(1,len);
                    if ~(i==1)
                        ss=yHI(i,:);
                        xin2(i,:)=ss;
                        wgt_xin2=wOH(j,i)*ss;
                        uji=uji+wgt_xin2;

                    else
                        uji=yHI(i,:);
                    end
                    uj=uj+uji;
                end
                uOH(j,:)=uj;
                yOH(j,:)=tanh(abs(uOH(j,:))).*exp(1i*angle(uOH(j,:)));
            end

            %loss
            temp    = yOH(ti)*yOH(ti)';
            er      = (1/2) .* sum( temp )+er ;

            zOt = zO_teach_set(iter, :);
            zO=yOH;
            for jj = 1:sizeO
              for ii = 1:sizeH 
         
                  if ii==1
                    deltaEwOH1(jj,ii)=0;
                  else
                    deltaEwOH1(jj,ii) =  (1- abs(zO(jj,ti)).^2) .* (abs(zO(jj,ti)) - abs(zOt(jj,ti)) .* ...
                             cos(angle(zO(jj,ti)) - angle(zOt(jj,ti)))) .* abs((xin2(ii,ti))) .* ...
                             cos(angle(zO(jj,ti)) - angle((xin2(ii,ti))) - angle(wOH(jj,ii))) - ...
                             abs(zO(jj,ti)) .* abs(zOt(jj,ti)) .* sin(angle(zO(jj,ti)) - angle(zOt(jj,ti))) .* ...
                             (abs((xin2(ii,ti))) ./ (abs(uOH(jj,ti)))).* ...
                             sin(angle(zO(jj,ti)) - angle((xin2(ii,ti))) - angle(wOH(jj,ii)));
                  end

              end
            end

            for jj = 1:sizeO
              for ii = 1:sizeH   
                  if  ii==1
                    deltaEwOH2(jj,ii)=0;
                  else
                    deltaEwOH2(jj,ii) =  (1- abs(zO(jj,ti)).^2) .* (abs(zO(jj,ti)) - abs(zOt(jj,ti)) .* ...
                             cos(angle(zO(jj,ti)) - angle(zOt(jj,ti))) ) .* abs((xin2(ii,ti))) .* ...
                             sin(angle(zO(jj,ti)) - angle((xin2(ii,ti))) - angle(wOH(jj,ii))) + ...
                             abs(zO(jj,ti)) .* abs(zOt(jj,ti)) .* sin(angle(zO(jj,ti)) - angle(zOt(jj,ti))) .* ...
                             (abs((xin2(ii,ti))) ./ (abs(uOH(jj,ti)))).* ...
                             cos(angle(zO(jj,ti)) - angle((xin2(ii,ti))) - angle(wOH(jj,ii)));
                  end
              end
            end


            % Hermite conjugate
            for h = 1:sizeH
                zHt(h,:) = zeros(1,len);
            end

            zH=yHI;
            for jj = 1:sizeH  
              for ii = 1:sizeI

                  if ii==1
                    deltaEwHI1(jj,ii)=0;
                  else
                    deltaEwHI1(jj,ii) =  (1- abs(zH(jj,ti)).^2) .* (abs(zH(jj,ti)) - abs(zHt(jj,ti)).* ...
                             cos(angle(zH(jj,ti)) - angle(zHt(jj,ti)))) .* abs((xin(ii,ti))) .* ...
                             cos(angle(zH(jj,ti)) - angle((xin(ii,ti))) - angle(wHI(jj,ii))) - ...
                             abs(zH(jj,ti)) .* abs(zHt(jj,ti)) .* sin(angle(zH(jj,ti)) - angle(zHt(jj,ti))) .* ...
                             (abs((xin(ii,ti))) ./ (abs(uHI(jj,ti)))).* ...
                             sin(angle(zH(jj,ti)) - angle((xin(ii,ti))) - angle(wHI(jj,ii)));
                  end
     
              end
            end
       


            for jj = 1:sizeH    
              for ii=1:sizeI

                  if ii==1
                    deltaEwHI2(jj,ii)=0;
                  else
                    deltaEwHI2(jj,ii) =  (1- abs(zH(jj,ti)).^2) .* (abs(zH(jj,ti)) - abs(zHt(jj,ti)) .* ...
                             cos(angle(zH(jj,ti)) - angle(zHt(jj,ti)))) .* abs((xin(ii,ti))) .* ...
                             sin(angle(zH(jj,ti)) - angle((xin(ii,ti))) - angle(wHI(jj,ii))) + ...
                             abs(zH(jj,ti)) .* abs(zHt(jj,ti)) .* sin(angle(zH(jj,ti)) - angle(zHt(jj,ti))) .* ...
                             (abs((xin(ii,ti))) ./ (abs(uHI(jj,ti)))).* ...
                             cos(angle(zH(jj,ti)) - angle((xin(ii,ti))) - angle(wHI(jj,ii)));
                  end

              end
            end


            wHI1 = abs(wHI) - k1 * deltaEwHI1;
            wOH1 = abs(wOH) - k2 * deltaEwOH1;

            wHI2 = angle(wHI) - k1 * deltaEwHI2;
            wOH2 = angle(wOH) - k2 * deltaEwOH2;

            wHI = wHI1 .* exp(1i* wHI2);
            wOH = wOH1 .* exp(1i* wOH2);
            zO_set(iter, :) = zO;  
        end
    end
        if counter==4000
            x=1;
        end
        er_matrix(counter) = er;
        counter = counter +1;

        if er<=min_er
            min_er=er;
            flag_count=1;
        else 
            flag_count=flag_count+1;
            if flag_count>=16
                k2=k2/1.2;
                k1=k1/1.2;
                flag_count=0;
                min_er=er;
            end
        end
        hint=[counter,er,k1,k2]
        if rem(counter,100)==0
            save wgt_freq22.mat wHI wOH
        end
end

end