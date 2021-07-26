
function [zO_set,signal] = Copy_of_doa_cvnn_module_test_CP10 (zI_set, wHI,wOH)
len=size(zI_set,3);
N_ang=size(zI_set,1);
sizeI=size(zI_set,2);
sizeH=12;
sizeO=1;
TDL_n=29;
fc=500e3;
Ts=1e-7;
fs=1/Ts;
df=fs/len;
factor=exp(-1i*2*pi*(0:len-1)*df*Ts);

%%

for iter=1:N_ang
    %TDL output/hidden layer output
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
%             yOH(j,:)=tanh(abs(uOH(j,:))).*exp(1i*angle(uOH(j,:)));
            yOH(j,:)=uj;
        end


        temp = yOH*yOH';
        zO_set(iter) = temp;  
        signal(iter,:)=yOH;
end



end




