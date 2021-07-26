function [P_ah]=cvnn_func()
    n_tgt = double(h5read('tgt_num.h5','/tgt_num'));
    sig = (h5read('signal.h5','/signal'));
    sig_real=sig(:,1).';
    sig_imag=sig(:,2).';
    noisedSig=sig_real+1j*sig_imag;
    Ns=1;
    Nsnap=8;
    mv=max(abs(noisedSig));
    RPP=noisedSig/mv;
    t=0:63;
    nfft=4096;
    len=size(RPP,2);
    snapLen=len-Nsnap+1;
    freqs = -0.5:1/nfft:0.5-1/nfft;
    final_ret=zeros(size(RPP,1),length(freqs));
    for indix=1:size(RPP,1)
    % for indix=1:1
        ss=RPP(indix,:);
        zI=zeros(Ns,Nsnap,snapLen);
        for si=1:Ns
            for i=1:Nsnap
                zI(si,i,:)=ss(si,i:i+snapLen-1);
            end
        end
        zO_teach_set=zeros(Ns,snapLen);
        zI_set=zI;
        [wHI, wOH, zO_set_train] = Copy_of_doa_cvnn_module_CP10(zI_set, zO_teach_set);


        zI=zeros(length(freqs),Nsnap,snapLen);
        for tgt=1:length(freqs)
            zIk=exp(1i*(2*pi*freqs(tgt)*t));
            zIk=zIk/max(abs(zIk));
            for i=1:Nsnap
                zI(tgt,i,:)=zIk(i:i+snapLen-1);
            end
        end

        load wgt_freq22.mat
        zI_set = zI;
        [zO_set_test,signal] = Copy_of_doa_cvnn_module_test_CP10(zI,wHI,wOH);
        final_ret(indix,:)=ones(1,nfft)./(zO_set_test+1e-10);
        P_ah(indix,:)=final_ret(indix,:)/max(abs(final_ret(indix,:)));
    end
end
