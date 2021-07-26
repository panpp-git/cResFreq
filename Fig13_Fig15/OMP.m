function [A]=OMP(D,X,L)
%=============================================
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% ||X-DA||
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).
%       X - the signals to represent
%       L - the max. number of coefficients for each signal.
% output arguments: 
%       A - sparse coefficient matrix.
%=============================================
[n,P]=size(X);
[n,K]=size(D);
for k=1:1:P,
    a=[];
    x=X(:,k);                            %the kth signal sample
    residual=x;                        %initial the residual vector
    indx=zeros(L,1);                %initial the index vector

     %the jth iter
    for j=1:1:L,
        
        %compute the inner product
        proj=D'*residual;            
        
        %find the max value and its index
        [maxVal,pos]=max(abs(proj));

        %store the index
        pos=pos(1);
        indx(j)=pos;                    
        
        %solve the Least squares problem.
        a=pinv(D(:,indx(1:j)))*x;    

        %compute the residual in the new dictionary
        residual=x-D(:,indx(1:j))*a;    

        
%the precision is fill our demand.
%         if sum(residual.^2) < 1e-6
%             break;
%         end
    end;
    temp=zeros(K,1);
    temp(indx(1:j))=a;
    A(:,k)=sparse(temp);
end;
return;