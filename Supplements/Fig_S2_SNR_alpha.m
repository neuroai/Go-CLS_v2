%Generalization and memorization performance as a
%function of alpha and SNR.

close all; clear all; clc

nepoch = 10000;
lr = 0.01;
SNR_log_interval = -2:0.1:3;
SNR_vec =10.^SNR_log_interval;
alpha_vec= 0.1:0.1:5;

Et_ES = zeros(length(SNR_vec),length(alpha_vec),nepoch);
Eg_ES = zeros(length(SNR_vec),length(alpha_vec),nepoch);
Eg_no_ES = zeros(length(SNR_vec),length(alpha_vec),nepoch);
Et_no_ES = zeros(length(SNR_vec),length(alpha_vec),nepoch);

for n = 1:length(SNR_vec)
    for m = 1:length(alpha_vec)
        SNR = SNR_vec(n);
        alpha = alpha_vec(m);
        
        if SNR == inf
            variance_w = 1;
            variance_e = 0;
        else
            variance_w = SNR/(SNR + 1);
            variance_e = 1/(SNR + 1);
        end
        
        
        parfor t = 1:nepoch
            
            
            train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
            Et_no_ES(n,m,t) = (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e;
            
            test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
            Eg_no_ES(n,m,t) = variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR);
            
            
        end
        
        [min_Eg, ES_time] = min(Eg_no_ES(n,m,:));
        
        Eg_ES(n,m,:) = Eg_no_ES(n,m,:);
        Eg_ES(n,m,ES_time+1:end) = Eg_no_ES(n,m,ES_time);
        
        Et_ES(n,m,:) = Et_no_ES(n,m,:);
        Et_ES(n,m,ES_time+1:end) = Et_no_ES(n,m,ES_time);
    end
    
end

figure(1)
imagesc(flip((Et_no_ES(:,:,1)-Et_no_ES(:,:,end))./Et_no_ES(:,:,1)))
colormap(redblue)
caxis([-1 1])
colorbar
set(gcf,'position',[500 500 440 325])

figure(2)
imagesc(flip((Eg_no_ES(:,:,1)-Eg_no_ES(:,:,end))./Eg_no_ES(:,:,1)))
colormap(redblue)
caxis([-1 1])
colorbar
set(gcf,'position',[500 500 440 325])

figure(3)
imagesc(flip((Et_ES(:,:,1)-Et_ES(:,:,end))./Et_ES(:,:,1)))
colormap(redblue)
caxis([-1 1])
colorbar
set(gcf,'position',[500 500 440 325])

figure(4)
imagesc(flip(Eg_ES(:,:,1)-Eg_ES(:,:,end))./Eg_ES(:,:,1))
colormap(redblue)
caxis([-1 1])
colorbar
set(gcf,'position',[500 500 440 325])


