%Generalization and memorization performance as a
%function of alpha and SNR, for student-only, notebook-only, and combined
%system

close all; clear all; clc

%% Batch

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

%% Optimal online student

Eg_opt_online = zeros(length(SNR_vec),length(alpha_vec));



parfor n = 1:length(SNR_vec)
        SNR = SNR_vec(n);
        variance_e = 1/(SNR + 1);
        [alpha, Eg] =  ode45(@(alpha,Eg)eg_prime(alpha,Eg,variance_e),(0.1:0.1:5),1);
        Eg_new = interp1(alpha,Eg,alpha_vec);  
        Eg_opt_online(n,:) = Eg_new;
     
end

figure (1)
hold on
plot(alpha_vec,Eg_opt_online(end,:),'b-')
plot(alpha_vec,Eg_ES(end,:,end),'r-')
plot(alpha_vec,ones(1,length(alpha_vec))*2,'m')
ylim([-0.05 2.1])
set(gcf,'position',[500 500 420 420])
ax = gca;
ax.XTick = [0 1 2 3 4 5];
ax.YTick = [0 0.5 1 1.5 2];
%saveas(gcf,'Fig_4d.pdf');

figure (2)
imagesc(flip(Eg_opt_online(:,:) - Eg_ES(:,:,end)))
colormap(redblue)
caxis([-0.3 0.3])
colorbar
set(gcf,'position',[500 500 420 325])
%saveas(gcf,'Fig_4e.pdf');

figure (3)
hold on
plot(alpha_vec,Eg_no_ES(25,:,end),'b-')
plot(alpha_vec,Eg_ES(25,:,end),'r-')
ylim([0 2.5])
set(gcf,'position',[500 500 420 420])
%saveas(gcf,'Fig_4f.pdf');

figure (4)
imagesc(flip(Eg_ES(:,:,end)-Eg_no_ES(:,:,end)))
colormap(redblue)
caxis([-1 1])
colorbar
set(gcf,'position',[500 500 420 325])
%saveas(gcf,'Fig_4g.pdf');

function dEg = eg_prime(alpha,Eg,variance_e)
         dEg = 2*variance_e - Eg - variance_e^2/Eg;
end





