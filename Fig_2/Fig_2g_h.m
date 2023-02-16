% code for Student-Teacher-Notebook framework,analytical curves
% Weinan Sun 10-10-2021
close all
clear all

nepoch = 2000;
learnrate = 0.005;
N_x_t = 100;
N_y_t = 1;
P=100;
M = 5000; %num of units in notebook

Et_big_early_stop = [];
Eg_big_early_stop = [];
Et_big = [];
Eg_big = [];

%SNR values sampled from log2 space
SNR_log_interval = -4:0.5:4;
SNR_vec =2.^SNR_log_interval;


Notebook_Train = (P-1)/(M-1); % Analytial solution for notebook training error, see supplementary material for derivations.

for count = 1:size(SNR_vec,2)
    disp(count)
    SNR = SNR_vec(count);
    Eg = [];
    Et = [];
    lr = learnrate;
    
    %use normalized variances
    if SNR == inf
        variance_w = 1;
        variance_e = 0;
    else
        variance_w = SNR/(SNR + 1);
        variance_e = 1/(SNR + 1);
    end
    
    alpha = P/N_x_t; % number of examples divided by input dimension
    
    % Analytical curves for training and testing errors, see supplementary
    % material for derivations

    for t = 1:1:nepoch
        
        train = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi)  ).*  (  lam.*variance_w + variance_e  ).*exp(-2*lam.*t./(1./lr)) ;
        Et = [Et (1/alpha)*(integral(train,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)*(1 - alpha)* variance_e ) + (1-1/alpha)*variance_e];
        
        test = @(lam) ( ( ( ((alpha^0.5+1).^2 - lam) .* (lam - (alpha^0.5-1).^2)  ).^0.5) ./  (lam*2*pi) ).*(exp(-2*lam*t/(1/lr)) + ((1-exp(-lam*t/(1/lr))).^2)./(lam*SNR));
        Eg = [Eg variance_w*(integral(test,(alpha^0.5-1)^2,(alpha^0.5+1)^2) + (alpha<1)* (1 - alpha) + 1/SNR)];
        
    end
    
    % Early stopping curves
    [mm, pp] = min(Eg);
    Eg_early_stop = Eg;
    Eg_early_stop(pp+1:end) = Eg(pp);
    
    Et_early_stop = Et;
    Et_early_stop(pp+1:end) = Et(pp);
    
    % Pick better training error value and convert to memory generalization
    % scores
    better_train_no_early_stop = min(Et,ones(1,nepoch)*Notebook_Train);
    control_curve = (Et(1) -  better_train_no_early_stop)/Et(1);
    lesion_curve = (Et(1) - Et)/Et(1);
    
    
    better_train_yes_early_stop = min(Et_early_stop,ones(1,nepoch)*Notebook_Train);
    control_curve_early_stop = (Et_early_stop(1) -  better_train_yes_early_stop)/Et_early_stop(1);
    lesion_curve_early_stop = (Et_early_stop(1) - Et_early_stop)/Et_early_stop(1);
    
    
    control_Eg_curve = (Eg(1) -  Eg)/Eg(1);
    control_Eg_curve_early_stop = (Eg_early_stop(1) -  Eg_early_stop)/Eg_early_stop(1);
    
    
    
    figure(1)
    
    hold on;
    plot(1:1:nepoch,Et,'-','color',[0 0 1 count/20],'LineWidth',2)
    plot(1:1:nepoch,Eg,'-','color',[1 0 0 count/20],'LineWidth',2)
    
    set(gca, 'FontSize', 12)
    set(gca, 'FontSize', 12)
    xlabel('Epoch')
    ylabel('Error')
    set(gca,'linewidth',1.5)
    ylim([0 2.5])

    figure(2)
    
    hold on;
    plot(1:1:nepoch,Et_early_stop,'-','color',[0 0 1 count/20],'LineWidth',2)
    plot(1:1:nepoch,Eg_early_stop,'-','color',[1 0 0 count/20],'LineWidth',2)
    
    set(gca, 'FontSize',12)
    set(gca, 'FontSize', 12)
    xlabel('Epoch')
    ylabel('Error')
    set(gca,'linewidth',1.5)
    ylim([0 2.5])
    
    
end

%% save figures
% figure(1)
% set(gcf,'position',[100,100,350,290])
% saveas(gcf,strcat('Fig_2g','.pdf'));
% figure(2)
% set(gcf,'position',[100,100,350,290])
% saveas(gcf,strcat('Fig_2h','.pdf'));



